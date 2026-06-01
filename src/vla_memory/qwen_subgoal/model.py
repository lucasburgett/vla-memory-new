"""Qwen3-VL-4B-Instruct subgoal-prediction policy with LoRA.

Wraps HuggingFace ``Qwen3VLForConditionalGeneration`` with PEFT LoRA so we can:

- Sample K subgoal candidates per state at high temperature (GRPO rollout).
- Compute per-token log-probs for the sampled subgoals (GRPO loss).
- Save / load the adapter so it plugs into the existing
  ``QwenVLSubgoalPredictor`` in ``robomme_policy_learning`` for evaluation.

The base model is shared between policy and reference: we use ``peft``'s
adapter-switching to run the reference forward pass with the SFT adapter and
the policy forward pass with the trainable adapter, keeping a single set of
base weights in GPU memory.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .prompts import build_messages, parse_subgoal_output


# Inference / sampling guards: keep image and video token budgets identical to
# how the existing eval pipeline runs the model. The Qwen processor reads these
# from env vars at construction time.
import os
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")


_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


@dataclasses.dataclass
class SampleResult:
    """One sampled subgoal candidate plus the bookkeeping GRPO needs."""

    text: str                    # raw decoded JSON response from the VLM
    subtask: str                 # parsed current_subtask — the grounded subgoal sent to π0.5
    keyframe_positions: List[int]  # parsed 1-indexed keyframe nominations (learned-selection path)
    token_ids: torch.Tensor      # (T,) int64 — generated tokens only
    logprobs: torch.Tensor       # unused placeholder; GRPO loss recomputes logp with grad
    prompt_input_ids: torch.Tensor      # (P,) int64 — to recompute logp later
    prompt_attention_mask: torch.Tensor  # (P,) int64
    pixel_values: Optional[torch.Tensor] # vision tower input (image grid)
    image_grid_thw: Optional[torch.Tensor]


def _first_stop_index(ids: torch.Tensor, stop_ids: Sequence[int]) -> int:
    """Index of the first token in ``ids`` matching any id in ``stop_ids``.

    Returns ``len(ids)`` if none match. The SFT'd Qwen3-VL terminates its turn
    with ``<|im_end|>``, which is NOT always the tokenizer's nominal
    ``eos_token_id`` (that can be ``<|endoftext|>``). Trimming on a single id
    therefore missed the real stop and let the decoded subgoal keep ``<|im_end|>``
    plus whatever the model rambled afterward — the v1 "generations never
    terminate" bug. Trim on the full stop set instead.
    """
    if not stop_ids:
        return int(ids.numel())
    stop = torch.as_tensor(list(stop_ids), device=ids.device, dtype=ids.dtype)
    hits = torch.isin(ids, stop).nonzero(as_tuple=False)
    return int(hits[0].item()) if hits.numel() > 0 else int(ids.numel())


def _looks_degenerate(token_ids: torch.Tensor, max_new_tokens: int) -> bool:
    """Heuristic: did greedy decoding collapse (the v4 SFT failure mode)?

    A healthy subgoal is short and terminates. Collapse looks like (a) running the
    whole ``max_new_tokens`` budget without emitting EOS, or (b) one token
    dominating the output (e.g. 'Waitwaitwait...'). Either flags a degenerate
    adapter. See ``project_sft_v4_adapter_degenerate``.
    """
    n = int(token_ids.numel())
    if n == 0:
        return True
    if n >= max_new_tokens:          # ran the whole budget → never stopped
        return True
    if n >= 4:
        _, counts = token_ids.unique(return_counts=True)
        if int(counts.max()) / n > 0.5:   # a single token is >half the output
            return True
    return False


class QwenSubgoalPolicy:
    """Policy + reference subgoal predictor backed by Qwen3-VL with LoRA.

    Designed for GRPO training. NOT meant for full-eval inference — for that
    use the existing ``Qwen3VLModel`` in the submodule which leans on ms-swift
    and is already battle-tested.
    """

    def __init__(
        self,
        adapter_init_path: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        # Imports are local so that simply importing this module on a CPU dev
        # machine doesn't drag in transformers / PEFT.
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )
        from peft import LoraConfig, PeftModel, get_peft_model

        self.device = device
        self.dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        # ``sdpa`` (PyTorch's native scaled-dot-product attention) matches what
        # the submodule's SFT recipe uses and avoids the flash-attn install
        # requirement (nvcc isn't in the CUDA runtime base image). ~20-30%
        # slower than flash_attention_2 but numerically identical and works on
        # any GPU.
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            device_map=device,
            trust_remote_code=True,
        )
        base.config.use_cache = True

        if adapter_init_path is not None:
            # Warm-start the policy from the SFT'd LoRA adapter, and load a
            # frozen copy as the reference policy for KL.
            self.model = PeftModel.from_pretrained(
                base, adapter_init_path, adapter_name="policy", is_trainable=True
            )
            self.model.load_adapter(adapter_init_path, adapter_name="reference")
            self.has_reference = True
        else:
            # Cold-start: train from base + fresh LoRA, no separate reference.
            # (Use this only for ablations — GRPO without a reference is unstable.)
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora_cfg, adapter_name="policy")
            self.has_reference = False

        self.model.set_adapter("policy")

        # Build the explicit stop-token set. The SFT model ends its turn with
        # <|im_end|>; the tokenizer's nominal eos_token_id may be <|endoftext|>.
        # We must (a) pass these to generate() — relying on
        # generation_config.eos_token_id silently failed when the PEFT load left
        # it unset, so generations ran to max_new_tokens forever (v1 no-EOS bug),
        # and (b) trim on the same set in sample_subgoals.
        tok = self.processor.tokenizer
        stop_ids: List[int] = []
        if tok.eos_token_id is not None:
            stop_ids.append(int(tok.eos_token_id))
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if (
            isinstance(im_end_id, int)
            and im_end_id >= 0
            and im_end_id != getattr(tok, "unk_token_id", None)
            and im_end_id not in stop_ids
        ):
            stop_ids.append(im_end_id)
        if not stop_ids:
            raise ValueError(
                "No stop token resolved for generation: tokenizer has no "
                "eos_token_id and '<|im_end|>' is not in the vocab. Subgoals "
                "would never terminate."
            )
        self._eos_token_id = stop_ids[0]   # pad_token_id / back-compat
        self._stop_token_ids = stop_ids    # full set for generate() + trim

    # ------------------------------------------------------------------
    # Input prep
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        key_frames: Sequence[np.ndarray],
        recent_frames: Sequence[np.ndarray],
        task_goal: str,
        has_video_demo: bool,
        history_subgoals: Optional[List[str]] = None,
        debug: bool = False,
        mode: str = "use",
    ) -> dict:
        """Build a multi-image MemER prompt.

        ``key_frames`` (the remembered past) are placed first, then
        ``recent_frames`` (current execution context) — the same order the
        prompt's ``<image>`` placeholders appear, which the Qwen processor pairs
        positionally with the ``images`` list. At least one frame is required.
        ``history_subgoals`` carries the completed-subtask timing signal.
        """
        frames = list(key_frames) + list(recent_frames)
        if not frames:
            raise ValueError(
                "MemER prompt needs at least one frame, but key_frames and "
                "recent_frames are both empty."
            )
        messages = build_messages(
            task_goal=task_goal,
            n_key_frames=len(key_frames),
            n_recent_frames=len(recent_frames),
            history_subgoals=history_subgoals,
            has_video_demo=has_video_demo,
            mode=mode,
        )

        # Convert each numpy HxWx3 uint8 frame to PIL, preserving key→recent order.
        pil_images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # The SFT dataset builder (`build_vlm_subgoal_dataset_memer.py`) emits
        # the literal strings ``<image>`` / ``<video>``. ms-swift expands those
        # to the Qwen3-VL vision-token sequence before tokenizing during SFT.
        # We replicate that substitution here so the HF processor sees the
        # tokens it actually recognizes (``<|vision_start|><|image_pad|>
        # <|vision_end|>``); without it, ``image_grid_thw`` carries the images'
        # patches but ``input_ids`` has no vision-start, and generate's
        # ``_expand_inputs_for_generation`` crashes with
        # ``split_with_sizes ... split_sizes=[0]``. ``replace`` hits every
        # ``<image>`` (one per frame), so the count stays in sync with ``images``.
        chat_text = chat_text.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        ).replace(
            "<video>", "<|vision_start|><|video_pad|><|vision_end|>"
        )

        inputs = self.processor(
            text=[chat_text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        if debug:
            iid = inputs["input_ids"][0]
            pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            n_img = int((iid == pad_id).sum()) if isinstance(pad_id, int) and pad_id >= 0 else -1
            pv = inputs.get("pixel_values")
            thw = inputs.get("image_grid_thw")
            shapes = [getattr(f, "shape", None) for f in frames]
            print(
                f"[qwen][debug] n_frames={len(frames)} "
                f"(key={len(key_frames)} recent={len(recent_frames)}) shapes={shapes} | "
                f"n_image_pad_tokens={n_img} prompt_len={int(iid.numel())} "
                f"pixel_values={tuple(pv.shape) if pv is not None else None} "
                f"grid_thw={thw.tolist() if thw is not None else None}",
                flush=True,
            )
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Sampling (rollout time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_subgoals(
        self,
        key_frames: Sequence[np.ndarray],
        recent_frames: Sequence[np.ndarray],
        task_goal: str,
        history_subgoals: Optional[List[str]] = None,
        k: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.95,
        has_video_demo: bool = False,
        debug: bool = False,
        mode: str = "use",
    ) -> List[SampleResult]:
        """Draw ``k`` subgoal candidates for one memory state. Used for GRPO rollouts.

        Conditions on ``key_frames`` (the remembered past) + ``recent_frames``
        (current execution context) + ``history_subgoals`` (completed-subtask
        timing). Each candidate's raw JSON response is parsed into ``subtask`` (the
        grounded subgoal handed to π0.5) and ``keyframe_positions``.
        ``max_new_tokens`` defaults to 128 because the MemER target is a JSON
        object, not a bare phrase.
        """
        self.model.set_adapter("policy")
        self.model.eval()

        inputs = self._prepare_inputs(
            key_frames=key_frames,
            recent_frames=recent_frames,
            task_goal=task_goal,
            has_video_demo=has_video_demo,
            history_subgoals=history_subgoals,
            debug=debug,
            mode=mode,
        )

        prompt_len = inputs["input_ids"].shape[1]
        # `num_return_sequences` lets us sample K candidates from the same prompt
        # in one forward — but with vision inputs we have to expand them too.
        gen_out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            return_dict_in_generate=True,
            pad_token_id=self._eos_token_id,
            # Pass the stop set explicitly — do NOT rely on
            # generation_config.eos_token_id (unset after the PEFT load → the
            # model never stopped: the v1 64-token-ramble bug).
            eos_token_id=self._stop_token_ids,
        )

        if debug:
            # Adapter-on-vs-off probe: greedy-decode the BASE model (adapter
            # disabled) on the SAME input. If BASE is coherent but the adapter
            # output is "Wait..." garbage → the SFT adapter is the problem; if
            # BASE is ALSO garbage → the base load / forward path is. Splits the two.
            try:
                with self.model.disable_adapter():
                    base_full = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        pad_token_id=self._eos_token_id,
                        eos_token_id=self._stop_token_ids,
                    )
                base_ids = base_full[0][prompt_len:]
                base_cut = _first_stop_index(base_ids, self._stop_token_ids)
                base_txt = self.processor.tokenizer.decode(
                    base_ids[:base_cut], skip_special_tokens=False
                ).strip()
                print(
                    f"[qwen][debug] BASE(no-adapter) greedy ntok={int(base_cut)} "
                    f"subgoal={base_txt!r}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[qwen][debug] base-adapter probe failed: {exc!r}", flush=True)

        sequences = gen_out.sequences  # (k, prompt_len + new_len)
        gen_ids = sequences[:, prompt_len:]
        # NOTE: we intentionally do NOT request output_scores. Sample-time
        # log-probs aren't used downstream — the GRPO loss recomputes them with
        # gradient in policy_logprobs — and output_scores would materialize a
        # (k, new_len, vocab) logits tensor per rollout (the costly bit).

        results: List[SampleResult] = []
        for i in range(k):
            # Trim at the first stop token (any id in the stop set) to avoid
            # scoring pad / post-turn tokens. Searching the full set — not just
            # self._eos_token_id — is what catches <|im_end|>.
            ids_i = gen_ids[i]
            cut = _first_stop_index(ids_i, self._stop_token_ids)
            # ``skip_special_tokens=False`` keeps Qwen3-VL's ``<|box_start|>`` /
            # ``<|box_end|>`` markers in the decoded string — grounded_subgoal
            # mode needs them. We trim the trailing EOS ourselves via ``cut``.
            text = self.processor.tokenizer.decode(
                ids_i[:cut], skip_special_tokens=False
            ).strip()
            subtask, keyframe_positions = parse_subgoal_output(text)

            results.append(
                SampleResult(
                    text=text,
                    subtask=subtask,
                    keyframe_positions=keyframe_positions,
                    token_ids=ids_i[:cut].detach().cpu(),
                    logprobs=torch.empty(0),  # unused; loss recomputes logp with grad
                    prompt_input_ids=inputs["input_ids"][0].detach().cpu(),
                    prompt_attention_mask=inputs["attention_mask"][0].detach().cpu(),
                    pixel_values=inputs.get("pixel_values", torch.empty(0)).detach().cpu(),
                    image_grid_thw=inputs.get("image_grid_thw", torch.empty(0)).detach().cpu(),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Greedy decode (eval / validation / probe set)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy_subgoal(
        self,
        key_frames: Sequence[np.ndarray],
        recent_frames: Sequence[np.ndarray],
        task_goal: str,
        history_subgoals: Optional[List[str]] = None,
        has_video_demo: bool = False,
        max_new_tokens: int = 128,
        mode: str = "use",
    ) -> Tuple[str, torch.Tensor]:
        """Deterministically greedy-decode ONE response. For eval / probe / validation.

        Unlike ``sample_subgoals`` (which samples K candidates for RL exploration),
        this is argmax decoding — the cleanest signal for "did the adapter collapse"
        and the right decode for a held-out probe metric. Returns the RAW decoded
        text (parse with ``parse_subgoal_output``) and ``gen_token_ids`` (pass to
        ``_looks_degenerate``).
        """
        self.model.set_adapter("policy")
        self.model.eval()
        inputs = self._prepare_inputs(
            key_frames=key_frames,
            recent_frames=recent_frames,
            task_goal=task_goal,
            has_video_demo=has_video_demo,
            history_subgoals=history_subgoals,
            mode=mode,
        )
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self._eos_token_id,
            eos_token_id=self._stop_token_ids,
        )
        ids = out[0][prompt_len:]
        cut = _first_stop_index(ids, self._stop_token_ids)
        gen_ids = ids[:cut].detach().cpu()
        text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
        return text, gen_ids

    # ------------------------------------------------------------------
    # Log-prob recomputation (gradient step)
    # ------------------------------------------------------------------

    def policy_logprobs(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        gen_token_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        adapter: str = "policy",
        gradient_enabled: bool = True,
    ) -> torch.Tensor:
        """Recompute per-token log p for ``gen_token_ids`` under the chosen adapter.

        Returns a tensor of shape ``(T,)`` summed over ``gen_token_ids``.
        """
        self.model.set_adapter(adapter)
        # eval() so LoRA dropout is OFF during log-prob recomputation. In train()
        # mode the SFT adapter's dropout (0.1) makes the recomputed log-probs —
        # and hence the KL term — stochastic, injecting gradient noise and a
        # spurious nonzero KL floor even when the policy hasn't moved. Gradient
        # flow is controlled purely by the enable_grad/no_grad context below.
        self.model.eval()

        # Concatenate prompt + generated → one teacher-forced forward.
        input_ids = torch.cat([prompt_input_ids, gen_token_ids], dim=0).unsqueeze(0).to(self.device)
        attn = torch.cat(
            [prompt_attention_mask, torch.ones_like(gen_token_ids)], dim=0
        ).unsqueeze(0).to(self.device)

        prompt_len = prompt_input_ids.shape[0]

        kwargs = dict(input_ids=input_ids, attention_mask=attn)
        if pixel_values.numel() > 0:
            kwargs["pixel_values"] = pixel_values.to(self.device, dtype=self.dtype)
            kwargs["image_grid_thw"] = image_grid_thw.to(self.device)

        ctx = torch.enable_grad() if gradient_enabled and adapter == "policy" else torch.no_grad()
        with ctx:
            out = self.model(**kwargs)
        logits = out.logits[0]  # (P + T, V)

        # Score positions for gen tokens: logits at index t predict token t+1, so
        # for the first generated token (at position prompt_len) we read logits[prompt_len - 1].
        score_logits = logits[prompt_len - 1 : prompt_len - 1 + gen_token_ids.shape[0]]
        logp_all = torch.log_softmax(score_logits.float(), dim=-1)
        chosen = logp_all.gather(1, gen_token_ids.to(self.device).unsqueeze(-1)).squeeze(-1)
        return chosen

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def activate_policy(self) -> None:
        """Make the trainable 'policy' adapter the active one.

        PEFT's ``set_adapter`` toggles ``requires_grad`` so that ONLY the active
        adapter trains. After a reference (KL) forward leaves 'reference' active,
        ``trainable_parameters()`` would return the wrong (frozen) adapter and a
        gradient step would touch nothing. Call this before grad-clip / optimizer
        step so they operate on the policy adapter. See
        ``feedback_peft_set_adapter_zeroes_grad`` for the failure this prevents.
        """
        self.model.set_adapter("policy")

    def save_policy_adapter(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.model.set_adapter("policy")
        self.model.save_pretrained(out_dir, selected_adapters=["policy"])

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]


__all__ = ["QwenSubgoalPolicy", "SampleResult"]
