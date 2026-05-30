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

from .prompts import build_messages


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

    text: str
    token_ids: torch.Tensor      # (T,) int64 — generated tokens only
    logprobs: torch.Tensor       # unused placeholder; GRPO loss recomputes logp with grad
    prompt_input_ids: torch.Tensor      # (P,) int64 — to recompute logp later
    prompt_attention_mask: torch.Tensor  # (P,) int64
    pixel_values: Optional[torch.Tensor] # vision tower input (image grid)
    image_grid_thw: Optional[torch.Tensor]


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
        self._eos_token_id = self.processor.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Input prep
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        image: np.ndarray,
        task_goal: str,
        history_subgoals: List[str],
        subgoal_type: str,
        has_video_demo: bool,
    ) -> dict:
        messages = build_messages(
            task_goal=task_goal,
            history_subgoals=history_subgoals,
            subgoal_type=subgoal_type,
            has_video_demo=has_video_demo,
        )

        # Convert numpy HxWx3 uint8 to PIL for the processor.
        pil_image = Image.fromarray(np.ascontiguousarray(image))

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # The SFT dataset builder (`build_vlm_subgoal_dataset_qwenvl.py`) emits
        # the literal strings ``<image>`` / ``<video>``. ms-swift expands those
        # to the Qwen3-VL vision-token sequence before tokenizing during SFT.
        # We replicate that substitution here so the HF processor sees the
        # tokens it actually recognizes (``<|vision_start|><|image_pad|>
        # <|vision_end|>``); without it, ``image_grid_thw`` carries 1 image's
        # worth of patches but ``input_ids`` has no vision-start, and
        # generate's ``_expand_inputs_for_generation`` crashes with
        # ``split_with_sizes ... split_sizes=[0]``.
        chat_text = chat_text.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        ).replace(
            "<video>", "<|vision_start|><|video_pad|><|vision_end|>"
        )

        inputs = self.processor(
            text=[chat_text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # Sampling (rollout time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_subgoals(
        self,
        image: np.ndarray,
        task_goal: str,
        history_subgoals: List[str],
        k: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_p: float = 0.95,
        subgoal_type: str = "simple_subgoal",
        has_video_demo: bool = False,
    ) -> List[SampleResult]:
        """Draw ``k`` subgoal candidates for one state. Used for GRPO rollouts."""
        self.model.set_adapter("policy")
        self.model.eval()

        inputs = self._prepare_inputs(
            image=image,
            task_goal=task_goal,
            history_subgoals=history_subgoals,
            subgoal_type=subgoal_type,
            has_video_demo=has_video_demo,
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
        )

        sequences = gen_out.sequences  # (k, prompt_len + new_len)
        gen_ids = sequences[:, prompt_len:]
        # NOTE: we intentionally do NOT request output_scores. Sample-time
        # log-probs aren't used downstream — the GRPO loss recomputes them with
        # gradient in policy_logprobs — and output_scores would materialize a
        # (k, new_len, vocab) logits tensor per rollout (the costly bit).

        results: List[SampleResult] = []
        for i in range(k):
            # Stop at first EOS to avoid scoring pad tokens.
            ids_i = gen_ids[i]
            eos_pos = (ids_i == self._eos_token_id).nonzero(as_tuple=False)
            cut = eos_pos[0].item() if eos_pos.numel() > 0 else ids_i.numel()
            # ``skip_special_tokens=False`` keeps Qwen3-VL's ``<|box_start|>`` /
            # ``<|box_end|>`` markers in the decoded string — grounded_subgoal
            # mode needs them. We trim the trailing EOS ourselves via ``cut``.
            text = self.processor.tokenizer.decode(
                ids_i[:cut], skip_special_tokens=False
            ).strip()

            results.append(
                SampleResult(
                    text=text,
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
