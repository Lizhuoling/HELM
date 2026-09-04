# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

from peft import LoraConfig, get_peft_model

from utils.model.wam_vla_lora.wam_context_module import WamDreamZeroContextModule


logger = logging.getLogger(__name__)

DEFAULT_WAM_LORA_TARGET_MODULES = (
    r"blocks\.\d+\."
    r"(self_attn\.(q|k|v|o)|cross_attn\.(q|k|v|o)|ffn\.(0|2))"
)


class WamDreamZeroLoraContextModule(WamDreamZeroContextModule):
    """WAM context module with a frozen DreamZero trunk and trainable LoRA adapters."""

    def __init__(
        self,
        *,
        wam_model_path: str | Path,
        text_embedding_cache: str | Path,
        state_dim: int,
        num_context_tokens: int,
        gr00t_backbone_dim: int,
        freeze_vae: bool = True,
        train_state_encoder: bool = True,
        train_context_tokens: bool = True,
        train_context_projector: bool = True,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_dropout: float = 0.0,
        lora_target_modules: str = DEFAULT_WAM_LORA_TARGET_MODULES,
    ) -> None:
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}.")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}.")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {lora_dropout}.")

        # The parent loads the cleaned DreamZero checkpoint before adapters are inserted.
        # This keeps its strict base-weight validation independent of LoRA tensors.
        super().__init__(
            wam_model_path=wam_model_path,
            text_embedding_cache=text_embedding_cache,
            state_dim=state_dim,
            num_context_tokens=num_context_tokens,
            gr00t_backbone_dim=gr00t_backbone_dim,
            freeze_vae=freeze_vae,
            train_dit=False,
            train_state_encoder=False,
            train_context_tokens=train_context_tokens,
            train_context_projector=train_context_projector,
        )

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.dit = get_peft_model(
            self.dit,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                init_lora_weights=True,
            ),
        )

        # PEFT freezes every non-adapter parameter. The replacement state encoder is
        # WAM-specific rather than part of the pretrained DreamZero trunk.
        self.dit.base_model.model.state_encoder.requires_grad_(train_state_encoder)

        lora_parameters = [
            (name, parameter)
            for name, parameter in self.dit.named_parameters()
            if "lora_" in name
        ]
        if not lora_parameters:
            raise RuntimeError(
                "No DreamZero modules matched wam_lora_target_modules: "
                f"{lora_target_modules!r}."
            )
        frozen_base_trainable = [
            name
            for name, parameter in self.dit.base_model.model.blocks.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if frozen_base_trainable:
            raise RuntimeError(
                "DreamZero base block parameters unexpectedly remain trainable: "
                f"{frozen_base_trainable[:20]}"
            )

        logger.info(
            "Enabled DreamZero LoRA: rank=%d, alpha=%d, dropout=%g, "
            "adapter_tensors=%d, trainable_adapter_parameters=%d",
            lora_rank,
            lora_alpha,
            lora_dropout,
            len(lora_parameters),
            sum(parameter.numel() for _, parameter in lora_parameters),
        )

    def merge_lora_for_inference(self) -> None:
        """Merge trained LoRA deltas into DreamZero weights in memory."""
        if not hasattr(self.dit, "merge_and_unload"):
            raise RuntimeError("DreamZero LoRA module does not support merge_and_unload().")
        self.dit = self.dit.merge_and_unload(safe_merge=True)
        remaining_lora = [name for name, _ in self.dit.named_parameters() if "lora_" in name]
        if remaining_lora:
            raise RuntimeError(
                "DreamZero LoRA parameters remain active after merge_and_unload(): "
                f"{remaining_lora[:20]}"
            )
        self.dit.eval()
        logger.info("Merged DreamZero LoRA adapters into base weights for inference")
