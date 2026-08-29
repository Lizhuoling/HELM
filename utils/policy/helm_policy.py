# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from utils.policy.policy_core import HelmPolicyBase


class HelmPolicy(HelmPolicyBase):
    """HELM policy for producing robot control-signal chunks."""

    def __init__(self, *args, merge_lora: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not getattr(self.model.config, "use_wam_vla_lora", False):
            raise ValueError(
                "Loaded checkpoint is not HELM-enabled: "
                "model.config.use_wam_vla_lora is false."
            )
        wam_module = getattr(self.model, "wam_context_module", None)
        if wam_module is None:
            raise RuntimeError(
                "Loaded HELM model has no WAM context module."
            )
        if not hasattr(wam_module.dit, "peft_config"):
            raise RuntimeError("Loaded HELM checkpoint has no DreamZero PEFT adapters.")
        self.lora_merged = False
        if merge_lora:
            wam_module.merge_lora_for_inference()
            self.lora_merged = True
