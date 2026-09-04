# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

from safetensors.torch import load_file
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data.language_utils import load_text_embedding_cache, resolve_text_embedding_index
from utils.vla.model.dreamzero.modules.wan2_1_submodule import (
    ENABLE_TENSORRT,
    sinusoidal_embedding_1d,
)


logger = logging.getLogger(__name__)


def instantiate(config: dict[str, Any]) -> Any:
    """Minimal Hydra-style instantiation for cleaned DreamZero configs."""
    import importlib

    cfg = dict(config)
    target = cfg.pop("_target_")
    cfg.pop("_convert_", None)
    cfg.pop("_recursive_", None)
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**cfg)


class SingleStateEncoder(nn.Module):
    """State projection used by the WAM/DreamZero context module."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class WamDreamZeroContextModule(nn.Module):
    """DreamZero/Wan feature module that emits context tokens for GR00T.

    The learned context tokens are fed to the DreamZero DiT in the same register
    layout as DreamZero's action-query tokens, but they are exposed to GR00T only
    as generic context features.
    """

    def __init__(
        self,
        *,
        wam_model_path: str | Path,
        text_embedding_cache: str | Path,
        state_dim: int,
        num_context_tokens: int,
        gr00t_backbone_dim: int,
        freeze_vae: bool = True,
        train_dit: bool = True,
        train_state_encoder: bool = True,
        train_context_tokens: bool = True,
        train_context_projector: bool = True,
    ) -> None:
        super().__init__()
        self.wam_model_path = Path(wam_model_path)
        self.state_dim = state_dim
        self.num_context_tokens = num_context_tokens

        with (self.wam_model_path / "cleaned_config.json").open("r") as f:
            self.config: dict[str, Any] = json.load(f)
        runtime_cfg = self.config["diffusion_action_runtime"]
        self.dit_dim = int(runtime_cfg["dit_dim"])
        self.hidden_size = int(runtime_cfg["hidden_size"])
        self.num_frame_per_block = int(runtime_cfg["num_frame_per_block"])
        self.num_state_per_block = int(runtime_cfg["num_state_per_block"])
        if self.num_state_per_block != 1:
            raise ValueError(f"Expected one DreamZero state token, got {self.num_state_per_block}.")

        self.text_embedding_cache = load_text_embedding_cache(text_embedding_cache)
        self.vae = instantiate(self.config["vae_config"])
        self.dit = instantiate(self.config["dit_config"])
        self._load_components()
        self._replace_state_encoder()
        self._vae_dtype = torch.float32
        self.vae.to(dtype=self._vae_dtype)

        self.context_tokens = nn.Parameter(torch.randn(num_context_tokens, self.dit_dim) * 0.02)
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.dit_dim),
            nn.Linear(self.dit_dim, gr00t_backbone_dim),
        )

        self.vae.requires_grad_(not freeze_vae)
        self.dit.requires_grad_(train_dit)
        self.dit.state_encoder.requires_grad_(train_state_encoder)
        self.context_tokens.requires_grad_(train_context_tokens)
        self.context_projector.requires_grad_(train_context_projector)

        for module_name in ("action_encoder", "action_decoder", "head"):
            module = getattr(self.dit, module_name, None)
            if module is not None:
                module.requires_grad_(False)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.vae.to(dtype=self._vae_dtype)
        return result

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "_vae_dtype"):
            self.vae.to(dtype=self._vae_dtype)
        return result

    def _component_path(self, key: str) -> Path:
        rel = self.config["kept_components"][key]
        if rel is None:
            raise ValueError(f"Cleaned checkpoint does not contain component {key}.")
        return self.wam_model_path / rel

    @staticmethod
    def _assert_no_meta_parameters(module: nn.Module, module_name: str) -> None:
        meta_keys = [
            name for name, parameter in module.named_parameters() if parameter.device.type == "meta"
        ]
        if meta_keys:
            raise RuntimeError(
                f"{module_name} still has meta parameters after checkpoint load: "
                f"{meta_keys[:20]} (total={len(meta_keys)})"
            )

    @staticmethod
    def _set_module_tensor(module: nn.Module, key: str, value: torch.Tensor, template: torch.Tensor) -> None:
        module_path, tensor_name = key.rsplit(".", 1) if "." in key else ("", key)
        owner = module.get_submodule(module_path) if module_path else module
        value = value.detach()
        if tensor_name in owner._parameters:
            owner._parameters[tensor_name] = nn.Parameter(  # noqa: SLF001
                value,
                requires_grad=template.requires_grad,
            )
        elif tensor_name in owner._buffers:
            owner._buffers[tensor_name] = value  # noqa: SLF001
        else:
            raise KeyError(f"Could not find parameter or buffer {key!r} in {module.__class__.__name__}.")

    @classmethod
    def _load_state_dict_materializing(
        cls,
        module: nn.Module,
        state: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        current = module.state_dict(keep_vars=True)
        missing = [key for key in current if key not in state]
        unexpected = [key for key in state if key not in current]
        mismatched = []
        for key, value in state.items():
            if key not in current:
                continue
            template = current[key]
            if tuple(value.shape) != tuple(template.shape):
                mismatched.append((key, tuple(value.shape), tuple(template.shape)))
                continue
            cls._set_module_tensor(module, key, value, template)
        if mismatched:
            raise RuntimeError(f"Checkpoint tensor shape mismatch: {mismatched[:10]}")
        return missing, unexpected

    @classmethod
    def _materialize_expected_meta(
        cls,
        module: nn.Module,
        allowed_prefixes: tuple[str, ...],
    ) -> None:
        for name, parameter in list(module.named_parameters()):
            if parameter.device.type == "meta" and name.startswith(allowed_prefixes):
                cls._set_module_tensor(module, name, torch.zeros_like(parameter, device="cpu"), parameter)
        for name, buffer in list(module.named_buffers()):
            if buffer.device.type == "meta" and name.startswith(allowed_prefixes):
                cls._set_module_tensor(module, name, torch.zeros_like(buffer, device="cpu"), buffer)

    def _load_components(self) -> None:
        dit_state = load_file(self._component_path("dit"))
        missing, unexpected = self.dit.load_state_dict(dit_state, strict=False, assign=True)
        if any(parameter.device.type == "meta" for parameter in self.dit.parameters()):
            missing, unexpected = self._load_state_dict_materializing(self.dit, dit_state)
        risky_missing = [
            key
            for key in missing
            if not key.startswith(("state_encoder.", "action_encoder.", "action_decoder.", "head."))
        ]
        if risky_missing or unexpected:
            raise RuntimeError(
                f"DreamZero DiT load mismatch: risky_missing={risky_missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        logger.info(
            "Verified WAM DiT weights from %s; required_missing=0, unexpected=0, "
            "replaced_or_unused=%d",
            self._component_path("dit"),
            len(missing),
        )
        self._materialize_expected_meta(
            self.dit,
            ("state_encoder.", "action_encoder.", "action_decoder.", "head."),
        )
        self._assert_no_meta_parameters(self.dit, "WAM DiT")

        vae_state = load_file(self._component_path("vae"))
        missing, unexpected = self.vae.model.load_state_dict(vae_state, strict=False, assign=True)
        if any(parameter.device.type == "meta" for parameter in self.vae.model.parameters()):
            missing, unexpected = self._load_state_dict_materializing(self.vae.model, vae_state)
        unused_decoder_prefixes = ("conv2.", "decoder.")
        risky_missing = [key for key in missing if not key.startswith(unused_decoder_prefixes)]
        if risky_missing or unexpected:
            raise RuntimeError(
                f"WAM VAE load mismatch: risky_missing={risky_missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        logger.info(
            "Verified WAM VAE encoder weights from %s; required_missing=0, unexpected=0, "
            "unused_decoder=%d",
            self._component_path("vae"),
            len(missing),
        )
        self._materialize_expected_meta(self.vae.model, unused_decoder_prefixes)
        self._assert_no_meta_parameters(self.vae.model, "WAM VAE")

    def _replace_state_encoder(self) -> None:
        self.dit.max_state_dim = self.state_dim
        self.dit.state_encoder = SingleStateEncoder(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dit_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.vae.parameters()):
            self.vae.eval()
        return self

    @staticmethod
    def _raise_if_nonfinite(name: str, value: torch.Tensor) -> None:
        if not torch.is_floating_point(value):
            return
        detached = value.detach()
        if torch.isfinite(detached).all():
            return
        finite = torch.isfinite(detached)
        finite_values = detached[finite].float()
        if finite_values.numel() > 0:
            min_val = finite_values.min().item()
            max_val = finite_values.max().item()
        else:
            min_val = float("nan")
            max_val = float("nan")
        raise FloatingPointError(
            f"Non-finite WAM tensor after {name}: "
            f"shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"nan_count={torch.isnan(detached).sum().item()}, "
            f"inf_count={torch.isinf(detached).sum().item()}, "
            f"finite_min={min_val:.6g}, finite_max={max_val:.6g}"
        )

    def _resolve_text_features(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [resolve_text_embedding_index(self.text_embedding_cache, text) for text in texts]
        text_features = self.text_embedding_cache["text_features"][indices].to(device=device, dtype=dtype)
        attention_mask = self.text_embedding_cache["attention_mask"][indices].to(device=device)
        return text_features, attention_mask

    @staticmethod
    def _rope_context_state_apply(
        x: torch.Tensor,
        *,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        context_register_length: int,
        num_context_per_block: int,
        num_state_per_block: int,
    ) -> torch.Tensor:
        block_width = num_context_per_block + num_state_per_block
        if context_register_length % block_width != 0:
            raise ValueError(
                f"Context register length {context_register_length} is not divisible by "
                f"register block width {block_width}."
            )
        chunk_size = context_register_length // block_width

        if ENABLE_TENSORRT:
            freqs_1d_context = freqs_action[: chunk_size * num_context_per_block]
            freqs_1d_state = freqs_state[: chunk_size * num_state_per_block]
            freqs = torch.cat([freqs, freqs_1d_context, freqs_1d_state], dim=0)
            freqs = freqs.unsqueeze(0).unsqueeze(2)
            x0, x1 = x.chunk(2, dim=-1)
            freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)
            return torch.cat(
                (
                    x0 * freqs_cos - x1 * freqs_sin,
                    x1 * freqs_cos + x0 * freqs_sin,
                ),
                dim=-1,
            )

        batch_size, seq_len, num_heads, _ = x.shape
        x_complex = torch.view_as_complex(
            x.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
        )
        freqs_1d_context = freqs_action[: chunk_size * num_context_per_block].view(
            chunk_size * num_context_per_block,
            1,
            -1,
        )
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block,
            1,
            -1,
        )
        freqs = torch.cat([freqs, freqs_1d_context, freqs_1d_state], dim=0)
        return torch.view_as_real(x_complex * freqs.unsqueeze(0)).flatten(3)

    def _forward_block_memory(
        self,
        block: nn.Module,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        context_register_length: int,
    ) -> torch.Tensor:
        e_chunks = (block.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        self_attn_input = block.norm1(x) * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)
        bsz, seq_len = self_attn_input.shape[:2]
        num_heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim
        q = block.self_attn.norm_q(block.self_attn.q(self_attn_input)).view(
            bsz, seq_len, num_heads, head_dim
        )
        k = block.self_attn.norm_k(block.self_attn.k(self_attn_input)).view(
            bsz, seq_len, num_heads, head_dim
        )
        v = block.self_attn.v(self_attn_input).view(bsz, seq_len, num_heads, head_dim)
        q = self._rope_context_state_apply(
            q,
            freqs=freqs,
            freqs_action=self.dit.freqs_action,
            freqs_state=self.dit.freqs_state,
            context_register_length=context_register_length,
            num_context_per_block=self.num_context_tokens,
            num_state_per_block=self.num_state_per_block,
        ).type_as(v)
        k = self._rope_context_state_apply(
            k,
            freqs=freqs,
            freqs_action=self.dit.freqs_action,
            freqs_state=self.dit.freqs_state,
            context_register_length=context_register_length,
            num_context_per_block=self.num_context_tokens,
            num_state_per_block=self.num_state_per_block,
        ).type_as(v)
        y = block.self_attn.attn(q, k, v).flatten(2)
        x = x + block.self_attn.o(y) * e_chunks[2].squeeze(2)
        x = x + block.cross_attn(block.norm3(x), context, context_lens)
        y = block.ffn(block.norm2(x) * (1 + e_chunks[4].squeeze(2)) + e_chunks[3].squeeze(2))
        return x + y * e_chunks[5].squeeze(2)

    def forward(
        self,
        *,
        images: torch.Tensor,
        texts: list[str],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected WAM context tokens with shape [B, N, gr00t_dim]."""
        if images.ndim == 5:
            images = images[:, 0]
        if images.ndim != 4:
            raise ValueError(f"Expected WAM images with shape [B, C, H, W], got {tuple(images.shape)}")
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype
        images = images / 127.5 - 1.0
        self._raise_if_nonfinite("image normalization", images)

        video = images.unsqueeze(2)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            video_latents = self.vae.encode(video.float(), tiled=False)
        self._raise_if_nonfinite("VAE encode", video_latents)
        dit_dtype = self.dit.patch_embedding.weight.dtype
        video_latents = video_latents.to(dtype=dit_dtype)
        x = self.dit.patch_embedding(video_latents)
        self._raise_if_nonfinite("patch embedding", x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long, device=device)
        freqs = self.dit._create_freqs(grid_size=grid_size, start_frame=0)
        x = x.flatten(start_dim=2).transpose(1, 2)
        visual_seq_len = x.shape[1]
        num_latent_frames = grid_size[0].item()

        self._raise_if_nonfinite("state input", state)
        state_features = self.dit.state_encoder(state)
        self._raise_if_nonfinite("state encoder", state_features)
        context_features = self.context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        self._raise_if_nonfinite("context tokens", context_features)
        context_register = torch.cat([context_features, state_features], dim=1)
        self._raise_if_nonfinite("context register", context_register)
        context_register_length = context_register.shape[1]
        x = torch.cat([x, context_register], dim=1)
        self._raise_if_nonfinite("visual/context concat", x)

        video_timestep = torch.zeros(batch_size, num_latent_frames, device=device, dtype=torch.long)
        video_timestep = video_timestep.unsqueeze(-1).expand(
            batch_size,
            num_latent_frames,
            visual_seq_len // num_latent_frames,
        )
        video_timestep = video_timestep.reshape(batch_size, -1)
        context_timestep = torch.zeros(batch_size, self.num_context_tokens, device=device, dtype=torch.long)
        state_timestep = torch.zeros(batch_size, state_features.shape[1], device=device, dtype=torch.long)
        timestep = torch.cat([video_timestep, context_timestep, state_timestep], dim=1)

        e = self.dit.time_embedding(
            sinusoidal_embedding_1d(self.dit.freq_dim, timestep.flatten()).type_as(x)
        )
        e = self.dit.time_projection(e.unflatten(dim=0, sizes=(batch_size, -1)))
        e = e.unflatten(dim=2, sizes=(6, self.dit.dim)).to(dtype=x.dtype)
        self._raise_if_nonfinite("time embedding", e)
        text_features, attention_mask = self._resolve_text_features(texts, device=device, dtype=dtype)
        self._raise_if_nonfinite("text cache lookup", text_features)
        dit_context = self.dit.text_embedding(text_features.to(device=device, dtype=x.dtype))
        self._raise_if_nonfinite("text embedding", dit_context)
        # DreamZero's Wan2.2 path attends over the fixed text embedding sequence after
        # text_embedding(); cached attention_mask describes the source text tokens only.
        context_lens = torch.full(
            (batch_size,),
            dit_context.shape[1],
            dtype=torch.long,
            device=attention_mask.device,
        )

        for block_idx, block in enumerate(self.dit.blocks):
            x = self._forward_block_memory(
                block,
                x,
                e,
                freqs,
                dit_context,
                context_lens,
                context_register_length,
            )
            self._raise_if_nonfinite(f"DreamZero block {block_idx}", x)

        context_tokens = x[:, visual_seq_len : visual_seq_len + self.num_context_tokens]
        self._raise_if_nonfinite("context token slice", context_tokens)
        projected_context = self.context_projector(context_tokens)
        self._raise_if_nonfinite("context projector", projected_context)
        return projected_context
