"""Nemotron-Labs-Diffusion 3B / 8B / 14B adapter (size panel: Exp 1, 3, 5).

Checkpoints (instruction-tuned IDs without ``-Base``, AGENTS.md section 2.0):
    nvidia/Nemotron-Labs-Diffusion-3B
    nvidia/Nemotron-Labs-Diffusion-8B
    nvidia/Nemotron-Labs-Diffusion-14B

Architectural requirements enforced here:
1. Reuse existing Nemotron embeddings, ``encoder.layers``, ``encoder.norm``, and
   ``diffusion_head``; refuse or strip any optional self-speculation LoRA.
2. Route DiffJev's required branch / slot / memory visibility ``allowed`` mask
   directly into the attention kernel (the inspected native diffusion attention
   path discards incoming masks).
3. Reset all semantic position-dependent factors (RoPE and ``cache_position``
   query scaling) from DiffJev's explicit semantic positions while keeping KV
   storage addresses separate from repeated semantic slot positions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .hf_base import HFDiffusionAdapterBase

NEMOTRON_CHECKPOINTS: Dict[str, str] = {
    "3B": "nvidia/Nemotron-Labs-Diffusion-3B",
    "8B": "nvidia/Nemotron-Labs-Diffusion-8B",
    "14B": "nvidia/Nemotron-Labs-Diffusion-14B",
}


def strip_optional_lora(hf_model: nn.Module) -> int:
    """Remove or disable any optional self-speculation LoRA modules if present.

    Returns the number of LoRA attributes stripped/disabled.
    """
    removed = 0
    for _, mod in hf_model.named_modules():
        for attr in list(vars(mod).keys()):
            if "lora" in attr.lower():
                setattr(mod, attr, None)
                removed += 1
        if hasattr(mod, "_modules"):
            for k in list(mod._modules.keys()):
                if "lora" in k.lower():
                    del mod._modules[k]
                    removed += 1
    return removed


class NemotronDiffusionAdapter(HFDiffusionAdapterBase):
    PATH_CANDIDATES: Dict[str, Tuple[str, ...]] = {
        "embed": (
            "encoder.embed_tokens",
            "model.encoder.embed_tokens",
            "model.embed_tokens",
            "embed_tokens",
        ),
        "layers": (
            "encoder.layers",
            "model.encoder.layers",
            "model.layers",
            "layers",
        ),
        "final_norm": (
            "encoder.norm",
            "model.encoder.norm",
            "model.norm",
            "norm",
        ),
        "head": (
            "diffusion_head",
            "model.diffusion_head",
            "lm_head",
        ),
    }

    def __init__(self, hf_model: nn.Module, tokenizer, size: str = "8B", **kw):
        if size not in NEMOTRON_CHECKPOINTS:
            raise ValueError(f"size must be one of {sorted(NEMOTRON_CHECKPOINTS)}")
        self.size = size
        self.MODEL_ID = NEMOTRON_CHECKPOINTS[size]
        strip_optional_lora(hf_model)
        super().__init__(hf_model, tokenizer, **kw)

    def _rope(self, x, pos, parts):
        encoder = getattr(self.hf_model, "encoder", None)
        if encoder is None:
            encoder = getattr(getattr(self.hf_model, "model", None), "encoder", None)
        rotary = getattr(encoder, "rotary_emb", None)
        params = getattr(self.hf_model.config, "rope_parameters", None)
        if rotary is None:
            if params:
                raise ValueError("Native Nemotron rotary_emb is required for YaRN parity")
            # Config-reduced fixtures without YaRN only.
            return super()._rope(x, pos, parts)
        # Reuse native YaRN frequencies AND its cos/sin amplitude scaling.
        # Storage offsets never enter either positional computation.
        cos, sin = rotary(x, pos)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos[:, None].to(x.dtype) + rotated * sin[:, None].to(x.dtype)

    def _apply_semantic_query_scaling(self, parts, q, pos):
        params = getattr(self.hf_model.config, "rope_parameters", None) or {}
        beta = params.get("llama_4_scaling_beta", 0.0)
        if not beta:
            return q
        original = params["original_max_position_embeddings"]
        scale = 1.0 + beta * torch.log1p(torch.floor(pos.float() / original))
        return q * scale[:, None, :, None].to(q.dtype)
