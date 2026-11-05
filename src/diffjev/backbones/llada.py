"""LLaDA-8B-Base adapter (method-development and complete-ablation anchor).

Checkpoint: ``GSAI-ML/LLaDA-8B-Base``.
Reuses the native ``model.transformer.wte``, ``model.transformer.blocks``,
``model.transformer.ln_f``, and ``model.transformer.ff_out`` (or tied embedding
head) without adding any trainable parameters.

In LLaDA's ``LLaDALlamaBlock`` (``modeling_llada.py``):
- Pre-attention norm: ``attn_norm``
- Projections: ``q_proj``, ``k_proj``, ``v_proj`` (or fused ``att_proj``) and
  output projection ``attn_out``
- Pre-FFN norm: ``ff_norm``
- Gated FFN: ``act(ff_proj(x)) * up_proj(x) -> ff_out``. The two
  pretrained projections are not interchangeable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .hf_base import BlockParts, HFDiffusionAdapterBase


class LLaDAAdapter(HFDiffusionAdapterBase):
    MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
    # Official LLaDA-8B-Base [MASK] token id is 126336 (used when tokenizer
    # does not populate mask_token_id by default).
    DEFAULT_MASK_TOKEN_ID = 126336

    PATH_CANDIDATES: Dict[str, Tuple[str, ...]] = {
        "embed": (
            "model.transformer.wte",
            "transformer.wte",
            "model.embed_tokens",
            "embed_tokens",
        ),
        "layers": (
            "model.transformer.blocks",
            "transformer.blocks",
            "model.layers",
            "layers",
        ),
        "final_norm": (
            "model.transformer.ln_f",
            "transformer.ln_f",
            "model.norm",
            "norm",
        ),
        "head": (
            "model.transformer.ff_out",
            "transformer.ff_out",
            "lm_head",
        ),
    }

    def _activation(self, layer, x):
        return self.layer_modules()[layer].act(x)
