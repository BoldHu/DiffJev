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

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .hf_base import BlockParts, HFDiffusionAdapterBase


def _compatible_llada_class(native_class):
    """Bridge the pinned v4-era wrapper to the v5 loading lifecycle.

    Keep the downloaded source and global Transformers classes untouched. The
    pinned Base checkpoint has independent embedding/output weights; do not
    infer tying from HF's unrelated ``tie_word_embeddings`` default.
    """
    if not hasattr(native_class, "get_expanded_tied_weights_keys"):
        return native_class  # Transformers 4 uses the original lifecycle.

    class CompatibleLLaDAModelLM(native_class):
        def __init__(self, config, *args, **kwargs):
            if config.weight_tying:
                raise ValueError("DiffJev's pinned LLaDA Base loader requires weight_tying=False")
            super().__init__(config, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                # Called BEFORE checkpoint tensors are loaded, inside HF's
                # from_pretrained initialization context, never after loading.
                self.post_init()

        def tie_weights(self, missing_keys=None, recompute_mapping=True):
            # v5 calls this with keywords during both post_init and finalization.
            # Preserve the native no-tying behavior of the pinned checkpoint.
            if self.config.weight_tying:
                raise ValueError("DiffJev's pinned LLaDA Base loader requires weight_tying=False")
            return super().tie_weights()

    return CompatibleLLaDAModelLM


class LLaDAAdapter(HFDiffusionAdapterBase):
    MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
    # Official LLaDA-8B-Base [MASK] token id is 126336 (used when tokenizer
    # does not populate mask_token_id by default).
    DEFAULT_MASK_TOKEN_ID = 126336

    @classmethod
    def from_local(cls, local_dir, torch_dtype=torch.bfloat16, device_map=None, **kwargs):
        """Load the native checkpoint with a model-local HF lifecycle bridge."""
        if not Path(local_dir).is_dir():
            raise FileNotFoundError(f"Missing local LLaDA snapshot: {local_dir}")
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        class_ref = getattr(config, "auto_map", {}).get("AutoModel")
        if config.model_type != "llada" or class_ref != "modeling_llada.LLaDAModelLM":
            raise ValueError("Expected the pinned LLaDA-8B-Base native implementation")
        native_class = get_class_from_dynamic_module(
            class_ref, local_dir, local_files_only=True
        )
        model_class = _compatible_llada_class(native_class)
        dtype_key = "dtype" if model_class is not native_class else "torch_dtype"
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        model, info = model_class.from_pretrained(
            local_dir, config=config, local_files_only=True,
            device_map=device_map, output_loading_info=True, **{dtype_key: torch_dtype}
        )
        problems = {key: info[key] for key in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                    if info.get(key)}
        if problems:
            raise RuntimeError(f"LLaDA checkpoint did not load exactly: {problems}")
        return cls(model, tokenizer, **kwargs)

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
