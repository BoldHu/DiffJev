"""Qwen2.5 native-weight causal reference, using its own tokenizer and RoPE."""
import torch
from .hf_base import HFDiffusionAdapterBase


class QwenCausalAdapter(HFDiffusionAdapterBase):
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model, tokenizer, **kwargs):
        # The causal baseline never uses a mask embedding.
        super().__init__(model, tokenizer, mask_token_id=0, **kwargs)

    @classmethod
    def from_local(cls, local_dir, torch_dtype=torch.bfloat16, device_map=None, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True,
            torch_dtype=torch_dtype, device_map=device_map, attn_implementation="sdpa")
        return cls(model, tok, **kwargs)

    def _rope(self, x, pos, parts):
        rotary = self.hf_model.model.rotary_emb
        cos, sin = rotary(x, pos)
        half = x.shape[-1] // 2
        return x*cos[:, None] + torch.cat((-x[..., half:], x[..., :half]), -1)*sin[:, None]
