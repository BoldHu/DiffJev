"""Tiny random Llama-style *bidirectional* transformer used for local tests.

This is a stand-in for a masked diffusion language model with the same
structure DiffJev relies on (pre-norm blocks, RoPE, GQA, SwiGLU, final RMSNorm,
linear vocabulary head with optional bias). It is randomly initialised; its
outputs carry no research meaning. It never downloads anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import BackboneAdapter, resolve_single_token
from .rope import apply_rope


SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "Yes", "No"]


class MockTokenizer:
    """Whitespace word-level tokenizer with a fixed vocabulary.

    Words listed in ``multi_token_words`` deliberately encode to two tokens so
    the verbalizer check can be exercised.
    """

    def __init__(self, words: Sequence[str], multi_token_words: Optional[Dict[str, Sequence[str]]] = None):
        vocab = list(SPECIAL_TOKENS)
        for w in words:
            if w not in vocab:
                vocab.append(w)
        self.multi = {k: list(v) for k, v in (multi_token_words or {}).items()}
        for pieces in self.multi.values():
            for p in pieces:
                if p not in vocab:
                    vocab.append(p)
        self.vocab = vocab
        self.tok2id = {t: i for i, t in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.tok2id["[PAD]"]

    @property
    def mask_token_id(self) -> int:
        return self.tok2id["[MASK]"]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        out: List[int] = []
        for w in text.split():
            if w.startswith("[") and w.endswith("]") and w[1:-1].isdigit():
                # Test-only digit alphabet permits distinct multi-token AR labels.
                out.extend(self.tok2id["digit:" + digit] for digit in w[1:-1])
                out.append(self.tok2id["label:end"])
                continue
            if w in self.multi:
                out.extend(self.tok2id[p] for p in self.multi[w])
            else:
                out.append(self.tok2id.get(w, self.tok2id["[UNK]"]))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)


def default_mock_tokenizer(n_words: int = 48) -> MockTokenizer:
    words = [f"w{i}" for i in range(n_words)] + ["query", "option", "choose", "bin", "label:end"] + [f"digit:{i}" for i in range(10)]
    return MockTokenizer(words, multi_token_words={"Maybe": ["May", "##be"]})


@dataclass
class MockConfig:
    vocab_size: int = 64
    hidden_size: int = 32
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 2
    ffn_size: int = 64
    rope_theta: float = 10000.0
    head_bias: bool = True
    init_std: float = 0.2
    seed: int = 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class MockBlock(nn.Module):
    def __init__(self, cfg: MockConfig):
        super().__init__()
        d, H, Hkv = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads
        self.dh = d // H
        self.H, self.Hkv = H, Hkv
        self.att_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * self.dh, bias=False)
        self.k_proj = nn.Linear(d, Hkv * self.dh, bias=False)
        self.v_proj = nn.Linear(d, Hkv * self.dh, bias=False)
        self.o_proj = nn.Linear(H * self.dh, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.gate = nn.Linear(d, cfg.ffn_size, bias=False)
        self.up = nn.Linear(d, cfg.ffn_size, bias=False)
        self.down = nn.Linear(cfg.ffn_size, d, bias=False)


class MockBackbone(BackboneAdapter):
    """Fully implemented adapter for the mock transformer."""

    def __init__(self, cfg: MockConfig, tokenizer: MockTokenizer, yes: str = "Yes", no: str = "No"):
        super().__init__()
        if cfg.vocab_size < tokenizer.vocab_size:
            cfg.vocab_size = tokenizer.vocab_size
        self.cfg = cfg
        self.tokenizer = tokenizer
        g = torch.Generator().manual_seed(cfg.seed)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([MockBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=cfg.head_bias)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) * cfg.init_std)
                elif name.endswith("bias"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        self.yes_token_id = resolve_single_token(tokenizer, yes)
        self.no_token_id = resolve_single_token(tokenizer, no)
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    # ---- structure ----
    @property
    def num_layers(self) -> int:
        return self.cfg.num_layers

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    def layer_modules(self):
        return list(self.layers)

    def final_norm_module(self):
        return self.norm

    def input_embedding_module(self):
        return self.embed_tokens

    def output_head_module(self):
        return self.lm_head

    # ---- primitives ----
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(ids)

    def layer_kv(self, layer: int, h: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        blk = self.layers[layer]
        B, n, _ = h.shape
        x = blk.att_norm(h)
        k = blk.k_proj(x).view(B, n, blk.Hkv, blk.dh).transpose(1, 2)
        v = blk.v_proj(x).view(B, n, blk.Hkv, blk.dh).transpose(1, 2)
        k = apply_rope(k, pos, self.cfg.rope_theta)
        return k, v

    def layer_forward(self, layer, h, pos, k, v, allowed):
        blk = self.layers[layer]
        B, n, _ = h.shape
        x = blk.att_norm(h)
        q = blk.q_proj(x).view(B, n, blk.H, blk.dh).transpose(1, 2)
        q = apply_rope(q, pos, self.cfg.rope_theta)
        rep = blk.H // blk.Hkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)  # [B, H, n, n_k]
        scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, n, blk.H * blk.dh)
        h = h + blk.o_proj(o)
        y = blk.ffn_norm(h)
        h = h + blk.down(F.silu(blk.gate(y)) * blk.up(y))
        return h

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h)

    def verify_adapter(self) -> Dict[str, bool]:
        return {"mock": True, "single_token_verbalizers": True, "explicit_mask_kernel": True}


def build_mock_backbone(
    cfg: Optional[MockConfig] = None,
    tokenizer: Optional[MockTokenizer] = None,
    dtype: torch.dtype = torch.float32,
) -> MockBackbone:
    tok = tokenizer or default_mock_tokenizer()
    model = MockBackbone(cfg or MockConfig(), tok)
    return model.to(dtype)

