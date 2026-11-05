"""Backbone-agnostic adapter interface.

The DiffJev algorithm (encoder, co-denoiser, diffusion, losses) only talks to
a ``BackboneAdapter``. Each real backbone (LLaDA, Nemotron-Labs-Diffusion)
implements the same small set of *layer-level* primitives on top of its own
pretrained modules, so switching backbones never touches the core modules.

Required primitives
-------------------
``embed(ids)``                         input embeddings (frozen)
``token_embedding(token_id)``          one input-embedding row, ``E(token)``
``layer_kv(l, h, pos)``                keys/values of layer ``l`` for rows ``h`` at
                                        semantic positions ``pos`` (RoPE applied)
``layer_forward(l, h, pos, k, v, allowed)``
                                        one full transformer block for query rows
                                        ``h`` attending to externally assembled
                                        keys/values under boolean mask ``allowed``
``final_norm(h)``                      the backbone's final normalisation
``yes_no_logit_diff(h_normed)``        ``logit(Yes) - logit(No)`` using only the two
                                        existing head rows (+ existing bias)

Why layer-level primitives: DiffJev needs keys assembled from several sources
(shared-query cache, own-branch tokens, other slots, static detail memory) with
custom visibility and *repeated* semantic positions. Passing explicit
semantic positions keeps KV *storage* addresses separate from semantic slot
positions (a requirement for Nemotron, whose native path ties scaling to
``cache_position``). Passing an explicit ``allowed`` mask to the attention
kernel avoids relying on native paths that may discard incoming masks.

Shapes
------
``h``: ``[B, n, d]``; ``pos``: ``[B, n]`` (long); ``k, v``: ``[B, H_kv, n_k, d_h]``;
``allowed``: ``[B, n, n_k]`` bool, every row must allow at least one key.

The adapter must add **no** trainable parameters (AGENTS.md section 2).
"""

from __future__ import annotations

import abc
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


class VerbalizerError(ValueError):
    """Raised when a verbalizer is not exactly one token."""


def resolve_single_token(tokenizer, text: str) -> int:
    """Return the id of ``text`` if it encodes to exactly one token.

    Never silently substitutes the first token of a multi-token verbalizer.
    ``tokenizer`` must provide ``encode(text, add_special_tokens=False)``.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    ids = list(ids)
    if len(ids) != 1:
        raise VerbalizerError(
            f"verbalizer {text!r} encodes to {len(ids)} tokens {ids}; "
            "DiffJev requires a verified single-token Yes/No verbalizer"
        )
    if getattr(tokenizer, "unk_token_id", None) is not None and ids[0] == tokenizer.unk_token_id:
        raise VerbalizerError(f"verbalizer {text!r} resolved to the unknown token")
    return int(ids[0])


class BackboneAdapter(nn.Module, abc.ABC):
    """Abstract adapter. Subclasses wrap pretrained modules without adding any."""

    # ---- token ids (set by subclasses after verbalizer verification) ----
    yes_token_id: int
    no_token_id: int
    mask_token_id: int
    pad_token_id: int

    # ---- structure -------------------------------------------------------
    @property
    @abc.abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int: ...

    @abc.abstractmethod
    def layer_modules(self) -> Sequence[nn.Module]:
        """The L transformer blocks in order (for freezing / counting)."""

    @abc.abstractmethod
    def final_norm_module(self) -> nn.Module: ...

    @abc.abstractmethod
    def input_embedding_module(self) -> nn.Module: ...

    @abc.abstractmethod
    def output_head_module(self) -> nn.Module: ...

    # ---- primitives ------------------------------------------------------
    @abc.abstractmethod
    def embed(self, ids: torch.Tensor) -> torch.Tensor: ...

    def token_embedding(self, token_id: int) -> torch.Tensor:
        w = self.input_embedding_module().weight
        return w[token_id]

    @abc.abstractmethod
    def layer_kv(
        self, layer: int, h: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def layer_forward(
        self,
        layer: int,
        h: torch.Tensor,
        pos: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def final_norm(self, h: torch.Tensor) -> torch.Tensor: ...

    def yes_no_head_rows(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact difference of the existing Yes/No head rows and biases.

        Default implementation assumes a linear head ``nn.Linear``-like module
        with ``weight [V, d]`` and optional ``bias [V]``. Override when the
        head is not linear (then a restricted projection may be impossible).
        """
        head = self.output_head_module()
        w = head.weight
        w_diff = w[self.yes_token_id] - w[self.no_token_id]
        b = getattr(head, "bias", None)
        b_diff = (b[self.yes_token_id] - b[self.no_token_id]) if b is not None else w.new_zeros(())
        return w_diff, b_diff

    def yes_no_logit_diff(self, h_normed: torch.Tensor) -> torch.Tensor:
        """``z = logit(Yes) - logit(No)`` without a full-vocabulary projection."""
        w_diff, b_diff = self.yes_no_head_rows()
        return h_normed @ w_diff.to(h_normed.dtype) + b_diff.to(h_normed.dtype)

    def full_logits(self, h_normed: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits (used only by tests / serialized baselines)."""
        head = self.output_head_module()
        return torch.nn.functional.linear(h_normed, head.weight, getattr(head, "bias", None))

    def attention_visibility(self, n, device, causal=False):
        mask = torch.ones(1, n, n, dtype=torch.bool, device=device)
        return torch.tril(mask) if causal else mask

    # ---- generic helpers -------------------------------------------------
    def lower_layer_indices(self, num_top: int) -> List[int]:
        return list(range(self.num_layers - num_top))

    def top_layer_indices(self, num_top: int) -> List[int]:
        if num_top > self.num_layers:
            raise ValueError("num_top exceeds number of layers")
        return list(range(self.num_layers - num_top, self.num_layers))

    def run_layers_dense(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        allowed: torch.Tensor,
        layers: Sequence[int],
    ) -> torch.Tensor:
        """Reference dense execution of several layers on one flat sequence."""
        for l in layers:
            k, v = self.layer_kv(l, h, pos)
            h = self.layer_forward(l, h, pos, k, v, allowed)
        return h

    def describe(self) -> Dict[str, object]:
        return {
            "class": type(self).__name__,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "mask_token_id": self.mask_token_id,
        }

    def verify_adapter(self) -> Dict[str, bool]:
        """Model-specific verification checklist. Real adapters must implement
        this against their pinned revision before use (AGENTS.md 2.0)."""
        raise NotImplementedError

