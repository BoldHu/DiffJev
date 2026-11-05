"""Shared-query + independent-branch packing and Set-RoPE positions.

AGENTS.md section 2.1: pack a shared query ``Q`` (task instructions included)
and branches ``[MASK_i] + option_i``. Positions in the paper notation are
1-indexed (query ``1..L_q``, slot ``L_q + 1``, options from ``L_q + 2``). This
code uses 0-indexed integer positions, which differ by a constant shift and
therefore produce identical RoPE *relative* geometry:

    query  : 0 .. L_q - 1
    slot   : L_q                      (tied across every branch: Set-RoPE)
    option : L_q + 1 .. L_q + |o_i|   (restarts for every branch)

The ``sequential`` mode exists only for the Exp 4 "untied sequential
positions" ablation: branch ``i`` continues where branch ``i-1`` ended, as in
a serialized prompt. It deliberately breaks permutation equivariance.

Candidate identity (``candidate_ids``) and display aliases are metadata. They
are carried alongside the tensors and never enter the model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class Candidate:
    """One candidate alternative.

    ``cid`` is an immutable semantic identity used for output mapping and tie
    breaks. ``alias`` is an optional display label (e.g. "B"); it is metadata
    only and is never tokenized into DiffJev inputs.
    """

    cid: str
    token_ids: List[int]
    alias: Optional[str] = None
    text: Optional[str] = None  # original semantic text; tokenize separately per backbone


@dataclass
class PackedInput:
    query_ids: torch.Tensor  # [q]
    query_pos: torch.Tensor  # [q]
    branch_ids: torch.Tensor  # [K, b_max]; column 0 is the slot ([MASK])
    branch_pos: torch.Tensor  # [K, b_max]
    branch_valid: torch.Tensor  # [K, b_max] bool (False = padding)
    option_lengths: torch.Tensor  # [K]
    candidate_ids: List[str]
    aliases: List[Optional[str]] = field(default_factory=list)
    position_mode: str = "tied"

    @property
    def num_candidates(self) -> int:
        return int(self.branch_ids.shape[0])

    @property
    def query_len(self) -> int:
        return int(self.query_ids.shape[0])

    @property
    def slot_pos(self) -> torch.Tensor:  # [K]
        return self.branch_pos[:, 0]

    def to(self, device) -> "PackedInput":
        return PackedInput(
            query_ids=self.query_ids.to(device),
            query_pos=self.query_pos.to(device),
            branch_ids=self.branch_ids.to(device),
            branch_pos=self.branch_pos.to(device),
            branch_valid=self.branch_valid.to(device),
            option_lengths=self.option_lengths.to(device),
            candidate_ids=list(self.candidate_ids),
            aliases=list(self.aliases),
            position_mode=self.position_mode,
        )

    def select(self, idx: Sequence[int]) -> "PackedInput":
        """Gather a subset/permutation of branches (identity mapping preserved)."""
        idx_t = torch.as_tensor(list(idx), dtype=torch.long, device=self.branch_ids.device)
        return PackedInput(
            query_ids=self.query_ids,
            query_pos=self.query_pos,
            branch_ids=self.branch_ids[idx_t],
            branch_pos=self.branch_pos[idx_t],
            branch_valid=self.branch_valid[idx_t],
            option_lengths=self.option_lengths[idx_t],
            candidate_ids=[self.candidate_ids[i] for i in idx],
            aliases=[self.aliases[i] for i in idx] if self.aliases else [],
            position_mode=self.position_mode,
        )


def pack_branches(
    query_ids: Sequence[int],
    candidates: Sequence[Candidate],
    mask_token_id: int,
    pad_token_id: int,
    position_mode: str = "tied",
    max_option_len: Optional[int] = None,
) -> PackedInput:
    """Build the shared query and the ``[MASK_i] + option_i`` branches.

    ``max_option_len`` applies one shared truncation policy to every option (it
    must be identical across compared methods).
    """
    if position_mode not in ("tied", "sequential"):
        raise ValueError(position_mode)
    if len(candidates) == 0:
        raise ValueError("need at least one explicit candidate slot")
    ids = [c.cid for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")

    q = len(query_ids)
    opts = [list(c.token_ids) for c in candidates]
    if max_option_len is not None:
        opts = [o[:max_option_len] for o in opts]
    lengths = [len(o) for o in opts]
    b_max = 1 + max(lengths)
    K = len(candidates)

    branch_ids = torch.full((K, b_max), pad_token_id, dtype=torch.long)
    branch_pos = torch.zeros((K, b_max), dtype=torch.long)
    branch_valid = torch.zeros((K, b_max), dtype=torch.bool)

    offset = q  # sequential-mode running start
    for i, o in enumerate(opts):
        n = 1 + len(o)
        branch_ids[i, 0] = mask_token_id
        if o:
            branch_ids[i, 1:n] = torch.tensor(o, dtype=torch.long)
        start = q if position_mode == "tied" else offset
        branch_pos[i, :n] = torch.arange(start, start + n)
        branch_valid[i, :n] = True
        offset += n

    return PackedInput(
        query_ids=torch.tensor(list(query_ids), dtype=torch.long),
        query_pos=torch.arange(q, dtype=torch.long),
        branch_ids=branch_ids,
        branch_pos=branch_pos,
        branch_valid=branch_valid,
        option_lengths=torch.tensor(lengths, dtype=torch.long),
        candidate_ids=ids,
        aliases=[c.alias for c in candidates],
        position_mode=position_mode,
    )


@dataclass
class DenseLayout:
    """Flat single-sequence layout. Used only as a *reference* for tests that
    check chunked branch encoding against an explicit dense mask; the
    production encoder never builds a dense mask over all candidate tokens."""

    input_ids: torch.Tensor  # [N]
    position_ids: torch.Tensor  # [N]
    segment_ids: torch.Tensor  # [N]; 0 = query, i + 1 = branch i, -1 = padding
    slot_index: torch.Tensor  # [K] flat index of each slot


def to_dense_layout(packed: PackedInput, pad_token_id: int, n_pad: int = 0) -> DenseLayout:
    ids = [packed.query_ids]
    pos = [packed.query_pos]
    seg = [torch.zeros(packed.query_len, dtype=torch.long)]
    slot_index = []
    cursor = packed.query_len
    for i in range(packed.num_candidates):
        valid = packed.branch_valid[i]
        n = int(valid.sum())
        ids.append(packed.branch_ids[i, :n])
        pos.append(packed.branch_pos[i, :n])
        seg.append(torch.full((n,), i + 1, dtype=torch.long))
        slot_index.append(cursor)
        cursor += n
    if n_pad:
        ids.append(torch.full((n_pad,), pad_token_id, dtype=torch.long))
        pos.append(torch.zeros(n_pad, dtype=torch.long))
        seg.append(torch.full((n_pad,), -1, dtype=torch.long))
    return DenseLayout(
        input_ids=torch.cat(ids),
        position_ids=torch.cat(pos),
        segment_ids=torch.cat(seg),
        slot_index=torch.tensor(slot_index, dtype=torch.long),
    )

