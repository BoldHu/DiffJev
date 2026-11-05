"""Lower-layer branch encoder and top-layer static detail memory.

Lower layers (AGENTS.md 2.1) are executed *without* a dense mask over all
candidate tokens:

1. The shared query runs alone through layers ``0 .. L-k-1`` (query reads query
   only); its per-layer keys/values are kept.
2. Branches are encoded in chunks. Each branch row reads the cached query KV of
   the same layer plus the valid tokens of its own branch. Chunks are
   independent, so chunking preserves the exact mathematical outputs.

At the boundary we keep ``h_i_base`` (slot hidden state), the query hidden
states and each option's hidden states (slot excluded).

Top-layer static context memory (AGENTS.md 2.2) is built per top layer:
query rows read query rows only; option rows read the query and their own
text; neither reads dynamic slots. Option memory is built lazily, only for
the candidates requested (survivors when pruning is enabled). Keys/values are
cached separately for every top layer and are valid for the whole rollout
(the dependencies are fixed); rebuild after parameter updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .backbones.base import BackboneAdapter
from .masks import branch_chunk_mask, option_memory_mask
from .packing import PackedInput


@dataclass
class BoundaryState:
    h_slot_base: torch.Tensor  # [K, d]
    query_h: torch.Tensor  # [q, d]
    query_pos: torch.Tensor  # [q]
    option_h: torch.Tensor  # [K, m_max, d]  (m_max may be 0)
    option_pos: torch.Tensor  # [K, m_max]
    option_valid: torch.Tensor  # [K, m_max]
    slot_pos: torch.Tensor  # [K]

    @property
    def num_candidates(self) -> int:
        return int(self.h_slot_base.shape[0])


def encode_branches(
    backbone: BackboneAdapter,
    packed: PackedInput,
    num_top_layers: int,
    chunk_size: int = 64,
) -> BoundaryState:
    lower = backbone.lower_layer_indices(num_top_layers)
    device = backbone.input_embedding_module().weight.device
    packed = packed.to(device)

    # 1) shared query, query-only attention
    hq = backbone.embed(packed.query_ids)[None]  # [1, q, d]
    posq = packed.query_pos[None]
    q_len = packed.query_len
    query_kv: List = []
    allowed_q = torch.ones(1, q_len, q_len, dtype=torch.bool, device=device)
    for l in lower:
        k, v = backbone.layer_kv(l, hq, posq)
        query_kv.append((k, v))
        hq = backbone.layer_forward(l, hq, posq, k, v, allowed_q)

    # 2) branches in independent chunks
    K, b_max = packed.branch_ids.shape
    outs = []
    for start in range(0, K, max(1, chunk_size)):
        sl = slice(start, min(K, start + chunk_size))
        ids = packed.branch_ids[sl]
        pos = packed.branch_pos[sl]
        valid = packed.branch_valid[sl]
        C = ids.shape[0]
        hb = backbone.embed(ids)  # [C, b, d]
        allowed = branch_chunk_mask(q_len, valid)
        for li, l in enumerate(lower):
            kb, vb = backbone.layer_kv(l, hb, pos)
            kq, vq = query_kv[li]
            k = torch.cat([kq.expand(C, -1, -1, -1), kb], dim=2)
            v = torch.cat([vq.expand(C, -1, -1, -1), vb], dim=2)
            hb = backbone.layer_forward(l, hb, pos, k, v, allowed)
        outs.append(hb)
    hb_all = torch.cat(outs, dim=0)  # [K, b, d]

    return BoundaryState(
        h_slot_base=hb_all[:, 0],
        query_h=hq[0],
        query_pos=packed.query_pos,
        option_h=hb_all[:, 1:],
        option_pos=packed.branch_pos[:, 1:],
        option_valid=packed.branch_valid[:, 1:],
        slot_pos=packed.branch_pos[:, 0],
    )


@dataclass
class DetailMemory:
    """Per-top-layer cached keys/values of the static context path."""

    query_k: List[torch.Tensor]  # each [1, Hkv, q, dh]
    query_v: List[torch.Tensor]
    option_k: Dict[int, List[torch.Tensor]] = field(default_factory=dict)  # cand -> per-layer [Hkv, m, dh]
    option_v: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    build_log: List[List[int]] = field(default_factory=list)  # candidates built per call

    @property
    def built_candidates(self) -> List[int]:
        return sorted(self.option_k.keys())


def build_query_memory(backbone: BackboneAdapter, boundary: BoundaryState, num_top_layers: int) -> DetailMemory:
    top = backbone.top_layer_indices(num_top_layers)
    hq = boundary.query_h[None]
    posq = boundary.query_pos[None]
    q = hq.shape[1]
    allowed = torch.ones(1, q, q, dtype=torch.bool, device=hq.device)
    qk, qv = [], []
    for l in top:
        k, v = backbone.layer_kv(l, hq, posq)
        qk.append(k)
        qv.append(v)
        hq = backbone.layer_forward(l, hq, posq, k, v, allowed)
    return DetailMemory(query_k=qk, query_v=qv)


def extend_option_memory(
    backbone: BackboneAdapter,
    boundary: BoundaryState,
    memory: DetailMemory,
    candidates: Sequence[int],
    num_top_layers: int,
    chunk_size: int = 64,
) -> DetailMemory:
    """Build static option memory for ``candidates`` not built yet (lazy)."""
    todo = [int(c) for c in candidates if int(c) not in memory.option_k]
    memory.build_log.append(list(todo))
    if not todo:
        return memory
    top = backbone.top_layer_indices(num_top_layers)
    q = boundary.query_h.shape[0]
    for start in range(0, len(todo), max(1, chunk_size)):
        idx = todo[start : start + chunk_size]
        idx_t = torch.tensor(idx, dtype=torch.long, device=boundary.option_h.device)
        ho = boundary.option_h[idx_t]  # [S, m, d]
        pos = boundary.option_pos[idx_t]
        valid = boundary.option_valid[idx_t]
        S = ho.shape[0]
        per_k = [[] for _ in idx]
        per_v = [[] for _ in idx]
        allowed = option_memory_mask(q, valid)
        for li, l in enumerate(top):
            ko, vo = backbone.layer_kv(l, ho, pos)
            for j in range(S):
                per_k[j].append(ko[j])
                per_v[j].append(vo[j])
            if li == len(top) - 1:
                break  # the last layer's outputs are never read by slots
            k = torch.cat([memory.query_k[li].expand(S, -1, -1, -1), ko], dim=2)
            v = torch.cat([memory.query_v[li].expand(S, -1, -1, -1), vo], dim=2)
            ho = backbone.layer_forward(l, ho, pos, k, v, allowed)
        for j, c in enumerate(idx):
            memory.option_k[c] = per_k[j]
            memory.option_v[c] = per_v[j]
    return memory


def gather_option_memory(memory: DetailMemory, candidates: Sequence[int], layer_idx: int):
    """Stack cached option K/V of ``candidates`` for top layer ``layer_idx``.

    Returns ``k, v`` of shape ``[s, Hkv, m_max, dh]`` (all candidates share the
    packed m_max, padding masked by ``option_valid``).
    """
    ks = torch.stack([memory.option_k[int(c)][layer_idx] for c in candidates], dim=0)
    vs = torch.stack([memory.option_v[int(c)][layer_idx] for c in candidates], dim=0)
    return ks, vs

