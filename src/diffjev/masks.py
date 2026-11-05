"""Visibility rules (AGENTS.md sections 2.1-2.2).

Lower layers (1 .. L-k):

    | Token group                  | Allowed keys/values      |
    |------------------------------|--------------------------|
    | Shared query                 | Query only               |
    | Branch i, including its slot | Query and branch i only  |
    | Padding                      | Excluded (as keys)       |

Top layers, static context path (cached once per rollout):
    query rows      -> query rows
    option-i rows   -> query rows + option-i rows   (never dynamic slots)

Top layers, dynamic slot rows:
    pass 1 (default)        -> active slots only
    pass >= 2               -> active slots + query memory + OWN option memory
    no-cross-slot ablation  -> the slot-to-slot block becomes the identity

The production encoder realises the lower-layer rule by chunked attention
(query first, then branches reading cached query KV). ``lower_layer_dense_mask``
is the explicit reference used by tests to check that chunking is exact.
"""

from __future__ import annotations

from typing import Optional

import torch


def lower_layer_dense_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Reference boolean mask ``allowed[row, key]`` for a flat layout.

    ``segment_ids``: 0 = query, i + 1 = branch i, -1 = padding.
    Padding rows may attend to themselves only (their outputs are discarded)
    so that no softmax row is empty.
    """
    seg = segment_ids
    row = seg[:, None]
    key = seg[None, :]
    key_valid = key >= 0
    query_row = row == 0
    branch_row = row > 0
    allowed = torch.zeros(seg.shape[0], seg.shape[0], dtype=torch.bool, device=seg.device)
    allowed |= query_row & (key == 0)
    allowed |= branch_row & key_valid & ((key == 0) | (key == row))
    pad = seg < 0
    allowed[pad, :] = False
    idx = torch.nonzero(pad, as_tuple=False).flatten()
    allowed[idx, idx] = True
    return allowed


def slot_block(num_slots: int, cross_slot: bool, device=None) -> torch.Tensor:
    """Slot-to-slot visibility among active slots (symmetric)."""
    if cross_slot:
        return torch.ones(num_slots, num_slots, dtype=torch.bool, device=device)
    return torch.eye(num_slots, dtype=torch.bool, device=device)


def slot_with_memory_mask(
    num_slots: int,
    query_len: int,
    option_valid: torch.Tensor,
    cross_slot: bool,
) -> torch.Tensor:
    """Mask for slot rows in passes that read detail memory.

    Layout: one query row per slot (batch dimension = slot); keys are
    ``[all active slots | shared query memory | own option memory]``.
    Returns ``allowed`` of shape ``[s, 1, s + q + m_max]``.
    ``option_valid``: ``[s, m_max]`` bool for each slot's own option tokens.
    """
    s = num_slots
    device = option_valid.device
    slot_part = slot_block(s, cross_slot, device)  # [s, s]
    query_part = torch.ones(s, query_len, dtype=torch.bool, device=device)
    allowed = torch.cat([slot_part, query_part, option_valid.bool()], dim=1)
    return allowed[:, None, :]


def option_memory_mask(query_len: int, option_valid: torch.Tensor) -> torch.Tensor:
    """Static option rows read the query and their own (valid) option text.

    Returns ``[S, m_max, q + m_max]``. Padding option rows read the query only
    (non-empty row; output discarded).
    """
    S, m = option_valid.shape
    device = option_valid.device
    q_part = torch.ones(S, m, query_len, dtype=torch.bool, device=device)
    o_part = option_valid[:, None, :].expand(S, m, m)
    return torch.cat([q_part, o_part], dim=2)


def branch_chunk_mask(query_len: int, branch_valid: torch.Tensor) -> torch.Tensor:
    """Lower-layer mask for a chunk of branches reading cached query KV.

    Returns ``[C, b_max, q + b_max]``: every row reads all query keys and the
    valid tokens of its own branch (slot included); padding keys excluded.
    """
    C, b = branch_valid.shape
    device = branch_valid.device
    q_part = torch.ones(C, b, query_len, dtype=torch.bool, device=device)
    b_part = branch_valid[:, None, :].expand(C, b, b)
    return torch.cat([q_part, b_part], dim=2)


def full_mask(n_rows: int, n_keys: int, batch: int = 1, device: Optional[torch.device] = None):
    return torch.ones(batch, n_rows, n_keys, dtype=torch.bool, device=device)

