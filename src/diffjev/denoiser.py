"""Top-layer slot co-denoiser (AGENTS.md 2.2, 3.2-3.4).

The final ``k`` transformer layers are reused on each denoising pass. Active
slots attend to one another symmetrically (tied slot positions, shared
weights). Pass 1 reads compressed slot states only; later passes also read the
static query memory and each slot's *own* option memory (never another
option's detail tokens). Boundary feedback is reset from ``h_i_base`` before
every evaluation; previous hidden states are never accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .backbones.base import BackboneAdapter
from .config import DiffJevConfig
from .diffusion import MASK, hard_to_soft, initial_soft_state, mean_field_update, slot_inputs
from .encoder import BoundaryState, DetailMemory, build_query_memory, extend_option_memory, gather_option_memory
from .masks import slot_block, slot_with_memory_mask
from .probability import compatibility_mask, structured_log_probs
from .pruning import pruning_active, select_survivors


@dataclass
class PassRecord:
    t: float
    active: torch.Tensor  # candidate indices (into the packed order)
    z: torch.Tensor  # [s]
    log_p: torch.Tensor  # [s + 1] over [ref, active...]
    used_memory: bool


@dataclass
class RolloutResult:
    passes: List[PassRecord]
    pruned: bool
    survivor_trace: List[torch.Tensor] = field(default_factory=list)  # active set entering each pass
    memory: Optional[DetailMemory] = None
    final_mu: Optional[torch.Tensor] = None

    @property
    def final(self) -> PassRecord:
        return self.passes[-1]

    @property
    def probability_scope(self) -> str:
        return "retained_support" if self.pruned else "full_support"


class SlotDenoiser:
    def __init__(self, backbone: BackboneAdapter, cfg: DiffJevConfig):
        self.backbone = backbone
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _embeddings(self):
        b = self.backbone
        return (
            b.token_embedding(b.yes_token_id),
            b.token_embedding(b.no_token_id),
            b.token_embedding(b.mask_token_id),
        )

    def boundary_inputs(self, boundary: BoundaryState, active: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        e_y, e_n, e_m = self._embeddings()
        return slot_inputs(boundary.h_slot_base[active], mu, e_y, e_n, e_m)

    def ensure_memory(
        self, boundary: BoundaryState, memory: Optional[DetailMemory], active: torch.Tensor
    ) -> DetailMemory:
        if memory is None:
            memory = build_query_memory(self.backbone, boundary, self.cfg.num_top_layers)
        return extend_option_memory(
            self.backbone, boundary, memory, active.tolist(), self.cfg.num_top_layers, self.cfg.branch_chunk_size
        )

    def slot_pass(
        self,
        x: torch.Tensor,
        boundary: BoundaryState,
        active: torch.Tensor,
        memory: Optional[DetailMemory],
    ) -> torch.Tensor:
        """Run the top-k layers on active slot rows; return ``z [s]``."""
        b = self.backbone
        top = b.top_layer_indices(self.cfg.num_top_layers)
        s = x.shape[0]
        pos = boundary.slot_pos[active]  # tied: identical values
        if memory is None:
            h = x[None]  # [1, s, d]
            p = pos[None]
            allowed = slot_block(s, self.cfg.cross_slot_attention, x.device)[None]
            for l in top:
                k, v = b.layer_kv(l, h, p)
                h = b.layer_forward(l, h, p, k, v, allowed)
            h = h[0]
        else:
            q = boundary.query_h.shape[0]
            opt_valid = boundary.option_valid[active]
            allowed = slot_with_memory_mask(s, q, opt_valid, self.cfg.cross_slot_attention)
            h = x[:, None, :]  # [s, 1, d]: one row per slot
            p = pos[:, None]
            for li, l in enumerate(top):
                ks, vs = b.layer_kv(l, h.transpose(0, 1), p.transpose(0, 1))  # [1, Hkv, s, dh]
                ko, vo = gather_option_memory(memory, active.tolist(), li)  # [s, Hkv, m, dh]
                k = torch.cat(
                    [ks.expand(s, -1, -1, -1), memory.query_k[li].expand(s, -1, -1, -1), ko], dim=2
                )
                v = torch.cat(
                    [vs.expand(s, -1, -1, -1), memory.query_v[li].expand(s, -1, -1, -1), vo], dim=2
                )
                h = b.layer_forward(l, h, p, k, v, allowed)
            h = h[:, 0]
        return b.yes_no_logit_diff(b.final_norm(h))

    # ------------------------------------------------------------------
    def rollout(
        self,
        boundary: BoundaryState,
        mode: str,
        full_support: bool = False,
        allow_pruning: bool = True,
        num_passes: Optional[int] = None,
        memory: Optional[DetailMemory] = None,
    ) -> RolloutResult:
        """Deterministic mean-field rollout from the all-mask state.

        Returns the denoiser distribution of the final pass (t = 1/T), before
        any update to zero noise. Teacher-free: no label enters this function.
        ``memory`` may pass an existing static cache built from the same
        boundary state and parameters (it is valid within a rollout).
        """
        cfg = self.cfg
        T = num_passes or cfg.num_passes
        t_sched = [(T - p) / T for p in range(T)]
        K = boundary.num_candidates
        device = boundary.h_slot_base.device
        active = torch.arange(K, device=device)
        mu = initial_soft_state(K, dtype=boundary.h_slot_base.dtype, device=device)
        do_prune = allow_pruning and pruning_active(K, mode, cfg.pruning_enabled, cfg.prune_min_k, full_support)
        passes: List[PassRecord] = []
        trace: List[torch.Tensor] = []
        pruned = False

        for p_idx, t in enumerate(t_sched):
            use_mem = p_idx > 0 or cfg.first_pass_detail_memory
            if use_mem:
                memory = self.ensure_memory(boundary, memory, active)
            trace.append(active.clone())
            x = self.boundary_inputs(boundary, active, mu[active])
            z = self.slot_pass(x, boundary, active, memory if use_mem else None)
            log_p = structured_log_probs(z)
            passes.append(PassRecord(t=t, active=active.clone(), z=z, log_p=log_p, used_memory=use_mem))
            if p_idx == T - 1:
                break
            s_next = t_sched[p_idx + 1]
            p_slot = log_p[1:].exp()
            if cfg.feedback_stop_gradient:
                p_slot = p_slot.detach()
            # update soft states of the current active set from the same old state
            mu = mu.clone()
            mu[active] = mean_field_update(mu[active], p_slot, t, s_next)
            if do_prune and p_idx < len(cfg.prune_schedule):
                cap = cfg.prune_schedule[p_idx]
                keep = select_survivors(z.detach(), cap)
                if keep.numel() < active.numel():
                    pruned = True
                active = active[keep]
        return RolloutResult(passes=passes, pruned=pruned, survivor_trace=trace, memory=memory, final_mu=mu)

    # ------------------------------------------------------------------
    def corrupted_view(
        self,
        boundary: BoundaryState,
        state: torch.Tensor,
        memory: Optional[DetailMemory] = None,
    ):
        """One denoiser evaluation on a hard corrupted state (training).

        Full-mask states use first-pass slot-only visibility; partially revealed
        states read static detail memory. Returns compatibility-constrained
        structured log-probs ``[K + 1]`` and the (possibly built) memory.
        """
        K = boundary.num_candidates
        device = boundary.h_slot_base.device
        active = torch.arange(K, device=device)
        full_mask = bool((state == MASK).all())
        use_mem = (not full_mask) or self.cfg.first_pass_detail_memory
        if use_mem:
            memory = self.ensure_memory(boundary, memory, active)
        mu = hard_to_soft(state, dtype=boundary.h_slot_base.dtype)
        x = self.boundary_inputs(boundary, active, mu)
        z = self.slot_pass(x, boundary, active, memory if use_mem else None)
        compat = compatibility_mask(state)
        return structured_log_probs(z, compat=compat), z, memory
