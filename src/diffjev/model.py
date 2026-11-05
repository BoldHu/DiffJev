"""``DiffJevModel``: wires packing, branch encoding, co-denoising and readout.

The model owns **no** parameters of its own: every parameter belongs to the
wrapped backbone adapter (AGENTS.md section 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from .backbones.base import BackboneAdapter
from .config import DiffJevConfig
from .denoiser import RolloutResult, SlotDenoiser
from .diffusion import clean_labels, corrupt, sample_clean_action, sample_t
from .encoder import BoundaryState, encode_branches
from .losses import brier, cumulative_brier, mask_loss, task_ce
from .packing import PackedInput, pack_branches
from .probability import forced_choice_log_probs, score_readout, structured_log_probs
from .pruning import gold_survived, top1_by_identity
from .tasks import REFERENCE_ID, DecisionExample, DecisionOutput


class DiffJevModel(nn.Module):
    def __init__(self, backbone: BackboneAdapter, cfg: Optional[DiffJevConfig] = None):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg or DiffJevConfig()
        self.denoiser = SlotDenoiser(backbone, self.cfg)

    # ------------------------------------------------------------------
    def pack(self, example: DecisionExample, max_option_len: Optional[int] = None) -> PackedInput:
        return pack_branches(
            example.query_ids,
            example.candidates,
            mask_token_id=self.backbone.mask_token_id,
            pad_token_id=self.backbone.pad_token_id,
            position_mode=self.cfg.position_mode,
            max_option_len=max_option_len,
        )

    def _lower_requires_grad(self) -> bool:
        b = self.backbone
        mods = [b.input_embedding_module()] + [b.layer_modules()[i] for i in b.lower_layer_indices(self.cfg.num_top_layers)]
        return any(p.requires_grad for m in mods for p in m.parameters())

    def encode(self, packed: PackedInput) -> BoundaryState:
        limit = getattr(self.backbone, "max_context_tokens", None)
        if limit is not None and max(packed.query_len, int(packed.branch_pos.max())+1) > limit:
            raise ValueError("context_limit: semantic branch position exceeds native context")
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._lower_requires_grad()):
            boundary = encode_branches(self.backbone, packed, self.cfg.num_top_layers, self.cfg.branch_chunk_size)
        return boundary

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        example: DecisionExample,
        full_support: bool = False,
        gold_id: Optional[str] = None,
        return_rollout: bool = False,
    ):
        """Teacher-free deterministic inference. ``example.target`` is never read;
        ``gold_id`` is used only to *record* stagewise survival (no rescue)."""
        packed = self.pack(example)
        boundary = self.encode(packed)
        ro = self.denoiser.rollout(boundary, example.mode, full_support=full_support)
        out = self.readout(example, packed, ro, gold_id)
        return (out, ro) if return_rollout else out

    def readout(
        self, example: DecisionExample, packed: PackedInput, ro: RolloutResult, gold_id: Optional[str]
    ) -> DecisionOutput:
        mode = example.mode
        temp = float(self.cfg.readout_temperature.get(mode, 1.0))
        fin = ro.final
        ids = [packed.candidate_ids[i] for i in fin.active.tolist()]
        log_p = structured_log_probs(fin.z, temperature=temp)
        p = log_p.exp().detach().cpu()
        z_cpu = (fin.z.float() if fin.z.dtype in (torch.float16, torch.bfloat16) else fin.z).detach().cpu()
        trace_ids = [[packed.candidate_ids[i] for i in a.tolist()] for a in ro.survivor_trace]
        survival = [gold_survived(t, gold_id) for t in trace_ids]
        out = DecisionOutput(
            mode=mode,
            candidate_ids=ids,
            candidate_probs=p[1:],
            reference_prob=float(p[0].item()),
            probability_scope=ro.probability_scope,
            z_final=z_cpu,
            survivor_ids_per_pass=trace_ids,
            gold_survival=survival,
            top1=top1_by_identity(p, [REFERENCE_ID] + ids),
        )
        if example.forced_choice:
            fc = forced_choice_log_probs(fin.z, temperature=temp).exp().detach().cpu()
            out.forced_choice_probs = fc
            out.top1 = top1_by_identity(fc, ids)
        if mode == "noul":
            out.p_yes = float(torch.sigmoid(z_cpu[0].double() / temp).item())
        if mode == "score":
            vals = torch.tensor(example.bin_values, dtype=log_p.dtype, device=log_p.device)
            sp, exp_score, cdf = score_readout(log_p, vals)
            out.score_probs = sp.detach().cpu()
            out.expected_score = float(exp_score.item())
            out.score_cdf = cdf.detach().cpu()
        return out

    # ------------------------------------------------------------------
    def training_losses(
        self,
        example: DecisionExample,
        gen: torch.Generator,
        t: Optional[float] = None,
        noise: Optional[torch.Tensor] = None,
        action: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Composite structured-denoising objective for one source example.

        ``L_mask`` (corrupted view) + mean over passes of teacher-free task CE
        + ``brier_weight * L_Brier`` on the final unpruned rollout pass.
        ``t``, ``noise`` and ``action`` may be supplied to couple stochastic
        draws (e.g. across candidate permutations).
        """
        if example.target is None:
            raise ValueError("training requires a target distribution")
        cfg = self.cfg
        packed = self.pack(example)
        boundary = self.encode(packed)
        device = boundary.h_slot_base.device
        work_dtype = torch.float64 if boundary.h_slot_base.dtype == torch.float64 else torch.float32
        target = example.target.to(device=device, dtype=work_dtype)
        K = packed.num_candidates
        losses: Dict[str, torch.Tensor] = {}
        total = boundary.h_slot_base.new_zeros((), dtype=work_dtype)
        memory = None

        if cfg.use_mask_loss:
            t_val = sample_t(gen, cfg.full_mask_prob, cfg.t_min) if t is None else float(t)
            a = sample_clean_action(target, gen) if action is None else int(action)
            labels = clean_labels(a, K, device=device)
            u = torch.rand(K, generator=gen).to(device) if noise is None else noise.to(device)
            state = corrupt(labels, t_val, u)
            log_p_c, _, memory = self.denoiser.corrupted_view(boundary, state, memory)
            lm = mask_loss(log_p_c.to(work_dtype), labels, state, t_val)
            losses["L_mask"] = lm
            losses["t"] = torch.tensor(t_val, device=device)
            total = total + lm

        # teacher-free rollout: no pruning in training (K <= 32 by default)
        ro = self.denoiser.rollout(boundary, example.mode, full_support=True, allow_pruning=False, memory=memory)
        if cfg.use_task_ce:
            ce = torch.stack([task_ce(rec.log_p.to(work_dtype), target) for rec in ro.passes]).mean()
            losses["L_task_CE"] = ce
            total = total + ce
        if cfg.brier_weight > 0:
            p_final = ro.final.log_p.to(work_dtype).exp()
            if example.mode == "score" and cfg.score_cumulative_brier:
                lb = cumulative_brier(p_final, target)
            else:
                lb = brier(p_final, target)
            losses["L_Brier"] = lb
            total = total + cfg.brier_weight * lb
        losses["total"] = total
        return losses
