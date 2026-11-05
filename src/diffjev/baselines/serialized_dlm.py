"""Serialized one-pass Diffusion Language Model (DLM) baseline (AGENTS.md sections 2, 6, 7).

Used for:
- Serialized DLM baseline on the ``LLaDA-8B-Base`` anchor (Exp 1-5).
- Exact-checkpoint ``Nemotron-Labs-Diffusion`` 3B/8B/14B serialized one-pass DLM
  controls in Exp 5 (adapted separately from the same initial checkpoint with
  the same final-four-layer-plus-normalization training policy).

Design:
- Concatenates the shared query and all candidate options into a single
  serialized sequence: ``[query, [MASK_1], option_1, ..., [MASK_K], option_K]``
  with ordinary sequential positions ``0 .. N-1``.
- Runs a single bidirectional pass across all ``L`` layers (never inflates cost
  with repeated full-prompt passes).
- Reads out each candidate's ``[MASK_i]`` slot with the exact same restricted
  Yes/No partition head (``z_i = logit_i(Yes) - logit_i(No)``) and
  ``structured_log_probs``.
- Explicitly checks context limits without truncating away gold options.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ..backbones.base import BackboneAdapter
from ..losses import brier, cumulative_brier, task_ce
from ..probability import forced_choice_log_probs, score_readout, structured_log_probs
from ..pruning import top1_by_identity
from ..tasks import REFERENCE_ID, DecisionExample
from ..training import apply_freeze_policy
from .base import BaselineAdapter, BaselineResult


@dataclass
class SerializedPrompt:
    input_ids: torch.Tensor  # [1, N]
    position_ids: torch.Tensor  # [1, N]
    slot_indices: torch.Tensor  # [K]
    candidate_ids: List[str]


def pack_serialized_dlm(
    example: DecisionExample, mask_token_id: int, device: torch.device
) -> SerializedPrompt:
    ids: List[int] = list(example.query_ids)
    slot_indices: List[int] = []
    cids: List[str] = []
    for cand in example.candidates:
        slot_indices.append(len(ids))
        ids.append(int(mask_token_id))
        ids.extend(int(t) for t in cand.token_ids)
        cids.append(cand.cid)
    N = len(ids)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    pos_ids = torch.arange(N, dtype=torch.long, device=device)[None, :]
    slots = torch.tensor(slot_indices, dtype=torch.long, device=device)
    return SerializedPrompt(input_ids=input_ids, position_ids=pos_ids, slot_indices=slots, candidate_ids=cids)


class SerializedDLMBaseline( nn.Module, BaselineAdapter):
    """One-pass serialized DLM with one Yes/No slot per candidate."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        name: str = "serialized_dlm",
        max_context_tokens: int = 262144,
        readout_temperature: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.name = name
        self.max_context_tokens = max_context_tokens
        self.readout_temperature = readout_temperature or {"choice": 1.0, "noul": 1.0, "score": 1.0}
        from ..config import DiffJevConfig
        self.cfg = DiffJevConfig(num_passes=1, use_mask_loss=False,
                                readout_temperature=self.readout_temperature)

    def forward_slots(self, prompt: SerializedPrompt) -> torch.Tensor:
        """Run one bidirectional forward pass over the serialized prompt; return ``z [K]``."""
        b = self.backbone
        N = int(prompt.input_ids.shape[1])
        device = prompt.input_ids.device
        allowed = b.attention_visibility(N, device)
        lower = b.lower_layer_indices(4) if b.num_layers > 4 else []
        top = b.top_layer_indices(min(4, b.num_layers))
        lower_grad = any(
            p.requires_grad for i in lower for p in b.layer_modules()[i].parameters()
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and lower_grad):
            h = b.embed(prompt.input_ids)
            if lower:
                h = b.run_layers_dense(h, prompt.position_ids, allowed, lower)
        if lower and not lower_grad and torch.is_grad_enabled():
            h = h.detach()
        h = b.run_layers_dense(h, prompt.position_ids, allowed, top)
        h_slots = b.final_norm(h[0, prompt.slot_indices])
        return b.yes_no_logit_diff(h_slots)

    def training_losses(
        self, example: DecisionExample, generator=None, brier_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Fine-tuning loss matching DiffJev's final-4-layer + norm policy."""
        if example.target is None:
            raise ValueError("training_losses requires example.target")
        device = self.backbone.input_embedding_module().weight.device
        prompt = pack_serialized_dlm(example, self.backbone.mask_token_id, device)
        z = self.forward_slots(prompt)
        work_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
        log_p = structured_log_probs(z.to(work_dtype))
        target = example.target.to(device=device, dtype=work_dtype)
        ce = task_ce(log_p, target)
        total = ce
        out = {"L_task_CE": ce}
        brier_weight = self.cfg.brier_weight
        if brier_weight > 0:
            p = log_p.exp()
            lb = cumulative_brier(p, target) if example.mode == "score" and self.cfg.score_cumulative_brier else brier(p, target)
            out["L_Brier"] = lb
            total = total + brier_weight * lb
        out["total"] = total
        return out

    @torch.no_grad()
    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        limit = max_context_tokens or self.max_context_tokens
        device = self.backbone.input_embedding_module().weight.device
        t0 = time.perf_counter()
        prompt = pack_serialized_dlm(example, self.backbone.mask_token_id, device)
        N = int(prompt.input_ids.shape[1])
        if N > limit:
            return BaselineResult(
                baseline_name=self.name,
                mode=example.mode,
                candidate_ids=[c.cid for c in example.candidates],
                top1=None,
                total_tokens=N,
                failed=True,
                failure_reason="context_limit",
            )
        try:
            z = self.forward_slots(prompt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return BaselineResult(
                    baseline_name=self.name,
                    mode=example.mode,
                    candidate_ids=prompt.candidate_ids,
                    top1=None,
                    total_tokens=N,
                    failed=True,
                    failure_reason="oom",
                )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        temp = float(self.readout_temperature.get(example.mode, 1.0))
        log_p = structured_log_probs(z, temperature=temp)
        p = log_p.exp().detach().cpu()
        prob_map = {cid: float(prob) for cid, prob in zip(prompt.candidate_ids, p[1:])}
        prob_map[REFERENCE_ID] = float(p[0].item())
        top1 = top1_by_identity(p, [REFERENCE_ID] + prompt.candidate_ids)
        if example.forced_choice:
            fc = forced_choice_log_probs(z, temperature=temp).exp().detach().cpu()
            top1 = top1_by_identity(fc, prompt.candidate_ids)
            prob_map = {cid: float(prob) for cid, prob in zip(prompt.candidate_ids, fc)}
        exp_score = None
        p_yes = None
        if example.mode == "noul":
            p_yes = float(torch.sigmoid(z[0].double() / temp).item())
        elif example.mode == "score" and example.bin_values is not None:
            vals = torch.tensor(example.bin_values, dtype=log_p.dtype, device=log_p.device)
            _, exp_t, _ = score_readout(log_p, vals)
            exp_score = float(exp_t.item())
        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=prompt.candidate_ids,
            top1=top1,
            prob_by_id=prob_map,
            expected_score=exp_score,
            p_yes=p_yes,
            local_gpu_ms=elapsed_ms,
            total_tokens=N,
        )

