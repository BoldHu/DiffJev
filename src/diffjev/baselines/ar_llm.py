"""Autoregressive LLM-as-Jev and exact-checkpoint Nemotron AR baseline.

Implements AGENTS.md sections 2.0, 6 (Exp 5), and 7:
- Separately adapted from the initial checkpoint with causal attention and the
  final-four-layer-plus-normalization training policy (never by switching
  attention modes on a checkpoint trained only as DiffJev).
- Full probability evaluation (Exp 1 / Exp 2): uses ONE causal prompt prefill
  and caches per-layer KV, then scores the **complete** token sequence of every
  valid candidate label (including multi-token labels ``[1] ... [K]`` or alias
  strings) along a suffix trie over the cached prompt KV. Never substitutes
  only the first token, and never runs ``K`` full prompt prefills.
- Fast top-1 timing mode (Exp 5): uses ONE causal prompt prefill and decodes
  one short valid decision label from the cached prompt KV without
  chain-of-thought or unnecessary full probability computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..backbones.base import BackboneAdapter
from ..pruning import top1_by_identity
from ..tasks import REFERENCE_ID, DecisionExample
from .base import BaselineAdapter, BaselineResult


@dataclass
class CausalSerializedPrompt:
    prompt_ids: torch.Tensor  # [1, N]
    position_ids: torch.Tensor  # [1, N]
    candidate_ids: List[str]
    label_token_seqs: List[List[int]]  # [K] token sequences for each candidate label
    ref_token_seq: List[int]  # token sequence for reference/No/lowest bin


def _default_label_tokens(tokenizer, idx: int, alias: Optional[str]) -> List[int]:
    text = alias if alias else f"[{idx + 1}]"
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            toks = list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            toks = list(tokenizer.encode(text))
        if toks:
            return [int(t) for t in toks]
    # Fallback synthetic multi-token encoding for numeric indices >= 4 when
    # using a tiny mock tokenizer without digit vocabulary
    return [1 + (idx // 8), 2 + (idx % 8)] if idx >= 4 else [1 + idx]


def pack_causal_prompt(
    example: DecisionExample,
    tokenizer,
    yes_token_id: int,
    no_token_id: int,
    device: torch.device,
) -> CausalSerializedPrompt:
    if example.meta.get("query_text") is not None:
        from ..tasks import _enc
        labels = [(c.alias or f"[{i+1}]") for i, c in enumerate(example.candidates)]
        ref_label = "[0]"
        if example.mode == "noul":
            labels, ref_label = ["Yes"], "No"
        if len(set(labels + [ref_label])) != len(labels) + 1:
            raise ValueError("AR answer aliases must be unique")
        options = [f"{label}: {c.text}" for label, c in zip(labels, example.candidates)]
        ref_text = (example.meta["bins"][0][1] if example.mode == "score" else
                    "No" if example.mode == "noul" else "None of these")
        if not example.forced_choice:
            options.insert(0, f"{ref_label}: {ref_text}")
        text = (example.meta["query_text"] + "\nAlternatives:\n" + "\n".join(options)
                + "\nReturn exactly one alternative label, without explanation.")
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        else:
            ids = _enc(tokenizer, text + "\nAnswer: ")
        eos = getattr(tokenizer, "eos_token_id", None)
        def label_tokens(label):
            return _enc(tokenizer, label) + ([int(eos)] if eos is not None else _enc(tokenizer, "\n"))
        seqs = [label_tokens(label) for label in labels]
        ref = label_tokens(ref_label)
        if len({tuple(s) for s in seqs + [ref]}) != len(seqs) + 1:
            raise ValueError("Tokenizer collapses distinct answer labels")
        return CausalSerializedPrompt(torch.tensor([ids], device=device),
            torch.arange(len(ids), device=device)[None], [c.cid for c in example.candidates], seqs, ref)
    ids: List[int] = list(example.query_ids)
    cids: List[str] = []
    label_seqs: List[List[int]] = []

    if example.mode == "noul":
        c = example.candidates[0]
        ids.extend(int(t) for t in c.token_ids)
        cids.append(c.cid)
        label_seqs.append([int(yes_token_id)])
        ref_seq = [int(no_token_id)]
    else:
        for i, cand in enumerate(example.candidates):
            lbl_toks = _default_label_tokens(tokenizer, i, cand.alias)
            ids.extend(lbl_toks)
            ids.extend(int(t) for t in cand.token_ids)
            cids.append(cand.cid)
            label_seqs.append(lbl_toks)
        ref_seq = [int(no_token_id)]

    N = len(ids)
    return CausalSerializedPrompt(
        prompt_ids=torch.tensor(ids, dtype=torch.long, device=device)[None, :],
        position_ids=torch.arange(N, dtype=torch.long, device=device)[None, :],
        candidate_ids=cids,
        label_token_seqs=label_seqs,
        ref_token_seq=ref_seq,
    )


class AutoregressiveJevBaseline(nn.Module, BaselineAdapter):
    """Causal AR baseline with 1-prefill cached KV and multi-token label scoring."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        tokenizer=None,
        name: str = "serialized_ar",
        max_context_tokens: int = 262144,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer or getattr(backbone, "tokenizer", None)
        self.name = name
        self.max_context_tokens = max_context_tokens
        from ..config import DiffJevConfig
        self.cfg = DiffJevConfig(num_passes=1, use_mask_loss=False, brier_weight=0.0)

    def _causal_prefill(
        self, prompt_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run one causal prefill across all L layers; return final last-token
        logits ``[V]`` and per-layer cached ``(K, V)``."""
        b = self.backbone
        N = int(prompt_ids.shape[1])
        device = prompt_ids.device
        causal = b.attention_visibility(N, device, causal=True)
        h = b.embed(prompt_ids)
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        lower = set(b.lower_layer_indices(4)) if b.num_layers > 4 else set()
        lower_grad = any(
            p.requires_grad for i in lower for p in b.layer_modules()[i].parameters()
        )
        for l in range(b.num_layers):
            in_lower = l in lower
            with torch.set_grad_enabled(
                torch.is_grad_enabled() and (lower_grad if in_lower else True)
            ):
                k, v = b.layer_kv(l, h, position_ids)
                kv_cache.append((k, v))
                h = b.layer_forward(l, h, position_ids, k, v, causal)
            if in_lower and not lower_grad and torch.is_grad_enabled():
                h = h.detach()
        last_norm = b.final_norm(h[:, -1:])
        last_logits = b.full_logits(last_norm)[0, 0]  # [V]
        return last_logits, kv_cache

    def _step_with_cache(
        self,
        token_ids: Sequence[int],
        start_pos: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        append_cache: bool = False,
    ) -> torch.Tensor:
        """Run a short continuation ``token_ids`` over cached prompt KV.

        Returns log-probabilities ``[len(token_ids), V]`` predicted at each
        continuation step.
        """
        b = self.backbone
        device = kv_cache[0][0].device
        m = len(token_ids)
        cont_ids = torch.tensor(list(token_ids), dtype=torch.long, device=device)[None, :]
        cont_pos = torch.arange(start_pos, start_pos + m, dtype=torch.long, device=device)[None, :]
        N_prompt = int(kv_cache[0][0].shape[2])
        # Each continuation row i can attend to all prompt keys + continuation keys <= i
        prompt_allow = torch.ones(1, m, N_prompt, dtype=torch.bool, device=device)
        cont_allow = torch.tril(torch.ones(1, m, m, dtype=torch.bool, device=device))
        allowed = torch.cat([prompt_allow, cont_allow], dim=2)

        h = b.embed(cont_ids)
        for l in range(b.num_layers):
            kc, vc = b.layer_kv(l, h, cont_pos)
            kp, vp = kv_cache[l]
            k = torch.cat([kp, kc], dim=2)
            v = torch.cat([vp, vc], dim=2)
            h = b.layer_forward(l, h, cont_pos, k, v, allowed)
            if append_cache:
                kv_cache[l] = (k, v)
        logits = b.full_logits(b.final_norm(h))[0]  # [m, V]
        return F.log_softmax(logits.float(), dim=-1)

    def score_label_sequence(
        self,
        seq: Sequence[int],
        first_log_probs: torch.Tensor,
        prompt_len: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Score a complete (single- or multi-token) label sequence exactly."""
        total = first_log_probs[int(seq[0])]
        if len(seq) > 1:
            cont_lp = self._step_with_cache(seq[:-1], prompt_len, kv_cache)
            for step_i, tok_id in enumerate(seq[1:]):
                total = total + cont_lp[step_i, int(tok_id)]
        return total

    def score_all_candidates(
        self, packed: CausalSerializedPrompt
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cand_log_scores [K], ref_log_score)`` from 1 prompt prefill."""
        last_logits, kv_cache = self._causal_prefill(packed.prompt_ids, packed.position_ids)
        first_lp = F.log_softmax(last_logits.float(), dim=-1)
        N = int(packed.prompt_ids.shape[1])

        # Deduplicate multi-token prefixes across candidates so shared prefixes
        # are evaluated at most once
        prefix_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        scores: List[torch.Tensor] = []
        for seq in packed.label_token_seqs:
            t_seq = tuple(int(x) for x in seq)
            if len(t_seq) == 1:
                scores.append(first_lp[t_seq[0]])
            else:
                prefix = t_seq[:-1]
                if prefix not in prefix_cache:
                    prefix_cache[prefix] = self._step_with_cache(prefix, N, kv_cache)
                cont_lp = prefix_cache[prefix]
                s = first_lp[t_seq[0]]
                for idx_step, tok_id in enumerate(t_seq[1:]):
                    s = s + cont_lp[idx_step, tok_id]
                scores.append(s)
        cand_scores = torch.stack(scores, dim=0)
        ref_score = self.score_label_sequence(packed.ref_token_seq, first_lp, N, kv_cache)
        return cand_scores, ref_score

    def training_losses(self, example: DecisionExample, generator=None) -> Dict[str, torch.Tensor]:
        """Supervised cross-entropy loss for adapting the AR baseline on the same trace."""
        if example.target is None:
            raise ValueError("training_losses requires example.target")
        b = self.backbone
        device = b.input_embedding_module().weight.device
        packed = pack_causal_prompt(
            example, self.tokenizer, b.yes_token_id, b.no_token_id, device
        )
        cand_scores, ref_score = self.score_all_candidates(packed)
        all_scores = torch.cat([ref_score.reshape(1), cand_scores], dim=0)
        log_p = F.log_softmax(all_scores, dim=-1)
        target = example.target.to(device=device, dtype=log_p.dtype)
        ce = -(target * log_p).sum()
        return {"L_task_CE": ce, "total": ce}

    @torch.no_grad()
    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        limit = max_context_tokens or self.max_context_tokens
        b = self.backbone
        device = b.input_embedding_module().weight.device
        t0 = time.perf_counter()
        packed = pack_causal_prompt(
            example, self.tokenizer, b.yes_token_id, b.no_token_id, device
        )
        N = int(packed.prompt_ids.shape[1])
        if N > limit:
            return BaselineResult(
                baseline_name=self.name,
                mode=example.mode,
                candidate_ids=packed.candidate_ids,
                top1=None,
                total_tokens=N,
                failed=True,
                failure_reason="context_limit",
            )
        try:
            cand_scores, ref_score = self.score_all_candidates(packed)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return BaselineResult(
                    baseline_name=self.name,
                    mode=example.mode,
                    candidate_ids=packed.candidate_ids,
                    top1=None,
                    total_tokens=N,
                    failed=True,
                    failure_reason="oom",
                )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        temperature = self.cfg.readout_temperature[example.mode]
        cand_scores, ref_score = cand_scores / temperature, ref_score / temperature
        if example.forced_choice:
            probs = F.softmax(cand_scores, dim=-1).detach().cpu()
            prob_map = {cid: float(p) for cid, p in zip(packed.candidate_ids, probs)}
            top1 = top1_by_identity(probs, packed.candidate_ids)
        else:
            all_scores = torch.cat([ref_score.reshape(1), cand_scores], dim=0)
            probs = F.softmax(all_scores, dim=-1).detach().cpu()
            prob_map = {cid: float(p) for cid, p in zip(packed.candidate_ids, probs[1:])}
            prob_map[REFERENCE_ID] = float(probs[0].item())
            top1 = top1_by_identity(probs, [REFERENCE_ID] + packed.candidate_ids)

        p_yes = prob_map.get(packed.candidate_ids[0]) if example.mode == "noul" else None
        exp_score = None
        if example.mode == "score" and example.bin_values is not None:
            all_scores = torch.cat([ref_score.reshape(1), cand_scores], dim=0)
            sp = F.softmax(all_scores, dim=-1).detach().cpu()
            vals = torch.tensor(example.bin_values, dtype=sp.dtype)
            exp_score = float((sp * vals).sum().item())

        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=packed.candidate_ids,
            top1=top1,
            prob_by_id=prob_map,
            expected_score=exp_score,
            p_yes=p_yes,
            local_gpu_ms=elapsed_ms,
            total_tokens=N,
        )

    @torch.no_grad()
    def predict_top1(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        """Exp 5 timing mode: 1 prompt prefill + short valid label decoding via trie."""
        limit = max_context_tokens or self.max_context_tokens
        b = self.backbone
        device = b.input_embedding_module().weight.device
        t0 = time.perf_counter()
        packed = pack_causal_prompt(
            example, self.tokenizer, b.yes_token_id, b.no_token_id, device
        )
        N = int(packed.prompt_ids.shape[1])
        if N > limit:
            return BaselineResult(
                baseline_name=self.name,
                mode=example.mode,
                candidate_ids=packed.candidate_ids,
                top1=None,
                total_tokens=N,
                failed=True,
                failure_reason="context_limit",
            )
        try:
            last_logits, kv_cache = self._causal_prefill(
                packed.prompt_ids, packed.position_ids
            )
            # Constrained greedy trie decoding over valid candidate label sequences
            active_indices = list(range(len(packed.label_token_seqs)))
            generated: List[int] = []
            max_len = max(len(s) for s in packed.label_token_seqs)
            cur_logits = last_logits
            for depth in range(max_len):
                valid_next = sorted(
                    {
                        packed.label_token_seqs[i][depth]
                        for i in active_indices
                        if len(packed.label_token_seqs[i]) > depth
                    }
                )
                if not valid_next:
                    break
                valid_t = torch.tensor(valid_next, dtype=torch.long, device=device)
                best_tok = int(valid_t[torch.argmax(cur_logits[valid_t])].item())
                generated.append(best_tok)
                active_indices = [
                    i
                    for i in active_indices
                    if len(packed.label_token_seqs[i]) > depth
                    and packed.label_token_seqs[i][depth] == best_tok
                ]
                if len(active_indices) == 1 and len(packed.label_token_seqs[active_indices[0]]) <= depth + 1:
                    break
                if depth + 1 < max_len:
                    cont_lp = self._step_with_cache([best_tok], N + depth, kv_cache, append_cache=True)
                    cur_logits = cont_lp[-1]
            chosen_cid = packed.candidate_ids[active_indices[0]] if active_indices else packed.candidate_ids[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return BaselineResult(
                    baseline_name=self.name,
                    mode=example.mode,
                    candidate_ids=packed.candidate_ids,
                    top1=None,
                    total_tokens=N,
                    failed=True,
                    failure_reason="oom",
                )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=packed.candidate_ids,
            top1=chosen_cid,
            prob_by_id=None,  # top-1 timing mode does not compute full distribution
            local_gpu_ms=elapsed_ms,
            total_tokens=N + len(generated),
            meta={"prompt_tokens":N,"decoded_tokens":max(0,len(generated)-1)},
        )

