"""External / reference baselines (AGENTS.md section 7):

- ``JevAPIBaseline``      : Official commercial Jev API client. Reads
                            ``JEV_API_KEY`` via ``load_jev_api_key()`` without
                            ever displaying or logging it; records API latency
                            strictly separately from local GPU latency.
- ``JevaBaseline``        : Single-pass set baseline ``Jeva-4B`` (loads from a
                            pinned local checkpoint or reports unavailable).
- ``OptionalDjevBaseline``: Optional ``djev`` cross-backbone baseline (requires
                            an explicitly verified public revision; never
                            invents a ``MetaEncoder-4B`` checkpoint).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

import torch

from ..pruning import top1_by_identity
from ..tasks import REFERENCE_ID, DecisionExample
from ..utils import load_jev_api_key
from .base import BaselineAdapter, BaselineResult


def _native_question(example):
    state = example.meta.get("native_state", example.meta.get("state", example.meta.get("query_text")))
    if state is None or any(c.text is None for c in example.candidates):
        raise ValueError("External baselines require original semantic text")
    native = example.meta.get("question")
    instruction = native.get("instructions", "") if isinstance(native, dict) else (native or example.meta["query_text"])
    if example.mode == "choice":
        aliases = [c.alias or f"option_{i+1}" for i, c in enumerate(example.candidates)]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Duplicate option aliases")
        mapping = dict(zip(aliases, [c.cid for c in example.candidates]))
        question = {"type": "choice", "instructions": instruction,
                    "criteria": {a: c.text for a, c in zip(aliases, example.candidates)}}
    elif example.mode == "noul":
        mapping = {"true": example.candidates[0].cid, "false": REFERENCE_ID}
        question = native if isinstance(native, dict) else {"type": "noul", "instructions": instruction}
    else:
        mapping = {str(i): c for i, c in enumerate([REFERENCE_ID]+[c.cid for c in example.candidates])}
        question = {"type": "score", "instructions": instruction,
                    "criteria": [d for _, d in example.meta["bins"]]}
    return state, question, mapping


class JevAPIBaseline(BaselineAdapter):
    """Official systemone protocol; explicit workflow opt-in is required for calls."""
    def __init__(self, endpoint="https://api.typesafe.ai/v1/systemone",
                 model_version="jev-1.13.0", transport=None, tokenizer=None):
        self.name = "jev_api_"+model_version
        self.endpoint, self.model_version = endpoint, model_version
        self.transport, self.tokenizer = transport, tokenizer

    def has_api_key(self):
        return load_jev_api_key() is not None

    def predict(self, example, max_context_tokens=None):
        cids = [c.cid for c in example.candidates]
        key = load_jev_api_key()
        if not key and self.transport is None:
            return BaselineResult(self.name, example.mode, cids, None, failed=True,
                                  failure_reason="missing_jev_api_key")
        state, question, mapping = _native_question(example)
        payload = {"model": self.model_version, "state": state, "questions": {"decision": question}}
        started = time.perf_counter()
        try:
            if self.transport:
                response = self.transport(payload)
            else:
                request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer "+key})
                with urllib.request.urlopen(request, timeout=60) as result:
                    response = json.load(result)
            from ..evaluation.official import load_official
            common, _, _ = load_official()
            parsed = common.parse_answer(response, "decision", example.mode)
            if parsed.get("parse_error"):
                raise ValueError(parsed["parse_error"])
        except Exception as error:
            return BaselineResult(self.name, example.mode, cids, None, failed=True,
                                  failure_reason="api_error:"+type(error).__name__)
        probabilities = parsed.get("probabilities")
        probs = {mapping[k]: float(v) for k,v in probabilities.items() if k in mapping} if probabilities else None
        p_yes = parsed.get("p_true")
        if p_yes is not None:
            probs = {cids[0]: p_yes, REFERENCE_ID: 1-p_yes}
        if probs and example.forced_choice:
            total = sum(probs.get(c,0) for c in cids)
            if total <= 0:
                raise ValueError("API supplied zero forced-choice mass")
            probs = {c: probs.get(c,0)/total for c in cids}
        top1 = mapping.get(str(parsed.get("pred")))
        if p_yes is not None:
            top1 = cids[0] if p_yes >= .5 else REFERENCE_ID
        return BaselineResult(self.name, example.mode, cids, top1, prob_by_id=probs,
            p_yes=p_yes, expected_score=parsed.get("expected_level"),
            api_latency_ms=1000*(time.perf_counter()-started), local_gpu_ms=None)


class JevaBaseline(BaselineAdapter):
    """Official LIQIIIII/Jeva-4B runtime, loaded locally with network disabled."""
    def __init__(self, local_dir=None, device="cuda", dtype=torch.bfloat16):
        self.name, self.local_dir = "jeva_4b", local_dir
        self.model = None
        if local_dir is not None:
            if not os.path.isdir(local_dir):
                raise FileNotFoundError(local_dir)
            from jeva import Jeva
            self.model = Jeva.from_pretrained(local_dir, device=device, dtype=dtype,
                local_files_only=True, merge_lora=True, cache="none", truncation="error")
            self.tokenizer = self.model.model.tokenizer

    @torch.inference_mode()
    def predict(self, example, max_context_tokens=None):
        cids = [c.cid for c in example.candidates]
        if self.model is None:
            return BaselineResult(self.name, example.mode, cids, None, failed=True,
                                  failure_reason="checkpoint_unavailable")
        state, question, mapping = _native_question(example)
        from jeva import Choice, Noul, Score
        if example.mode == "choice":
            q = Choice(question["instructions"], question["criteria"], allow_null=not example.forced_choice)
        elif example.mode == "noul":
            q = Noul(question["instructions"] + "\n" + str(question.get("criteria", "")))
        else:
            q = Score(question["instructions"], question["criteria"], values=example.bin_values)
        self.model.clear_cache()
        answer = self.model.system_one(state, {"decision": q}).answers["decision"]
        if example.mode == "choice":
            probs = {mapping[k]: p for k,p in answer.probabilities.items()}
            if not example.forced_choice:
                probs[REFERENCE_ID] = answer.null_probability
            top1 = mapping.get(answer.choice, REFERENCE_ID)
        elif example.mode == "noul":
            probs = {cids[0]: answer.noul, REFERENCE_ID: 1-answer.noul}
            top1 = cids[0] if answer.noul >= .5 else REFERENCE_ID
        else:
            probs = dict(zip([REFERENCE_ID]+cids, answer.candidate_probabilities))
            top1 = min(probs, key=lambda c: (-probs[c],c))
        return BaselineResult(self.name, example.mode, cids, top1, prob_by_id=probs,
            p_yes=answer.noul, expected_score=answer.score,
            local_gpu_ms=answer.latency_ms, total_tokens=answer.usage.requested_input_tokens)


class OptionalDjevBaseline(BaselineAdapter):
    """Optional ``djev`` baseline (never run unless public revision is verified)."""

    def __init__(self, verified_revision: Optional[str] = None):
        self.name = "djev_optional"
        self.verified_revision = verified_revision

    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=[c.cid for c in example.candidates],
            top1=None,
            failed=True,
            failure_reason="optional_unverified_revision" if not self.verified_revision else "not_loaded",
        )

