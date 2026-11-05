"""Fixed-budget sampling mixture and replayable trace (AGENTS.md 5.1).

Each effective batch of 32 draws 8 records from each of the four buckets.
Sampling is with replacement when necessary. The trace is deterministic for a
seed and is meant to be frozen and replayed by every controlled comparison.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Dict, List, Sequence

BUCKETS = ("openjev_choice", "openjev_noul", "openjev_score", "toolret_choice")


def build_trace(
    bucket_ids: Dict[str, Sequence[str]],
    num_updates: int = 3125,
    per_bucket_per_batch: int = 8,
    seed: int = 0,
) -> List[List[Dict[str, str]]]:
    """Return ``num_updates`` batches; each batch a list of {bucket, rid}.

    Within a bucket, records are drawn from successive shuffled epochs (each
    unique record used before any repeats); this is sampling with
    replacement across epochs.
    """
    rng = random.Random(seed)
    streams = {}
    for b in BUCKETS:
        ids = list(bucket_ids.get(b, []))
        if not ids:
            raise ValueError(f"bucket {b} is empty; use the documented fallback source")
        streams[b] = {"ids": ids, "order": [], "pos": 0}

    def draw(b: str) -> str:
        s = streams[b]
        if s["pos"] >= len(s["order"]):
            s["order"] = list(s["ids"])
            rng.shuffle(s["order"])
            s["pos"] = 0
        r = s["order"][s["pos"]]
        s["pos"] += 1
        return r

    trace = []
    for _ in range(num_updates):
        batch = [{"bucket": b, "rid": draw(b)} for b in BUCKETS for _ in range(per_bucket_per_batch)]
        rng.shuffle(batch)
        trace.append(batch)
    return trace


def trace_stats(trace: List[List[Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for b in BUCKETS:
        c = Counter(x["rid"] for batch in trace for x in batch if x["bucket"] == b)
        n = sum(c.values())
        out[b] = {
            "presentations": n,
            "unique": len(c),
            "repetition_rate": (n - len(c)) / n if n else 0.0,
            "max_repeats": max(c.values()) if c else 0,
        }
    return out


def save_trace(trace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f)

