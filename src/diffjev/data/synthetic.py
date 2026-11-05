"""Synthetic toy tasks for the mock backbone (no download, no research meaning).

choice : query "choose w_k"; exactly one option contains w_k (or none, giving a
         reference target); options have heterogeneous lengths.
noul   : proposition contains w_k or not.
score  : number of times w_k occurs in a text maps to an ordered bin.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ..packing import Candidate
from ..tasks import REFERENCE_ID, DecisionExample, build_choice, build_noul, build_score

N_WORDS = 48


def _words(rng: random.Random, n: int, exclude: Sequence[int] = ()) -> List[str]:
    pool = [i for i in range(N_WORDS) if i not in set(exclude)]
    return [f"w{rng.choice(pool)}" for _ in range(n)]


def synthetic_choice(
    tokenizer, rng: random.Random, K: int, null_prob: float = 0.0, max_len: int = 5, forced_choice: bool = False
) -> DecisionExample:
    key = rng.randrange(N_WORDS)
    is_null = rng.random() < null_prob
    gold = None if is_null else rng.randrange(K)
    options = []
    for i in range(K):
        n = rng.randint(1, max_len)
        toks = _words(rng, n, exclude=[key])
        if i == gold:
            toks[rng.randrange(n)] = f"w{key}"
        options.append((f"c{i:04d}", "option " + " ".join(toks)))
    target = {REFERENCE_ID: 1.0} if gold is None else {f"c{gold:04d}": 1.0}
    return build_choice(tokenizer, f"choose w{key}", options, target, forced_choice=forced_choice)


def synthetic_noul(tokenizer, rng: random.Random, max_len: int = 5) -> DecisionExample:
    key = rng.randrange(N_WORDS)
    n = rng.randint(1, max_len)
    toks = _words(rng, n, exclude=[key])
    yes = rng.random() < 0.5
    if yes:
        toks[rng.randrange(n)] = f"w{key}"
    return build_noul(tokenizer, f"query w{key}", "option " + " ".join(toks), 1.0 if yes else 0.0)


def synthetic_score(tokenizer, rng: random.Random, M: int = 5) -> DecisionExample:
    key = rng.randrange(N_WORDS)
    count = rng.randrange(M)
    toks = _words(rng, 4, exclude=[key]) + [f"w{key}"] * count
    rng.shuffle(toks)
    bins = [(float(m), f"bin w{m}") for m in range(M)]
    rng.shuffle(bins)  # adapter must sort by numeric value
    soft = {float(count): 0.8}
    if count + 1 < M:
        soft[float(count + 1)] = 0.2
    else:
        soft[float(count)] = 1.0
    return build_score(tokenizer, "rubric", f"query w{key} " + " ".join(toks), bins, soft)


def synthetic_mixture(tokenizer, n: int, seed: int = 0, ks: Sequence[int] = (2, 3, 4, 6)) -> List[DecisionExample]:
    rng = random.Random(seed)
    out: List[DecisionExample] = []
    for i in range(n):
        r = i % 3
        if r == 0:
            out.append(synthetic_choice(tokenizer, rng, rng.choice(list(ks)), null_prob=0.15))
        elif r == 1:
            out.append(synthetic_noul(tokenizer, rng))
        else:
            out.append(synthetic_score(tokenizer, rng))
    return out

