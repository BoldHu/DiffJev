"""Deterministic proportional stratification, independent of input row order."""
import random
from collections import defaultdict


def stratified_subset(rows, count, seed=0, stratum=lambda x: x.domain, identity=lambda x: x.qid):
    groups = defaultdict(list)
    for row in rows:
        groups[str(stratum(row))].append(row)
    rng = random.Random(seed)
    total = len(rows)
    count = min(count, total)
    quotas = {k: count * len(v) // max(1, total) for k, v in groups.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda k: (-(count * len(groups[k]) % max(1,total)), k))
    for k in order[:remaining]:
        quotas[k] += 1
    result = []
    for k in sorted(groups):
        pool = sorted(groups[k], key=identity)
        rng.shuffle(pool)
        result.extend(pool[:quotas[k]])
    return sorted(result, key=identity)
