"""Deterministic overlap audit, including conservative near-duplicate grouping."""
from collections import defaultdict
import hashlib
from .splits import normalize_text, UnionFind
from ..progress import progress


def audit_groups(records, evaluation, threshold=.90, bucket_limit=200):
    """4-band SimHash candidates, verified with word-shingle Jaccard.

    Approximate candidate generation is not a proof of zero overlap. Oversized
    buckets are quarantined; the audit algorithm and thresholds travel with data.
    """
    rows = list(records) + list(evaluation)
    uf, buckets, exact = UnionFind(), defaultdict(list), {}
    signatures, strings = {}, {}
    uncertain = set()
    for i, row in enumerate(progress(rows, "Overlap audit / source and evaluation records")):
        key = str(i)
        uf.find(key)
        text = normalize_text(row.get("state") or row["query"])
        strings[key] = text
        if text in exact:
            uf.union(key, exact[text])
            continue
        exact[text] = key
        words = text.split()
        if len(words) < 8:
            continue
        shingles = {" ".join(words[j:j+4]) for j in range(len(words)-3)}
        votes = [0]*64
        for gram in shingles:
            h = int.from_bytes(hashlib.blake2b(gram.encode(),digest_size=8).digest(),"little")
            for bit in range(64):
                votes[bit] += 1 if (h >> bit) & 1 else -1
        sig = sum(1 << b for b,v in enumerate(votes) if v >= 0)
        candidates = set()
        for band in range(4):
            bucket = buckets[(band, (sig >> (16*band)) & 65535)]
            if len(bucket) >= bucket_limit:
                uncertain.update(bucket)
                uncertain.add(key)
            else:
                candidates.update(bucket)
            bucket.append(key)
        for other in candidates:
            w = strings[other].split()
            grams = {" ".join(w[j:j+4]) for j in range(len(w)-3)}
            if len(shingles & grams)/max(1,len(shingles | grams)) >= threshold:
                uf.union(key, other)
    # Use a content hash rather than row number to make the split order-independent.
    group_texts = defaultdict(list)
    for i in range(len(rows)):
        group_texts[uf.find(str(i))].append(strings[str(i)])
    group_ids = {g: hashlib.sha256(min(ts).encode()).hexdigest() for g,ts in group_texts.items()}
    for i,row in enumerate(rows):
        row["audit_group"] = group_ids[uf.find(str(i))]
        if str(i) in uncertain:
            row["audit_uncertain"] = True
    return {"algorithm":"word4-simhash64-bands4-jaccard", "threshold":threshold,
            "bucket_limit":bucket_limit, "uncertain_records":len(uncertain),
            "groups":len(group_ids), "exhaustive":False}
