"""Strict parsers for the pinned public schemas, before tokenization.

Ambiguous records raise ValueError and are quarantined by the preparation workflow.
No labels, missing states, or negative certificates are inferred from model scores.
"""
import hashlib
import json
import math
import unicodedata
from collections import Counter

from ..tasks import REFERENCE_ID
from .splits import normalize_text


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def semantic_id(text):
    return "semantic:" + digest(semantic_key(text))


def semantic_key(text):
    # Code punctuation, numeric signs, and meaningful names are model semantics.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def distribution(values, n):
    p = [float(x) for x in values]
    if len(p) != n or any(not math.isfinite(x) or x < 0 for x in p) or abs(sum(p)-1) > 1e-5:
        raise ValueError("invalid native target distribution")
    return p


def benchmark_candidate_ids(record, aliases, texts):
    """Keep native alternatives distinct, including duplicate text occurrences.

    For duplicates, option_order supplies the immutable source option identity.
    This metadata is independent of display order, aliases, and gold labels; it
    is never inserted into query/option text or exposed to the backbone.
    """
    semantic = [semantic_id(text) for text in texts]
    counts = Counter(semantic)
    if max(counts.values(),default=0) < 2:
        return semantic, []
    order = record.get("option_order")
    if (not isinstance(order,list) or len(order)!=len(texts)
            or any(type(i) is not int for i in order)
            or sorted(order)!=list(range(len(texts)))):
        raise ValueError(f"{record['case_id']}: duplicate option texts require a valid canonical option_order")
    ids = [cid if counts[cid]==1 else cid+":source:"+digest(
        [record["slice"],record["family_id"],order[i]]) for i,cid in enumerate(semantic)]
    groups=[]
    for cid,count in counts.items():
        if count < 2:
            continue
        members=[i for i,key in enumerate(semantic) if key==cid]
        groups.append({"semantic_id":cid,
            "match":"identical_text" if len({texts[i] for i in members})==1 else "normalized_text_collision",
            "candidate_ids":[ids[i] for i in members],"native_aliases":[aliases[i] for i in members],
            "source_option_indices":[order[i] for i in members]})
    return ids,groups


def openjev(record, max_train_k=32):
    options = record["options"]
    if not options or len(options) > max_train_k:
        return None
    mode = record["task"]
    if mode not in ("choice", "noul", "score"):
        raise ValueError("unsupported OpenJev task")
    state, question = record.get("state", ""), record["question"]
    if not isinstance(state, str) or not isinstance(question, str) or not state.strip():
        raise ValueError("OpenJev needs the original state and question")
    texts = [str(o["name"]) + ("\n" + str(o["criteria"]) if o.get("criteria") else "") for o in options]
    ids = [str(o["id"]) for o in options]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source option IDs")
    answer = record["answer"]
    p = distribution(answer["probabilities"], len(options))
    if answer["kind"] not in ("hard","soft") or (answer["kind"] == "hard" and max(p) < 1-1e-6):
        raise ValueError("native answer kind disagrees with its probability distribution")
    base = dict(rid="openjev:"+digest(record), source="openjev", mode=mode,
                bucket="openjev_"+mode, domain=str(record.get("category", "general")),
                query=f"{state}\n\n{question}", state=state, question=question,
                template_id=str(record.get("template_id") or ""),
                is_soft=answer["kind"] == "soft", native_options=options,
                original_k=len(options), native_answer=answer)
    if mode == "score":
        values = record.get("score_values")
        if values is None or len(values) != len(options) or len(set(values)) != len(values):
            raise ValueError("score_values must uniquely identify every native bin")
        base.update(rubric=question, bins=sorted(zip(map(float, values), texts)),
                    target={float(v): mass for v, mass in zip(values, p)})
    elif mode == "noul":
        polarity = [normalize_text(o["name"]) for o in options]
        positive = [i for i, s in enumerate(polarity) if s in ("yes", "true", "是", "正确")]
        negative = [i for i, s in enumerate(polarity) if s in ("no", "false", "否", "错误")]
        if len(positive) != 1 or len(negative) != 1 or len(options) != 2:
            raise ValueError("noul label polarity is ambiguous; supply an audited mapping")
        base["query"] += "\nLabel definitions:\n" + "\n".join(texts)
        base.update(proposition=question+"\n"+texts[positive[0]], p_yes=p[positive[0]])
    else:
        # o0, o1 etc. are local aliases, never global candidate identities.
        cids = [semantic_id(t) for t in texts]
        if len(cids) != len(set(cids)):
            raise ValueError("duplicate semantic options require source audit")
        from .sources import AUDITED_NULL_PHRASES
        target, kept = {REFERENCE_ID: 0.0}, []
        for cid, text, mass, opt in zip(cids, texts, p, options):
            if opt.get("is_null") or (not opt.get("criteria") and normalize_text(opt["name"]) in AUDITED_NULL_PHRASES):
                target[REFERENCE_ID] += mass
            else:
                target[cid] = mass
                kept.append((cid, text))
        if not kept:
            return None
        base.update(options=kept, target=target,
                    null_certified=bool(record.get("null_certified", False)),
                    eligible_negative_ids=record.get("eligible_negative_ids", []))
    return base


def jevbench(record):
    q = record["question"]
    mode = q["type"]
    state = record.get("state")
    if state is None or state == "":
        raise ValueError("withheld JevBench state must be restored from the pinned source before evaluation")
    state_text = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if record.get("state_hash") and not hashlib.sha256(state_text.encode()).hexdigest().startswith(record["state_hash"]):
        raise ValueError("JevBench state_hash mismatch")
    instructions = q.get("instructions", "")
    base = dict(case_id=record["case_id"], family_id=record["family_id"],
                dataset=record["slice"], mode=mode,
                query=f"{state_text}\n\n{instructions}", native_case=record,
                state=state_text, native_state=state, question=q)
    criteria = q.get("criteria")
    gold = record["gold"]
    if mode == "choice":
        if not isinstance(criteria, dict) or gold not in criteria:
            raise ValueError("invalid JevBench choice criteria/gold")
        aliases = list(criteria)
        texts = [str(criteria[a]) for a in aliases]
        cids,duplicate_groups = benchmark_candidate_ids(record, aliases, texts)
        base.update(options=list(zip(cids, texts)), aliases=aliases,
                    gold_id=cids[aliases.index(gold)], forced_choice=True,
                    native_alias_by_id=dict(zip(cids, aliases)),
                    duplicate_option_groups=duplicate_groups)
    elif mode == "noul":
        if not isinstance(gold, bool):
            raise ValueError("expected native Boolean noul gold")
        if isinstance(criteria, dict):
            base["query"] += "\nLabel definitions: " + json.dumps(criteria, ensure_ascii=False)
            proposition = criteria.get("true", instructions)
        else:
            proposition = criteria or instructions
        base.update(proposition=str(proposition), p_yes=float(gold))
    elif mode == "score":
        if not isinstance(criteria, list) or not 0 <= int(gold) < len(criteria):
            raise ValueError("invalid ordinal criteria/gold")
        base.update(rubric=instructions, bins=list(enumerate(criteria)),
                    target={float(i): float(i == int(gold)) for i in range(len(criteria))})
    else:
        raise ValueError("not a core text benchmark task")
    return base
