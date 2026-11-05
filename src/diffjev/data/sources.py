"""Source dataset adapters and manifest preparation pipeline (AGENTS.md section 5).

Implements record mappers and manifest builders for:
- ``shenjunhao/OpenJevData-140k`` (``split="SFT"``, excluding RL pool)
- ``mangopy/ToolRet-Training-20w`` (``split="train"``, using original ``query``)
- Fallbacks (used ONLY when a bucket has missing/rejected records):
    - ``allenai/ai2_arc`` (``ARC-Easy`` / ``ARC-Challenge``, ``split="train"``) -> choice
    - ``google/boolq`` (``split="train"``) -> noul
    - ``nvidia/HelpSteer2`` (``split="train"``, helpfulness 0-4 rubric) -> score
- Evaluation suite (used for overlap auditing and Exp 1/2 evaluation only, never
  for tuning or negative mining):
    - ``Leanmcp/jevbench``
"""

from __future__ import annotations

import json
import os
import random
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch

from ..packing import Candidate
from ..tasks import (
    REFERENCE_ID,
    DecisionExample,
    build_choice,
    build_noul,
    build_score,
    merge_null_equivalent,
)
from .augmentation import DEFAULT_K_PROPOSAL, augment_menu
from .mixture import BUCKETS, build_trace, trace_stats
from .splits import normalize_text, split_records

SOURCES: Dict[str, Dict[str, Any]] = {
    "openjev": {
        "repo": "shenjunhao/OpenJevData-140k",
        "split": "SFT",
        "url": "https://huggingface.co/datasets/shenjunhao/OpenJevData-140k",
    },
    "toolret": {
        "repo": "mangopy/ToolRet-Training-20w",
        "split": "train",
        "url": "https://huggingface.co/datasets/mangopy/ToolRet-Training-20w",
    },
    "arc": {
        "repo": "allenai/ai2_arc",
        "configs": ["ARC-Easy", "ARC-Challenge"],
        "split": "train",
        "url": "https://huggingface.co/datasets/allenai/ai2_arc",
    },
    "boolq": {
        "repo": "google/boolq",
        "split": "train",
        "url": "https://huggingface.co/datasets/google/boolq",
    },
    "helpsteer2": {
        "repo": "nvidia/HelpSteer2",
        "split": "train",
        "url": "https://huggingface.co/datasets/nvidia/HelpSteer2",
    },
    "jevbench": {
        "repo": "Leanmcp/jevbench",
        "url": "https://huggingface.co/datasets/Leanmcp/jevbench",
    },
}

#: Exact normalized phrases that are semantically identical to the virtual null
#: reference. Arbitrary options merely containing "no" are NEVER treated as null.
AUDITED_NULL_PHRASES: Set[str] = {
    "none of the above",
    "none of these",
    "none of the options",
    "neither of these",
    "no valid option",
    "no valid tool",
    "none applicable",
}

POSITIVE_POLARITY_TOKENS: Set[str] = {
    "yes",
    "true",
    "1",
    "entailment",
    "entailed",
    "supported",
    "valid",
    "correct",
    "positive",
}
NEGATIVE_POLARITY_TOKENS: Set[str] = {
    "no",
    "false",
    "0",
    "contradiction",
    "not entailed",
    "unsupported",
    "invalid",
    "incorrect",
    "negative",
}

HELPSTEER2_RUBRIC = (
    "Evaluate the helpfulness of the response on an ordered 0 to 4 scale. "
    "Reference bin 0: The response is completely unhelpful, irrelevant, or empty."
)
HELPSTEER2_BINS: List[Tuple[float, str]] = [
    (0.0, "Score 0: Completely unhelpful, irrelevant, or empty."),
    (1.0, "Score 1: Minimally helpful; misses key parts of the instruction."),
    (2.0, "Score 2: Partially helpful; addresses the prompt with notable gaps."),
    (3.0, "Score 3: Helpful and accurate; satisfies most user requirements."),
    (4.0, "Score 4: Exceptionally helpful, thorough, and well-aligned."),
]


def load_local_dataset(name: str, data_root: str, split: Optional[str] = None):
    """Load an already-downloaded dataset snapshot from ``data_root/raw/<name>``.

    Never triggers network downloads. Supports HuggingFace ``save_to_disk``
    directories as well as local Parquet / JSONL / Arrow files.
    """
    path = os.path.join(data_root, "raw", name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path} missing; download the pinned snapshot first (see README.md)."
        )
    if name == "jevbench":
        from pathlib import Path
        names = {"aegis2", "aegis2_response", "atbench500", "banking77", "jailbreak_classification",
                 "medmcqa", "medqa_usmle", "mmlu_pro", "prompt_injections", "pubmedqa",
                 "scienceqa_text", "sst5"}
        files = sorted(p for p in (Path(path) / "cases").glob("*.jsonl") if p.stem in names)
        if {p.stem for p in files} != names:
            raise ValueError("Pinned JevBench original 12 text case files are required")
        rows = [json.loads(line) for p in files for line in p.read_text(encoding="utf-8").split("\n") if line]
        # Some licensed source states are withheld in the public manifest.
        restored = Path(path) / "restored_states.jsonl"
        states = {r["case_id"]: r["state"] for r in
                  (json.loads(l) for l in restored.read_text(encoding="utf-8").split("\n"))} if restored.exists() else {}
        for row in rows:
            if "state" not in row and row["case_id"] in states:
                row["state"] = states[row["case_id"]]
        if len(rows) != 8016 or len({r["family_id"] for r in rows}) != 5516:
            raise ValueError("Unexpected JevBench manifest: audit the pinned release before proceeding")
        return rows
    # Lightweight JSONL input also supports local audited exports.
    jsonl_file = os.path.join(path, f"{split or 'data'}.jsonl")
    if os.path.isfile(jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    from datasets import load_dataset, load_from_disk

    try:
        ds = load_from_disk(path)
        if split is not None and isinstance(ds, dict) and split in ds:
            return ds[split]
        return ds
    except Exception:
        return load_dataset(path, split=split)


def inspect_schema(ds, n: int = 3) -> Dict[str, object]:
    """Return column names and a few raw rows for schema verification."""
    if isinstance(ds, list):
        cols = sorted({k for row in ds[:n] for k in row.keys()})
        return {"columns": cols, "num_rows": len(ds), "examples": ds[:n]}
    cols = list(getattr(ds, "column_names", []))
    return {"columns": cols, "num_rows": len(ds), "examples": [ds[i] for i in range(min(n, len(ds)))]}


def _is_audited_null_option(opt_dict_or_text: Any) -> bool:
    if isinstance(opt_dict_or_text, dict):
        if opt_dict_or_text.get("is_null") or opt_dict_or_text.get("is_reference"):
            return True
        text = str(opt_dict_or_text.get("text", opt_dict_or_text.get("description", "")))
    else:
        text = str(opt_dict_or_text)
    return normalize_text(text) in AUDITED_NULL_PHRASES


def _format_tool_text(tool: Any) -> Tuple[str, str]:
    """Return ``(stable_tool_id, semantic_text)`` preserving tool name & description."""
    from .native import semantic_key
    if isinstance(tool, str):
        try:
            parsed = json.loads(tool)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return _format_tool_text(parsed)
        return "tool:" + hashlib.sha256(semantic_key(tool).encode()).hexdigest(), tool.strip()
    if isinstance(tool, dict):
        # Retain all semantic fields, including return types, endpoints and schemas.
        # Dataset/display IDs alone are metadata, not part of a tool definition.
        semantic = {k:v for k,v in tool.items() if k not in {"id", "tool_id", "cid", "alias"}}
        if not semantic:
            raise ValueError("Tool has no semantic definition")
        sem = json.dumps(semantic,sort_keys=True,ensure_ascii=False,separators=(",", ":"))
        cid = hashlib.sha256(semantic_key(sem).encode()).hexdigest()
        return "tool:"+cid, sem
    raise ValueError(f"Unsupported tool representation: {type(tool)}")


# ---------------------------------------------------------------------------
# Record mappers (produce standardized intermediate dicts before tokenization)
# ---------------------------------------------------------------------------


def map_openjev_record(
    record: Dict[str, Any], max_train_k: int = 32
) -> Iterator[Dict[str, Any]]:
    """Map one ``shenjunhao/OpenJevData-140k`` SFT record.

    Enforces AGENTS.md section 5.2:
    - Excludes RL split records if present.
    - Filters native ``K > max_train_k`` without cropping.
    - Preserves soft targets in task cross-entropy.
    - Maps ``noul`` by semantic polarity.
    - Sorts ``score`` bins by numeric ``score_values`` with lowest bin as reference.
    - Merges audited null-equivalent options into the virtual reference.
    """
    if str(record.get("split", record.get("subset", "SFT"))).upper() != "SFT":
        return
    if "task" in record and isinstance(record.get("answer"), dict):
        from .native import openjev
        mapped = openjev(record, max_train_k)
        if mapped is not None:
            yield mapped
        return
    split_tag = str(record.get("split", record.get("subset", "SFT"))).upper()
    if "RL" in split_tag and split_tag != "SFT":
        return

    rid = str(record.get("id") or record.get("uid") or record.get("question_id") or "")
    mode = str(record.get("mode") or record.get("task_type") or record.get("type") or "choice").lower()
    if mode in ("yes_no", "binary", "bool"):
        mode = "noul"
    elif mode in ("rating", "rubric", "likert", "ordinal"):
        mode = "score"
    elif mode not in ("choice", "noul", "score"):
        mode = "choice"

    query = str(
        record.get("query")
        or record.get("question")
        or record.get("instruction")
        or record.get("input")
        or ""
    ).strip()
    if not query:
        return

    domain = str(record.get("domain") or record.get("dataset") or record.get("source") or "general")
    template_id = str(record.get("template_id") or record.get("family_id") or "")

    if mode == "noul":
        prop = str(
            record.get("proposition")
            or record.get("hypothesis")
            or record.get("candidate")
            or (record.get("options", ["Yes"])[0] if record.get("options") else "Yes")
        ).strip()
        p_yes: Optional[float] = None
        if "soft_target" in record or "probs" in record or "target_dist" in record:
            dist = record.get("soft_target") or record.get("probs") or record.get("target_dist")
            if isinstance(dist, dict):
                pos_mass = sum(
                    float(v)
                    for k, v in dist.items()
                    if normalize_text(str(k)) in POSITIVE_POLARITY_TOKENS
                )
                neg_mass = sum(
                    float(v)
                    for k, v in dist.items()
                    if normalize_text(str(k)) in NEGATIVE_POLARITY_TOKENS
                )
                tot = pos_mass + neg_mass
                if tot > 0:
                    p_yes = pos_mass / tot
            elif isinstance(dist, (list, tuple)) and len(dist) == 2:
                # Convention [P(No), P(Yes)] or explicit label order
                labels = record.get("labels") or ["No", "Yes"]
                if normalize_text(str(labels[0])) in POSITIVE_POLARITY_TOKENS:
                    p_yes = float(dist[0])
                else:
                    p_yes = float(dist[1])
        if p_yes is None:
            raw_lbl = record.get("label", record.get("answer", record.get("target")))
            if isinstance(raw_lbl, (int, float)) and not isinstance(raw_lbl, bool):
                if 0.0 <= float(raw_lbl) <= 1.0:
                    p_yes = float(raw_lbl)
            else:
                norm_lbl = normalize_text(str(raw_lbl))
                if norm_lbl in POSITIVE_POLARITY_TOKENS:
                    p_yes = 1.0
                elif norm_lbl in NEGATIVE_POLARITY_TOKENS:
                    p_yes = 0.0
        if p_yes is None:
            return
        yield {
            "rid": rid or f"openjev_noul:{normalize_text(query)[:48]}",
            "bucket": "openjev_noul",
            "mode": "noul",
            "domain": domain,
            "template_id": template_id,
            "query": query,
            "proposition": prop,
            "p_yes": float(p_yes),
            "is_soft": not (p_yes in (0.0, 1.0)),
        }
        return

    if mode == "score":
        rubric = str(record.get("rubric") or record.get("criteria") or "Score the response.").strip()
        raw_bins = record.get("bins") or record.get("options") or record.get("scale") or []
        score_vals = record.get("score_values") or record.get("values")
        bins: List[Tuple[float, str]] = []
        if raw_bins and isinstance(raw_bins[0], dict):
            for b in raw_bins:
                v = float(b.get("value", b.get("score")))
                d = str(b.get("description") or b.get("text") or f"Score {v:g}")
                bins.append((v, d))
        elif score_vals is not None and raw_bins:
            for v, d in zip(score_vals, raw_bins):
                bins.append((float(v), str(d)))
        elif score_vals is not None:
            for v in score_vals:
                fv = float(v)
                bins.append((fv, f"Score {fv:g}"))
        if len(bins) < 2 or len(bins) - 1 > max_train_k:
            return
        bins_sorted = sorted(bins, key=lambda x: x[0])
        vals_sorted = [b[0] for b in bins_sorted]
        target_map: Dict[float, float] = {v: 0.0 for v in vals_sorted}
        dist = record.get("soft_target") or record.get("probs") or record.get("target_dist")
        is_soft = False
        if isinstance(dist, dict):
            is_soft = True
            for k, p in dist.items():
                target_map[float(k)] = float(p)
        elif isinstance(dist, (list, tuple)) and len(dist) == len(bins):
            is_soft = True
            for (v, _), p in zip(bins, dist):
                target_map[float(v)] = float(p)
        else:
            ans = record.get("label", record.get("score", record.get("answer")))
            if ans is None:
                return
            target_map[float(ans)] = 1.0
        tot = sum(target_map.values())
        if tot <= 0:
            return
        target_map = {k: v / tot for k, v in target_map.items()}
        yield {
            "rid": rid or f"openjev_score:{normalize_text(query)[:48]}",
            "bucket": "openjev_score",
            "mode": "score",
            "domain": domain,
            "template_id": template_id,
            "rubric": rubric,
            "query": query,
            "bins": bins_sorted,
            "target": target_map,
            "is_soft": is_soft,
        }
        return

    # mode == "choice"
    raw_opts = record.get("options") or record.get("candidates") or record.get("choices") or []
    if not raw_opts or len(raw_opts) > max_train_k:
        # Filter native K > 32; never crop candidates
        return

    options: List[Tuple[str, str]] = []
    null_cids: List[str] = []
    for idx, opt in enumerate(raw_opts):
        if isinstance(opt, dict):
            cid = str(opt.get("id") or opt.get("cid") or f"opt_{idx}")
            text = str(opt.get("text") or opt.get("description") or opt.get("content") or "").strip()
        else:
            cid = f"opt_{idx}"
            text = str(opt).strip()
        if not text:
            return
        options.append((cid, text))
        if _is_audited_null_option(opt):
            null_cids.append(cid)

    target_dist: Dict[str, float] = {cid: 0.0 for cid, _ in options}
    target_dist[REFERENCE_ID] = 0.0
    dist = record.get("soft_target") or record.get("probs") or record.get("target_dist")
    is_soft = False
    if isinstance(dist, dict):
        is_soft = True
        for k, v in dist.items():
            if k in (REFERENCE_ID, "null", "none", "abstain"):
                target_dist[REFERENCE_ID] += float(v)
            elif k in target_dist:
                target_dist[k] += float(v)
            elif str(k).isdigit() and int(k) < len(options):
                target_dist[options[int(k)][0]] += float(v)
    elif isinstance(dist, (list, tuple)) and len(dist) in (len(options), len(options) + 1):
        is_soft = True
        if len(dist) == len(options) + 1:
            target_dist[REFERENCE_ID] = float(dist[0])
            for (cid, _), v in zip(options, dist[1:]):
                target_dist[cid] = float(v)
        else:
            for (cid, _), v in zip(options, dist):
                target_dist[cid] = float(v)
    else:
        ans = record.get("answer", record.get("label", record.get("target")))
        if ans is None or ans in (REFERENCE_ID, "null", "none", -1):
            target_dist[REFERENCE_ID] = 1.0
        elif isinstance(ans, int) and 0 <= ans < len(options):
            target_dist[options[ans][0]] = 1.0
        elif str(ans) in target_dist:
            target_dist[str(ans)] = 1.0
        else:
            # Match by option text or letter index ("A", "B", ...)
            ans_str = str(ans).strip()
            if len(ans_str) == 1 and "A" <= ans_str.upper() <= "Z":
                idx = ord(ans_str.upper()) - ord("A")
                if 0 <= idx < len(options):
                    target_dist[options[idx][0]] = 1.0
            else:
                for cid, txt in options:
                    if normalize_text(txt) == normalize_text(ans_str):
                        target_dist[cid] = 1.0
                        break

    # Merge audited null-equivalent options into REFERENCE_ID and remove duplicate slots
    if null_cids:
        for nc in null_cids:
            target_dist[REFERENCE_ID] += target_dist.pop(nc, 0.0)
        options = [(cid, txt) for cid, txt in options if cid not in set(null_cids)]
        if not options:
            return

    tot = sum(target_dist.values())
    if tot <= 0:
        return
    target_dist = {k: v / tot for k, v in target_dist.items()}
    max_p = max(target_dist.values())
    if max_p < 1.0 - 1e-6:
        is_soft = True

    from .native import semantic_id
    identities = {cid:semantic_id(text) for cid,text in options}
    if len(set(identities.values())) != len(identities):
        raise ValueError("duplicate semantic alternatives")
    options = [(identities[cid],text) for cid,text in options]
    target_dist = {identities.get(cid,cid):p for cid,p in target_dist.items()}

    yield {
        "rid": rid or f"openjev_choice:{normalize_text(query)[:48]}",
        "bucket": "openjev_choice",
        "mode": "choice",
        "domain": domain,
        "template_id": template_id,
        "query": query,
        "options": options,
        "target": target_dist,
        "is_soft": is_soft,
        "null_certified": bool(record.get("null_certified", False)),
    }


def map_toolret_record(record: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Map one ``mangopy/ToolRet-Training-20w`` record.

    Enforces AGENTS.md section 5.2:
    - Uses the original ``query``, NEVER the generated ``prompt``.
    - Deduplicates tools by normalized ID and semantic description.
    - Retains ONLY single-positive records after deduplication; never turns
      other positives into negatives.
    - Keeps the supplied negatives (e.g. 15) as a retrieval pool for derived
      menu sampling.
    """
    from .native import semantic_key
    query = str(record.get("query") or "").strip()
    if not query:
        return
    rid = str(record.get("id") or record.get("qid") or record.get("query_id") or "")

    raw_pos = (
        record.get("positive_tools")
        or record.get("positives")
        or record.get("pos")
        or (record["positive"] if isinstance(record.get("positive"), list) else [record["positive"]] if "positive" in record else [])
        or ([record["gold_tool"]] if "gold_tool" in record else [])
    )
    raw_neg = (
        record.get("negative")
        or record.get("negative_tools")
        or record.get("negatives")
        or record.get("neg")
        or record.get("candidates")
        or []
    )

    pos_map: Dict[str, Tuple[str, str]] = {}
    for t in raw_pos:
        cid, sem = _format_tool_text(t)
        key = semantic_key(sem)
        pos_map.setdefault(key, (cid, sem))

    # Keep strictly single-positive records after tool deduplication
    if len(pos_map) != 1:
        return
    pos_key, (pos_cid, pos_sem) = next(iter(pos_map.items()))

    neg_list: List[Tuple[str, str]] = []
    seen_keys: Set[str] = {pos_key}
    seen_cids: Set[str] = {pos_cid}
    for t in raw_neg:
        cid, sem = _format_tool_text(t)
        key = semantic_key(sem)
        if key in seen_keys or cid in seen_cids:
            continue
        seen_keys.add(key)
        seen_cids.add(cid)
        neg_list.append((cid, sem))

    if not neg_list:
        return

    domain = str(record.get("domain") or record.get("category") or "toolret")
    yield {
        "rid": "toolret:" + (rid or hashlib.sha256(query.encode()).hexdigest()),
        "bucket": "toolret_choice",
        "mode": "choice",
        "domain": domain,
        "template_id": str(record.get("template_id") or ""),
        "query": query,
        "positive": (pos_cid, pos_sem),
        "negatives": neg_list,
        "options": [(pos_cid, pos_sem)] + neg_list,
        "target": {pos_cid: 1.0, REFERENCE_ID: 0.0},
        "is_soft": False,
        "is_retrieval_pool": True,
        "null_certified": bool(record.get("null_certified", False)),
    }


def map_arc_record(record: Dict[str, Any], max_train_k: int = 32) -> Iterator[Dict[str, Any]]:
    """Fallback choice mapper for ``allenai/ai2_arc`` (used ONLY if bucket insufficient)."""
    q = str(record.get("question") or "").strip()
    choices = record.get("choices") or {}
    texts = list(choices.get("text") or [])
    labels = list(choices.get("label") or [])
    ans_key = str(record.get("answerKey") or "").strip()
    if not q or not texts or len(texts) > max_train_k or ans_key not in labels:
        return
    gold_idx = labels.index(ans_key)
    from .native import semantic_id
    opts = [(semantic_id(str(t).strip()), str(t).strip()) for t in texts]
    target = {cid: (1.0 if i == gold_idx else 0.0) for i, (cid, _) in enumerate(opts)}
    target[REFERENCE_ID] = 0.0
    yield {
        "rid": f"arc:{record.get('id', normalize_text(q)[:48])}",
        "bucket": "openjev_choice",
        "mode": "choice",
        "domain": "ai2_arc",
        "template_id": "",
        "query": q,
        "options": opts,
        "target": target,
        "is_soft": False,
        "null_certified": bool(record.get("null_certified", False)),
        "is_fallback": True,
    }


def map_boolq_record(record: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Fallback noul mapper for ``google/boolq`` (used ONLY if bucket insufficient)."""
    passage = str(record.get("passage") or "").strip()
    question = str(record.get("question") or "").strip()
    ans = record.get("answer")
    if not question or ans is None:
        return
    query = f"Passage: {passage}\nQuestion: {question}" if passage else question
    yield {
        "rid": "boolq:" + hashlib.sha256(query.encode()).hexdigest(),
        "bucket": "openjev_noul",
        "mode": "noul",
        "domain": "boolq",
        "template_id": "",
        "query": query,
        "proposition": question,
        "p_yes": 1.0 if bool(ans) else 0.0,
        "is_soft": False,
        "is_fallback": True,
    }


def map_helpsteer2_record(record: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Fallback score mapper for ``nvidia/HelpSteer2`` helpfulness 0-4 rubric."""
    prompt = str(record.get("prompt") or "").strip()
    response = str(record.get("response") or "").strip()
    helpfulness = record.get("helpfulness")
    if not prompt or not response or helpfulness is None:
        return
    val = float(int(round(float(helpfulness))))
    if not (0.0 <= val <= 4.0):
        return
    target = {float(m): (1.0 if float(m) == val else 0.0) for m in range(5)}
    yield {
        "rid": "helpsteer2:" + hashlib.sha256((prompt+'\n'+response).encode()).hexdigest(),
        "bucket": "openjev_score",
        "mode": "score",
        "domain": "helpsteer2",
        "template_id": "",
        "rubric": HELPSTEER2_RUBRIC,
        "query": f"Prompt: {prompt}\nResponse: {response}",
        "bins": list(HELPSTEER2_BINS),
        "target": target,
        "is_soft": False,
        "is_fallback": True,
    }


def map_jevbench_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map one ``Leanmcp/jevbench`` benchmark case (Exp 1 & Exp 2).

    Preserves dataset name, canonical family ID (grouping reordered twins),
    display aliases, and maps ``sst5`` to ``score`` mode.
    """
    if isinstance(record.get("question"), dict):
        from .native import jevbench
        return jevbench(record)
    dataset = str(record.get("dataset") or record.get("task") or "jevbench").lower()
    case_id = str(record.get("id") or record.get("case_id") or "")
    family_id = str(
        record.get("family_id")
        or record.get("canonical_id")
        or record.get("group_id")
        or case_id.split("__")[0]
        or normalize_text(str(record.get("query") or record.get("question") or ""))[:64]
    )
    mode = str(record.get("mode") or record.get("type") or "").lower()
    if "sst5" in dataset or mode == "score":
        mode = "score"
    elif mode in ("noul", "binary", "yes_no"):
        mode = "noul"
    else:
        mode = "choice"

    query = str(record.get("query") or record.get("question") or record.get("input") or "").strip()
    if not query:
        return None

    if mode == "noul":
        prop = str(record.get("proposition") or record.get("hypothesis") or "Yes").strip()
        lbl = record.get("label", record.get("answer"))
        p_yes = None
        if lbl is not None:
            p_yes = 1.0 if normalize_text(str(lbl)) in POSITIVE_POLARITY_TOKENS else 0.0
        return {
            "case_id": case_id or f"{dataset}:{family_id}",
            "family_id": family_id,
            "dataset": dataset,
            "mode": "noul",
            "query": query,
            "proposition": prop,
            "p_yes": p_yes,
        }

    if mode == "score":
        rubric = str(
            record.get("rubric")
            or "Classify the sentiment on a 5-point scale from 0 (very negative, reference bin) to 4 (very positive)."
        )
        raw_bins = record.get("bins") or [
            (0.0, "0: very negative"),
            (1.0, "1: negative"),
            (2.0, "2: neutral"),
            (3.0, "3: positive"),
            (4.0, "4: very positive"),
        ]
        bins = sorted([(float(v), str(d)) for v, d in raw_bins], key=lambda x: x[0])
        lbl = record.get("label", record.get("score", record.get("answer")))
        target = None
        if lbl is not None:
            target = {v: (1.0 if v == float(lbl) else 0.0) for v, _ in bins}
        return {
            "case_id": case_id or f"{dataset}:{family_id}",
            "family_id": family_id,
            "dataset": dataset,
            "mode": "score",
            "rubric": rubric,
            "query": query,
            "bins": bins,
            "target": target,
        }

    raw_opts = record.get("options") or record.get("candidates") or record.get("choices") or []
    options: List[Tuple[str, str]] = []
    aliases: List[str] = []
    for idx, opt in enumerate(raw_opts):
        if isinstance(opt, dict):
            cid = str(opt.get("id") or opt.get("cid") or f"c_{idx}")
            txt = str(opt.get("text") or opt.get("description") or "").strip()
            alias = str(opt.get("alias") or opt.get("label") or chr(ord("A") + (idx % 26)))
        else:
            cid = f"c_{idx}"
            txt = str(opt).strip()
            alias = chr(ord("A") + (idx % 26))
        options.append((cid, txt))
        aliases.append(alias)
    gold = record.get("answer", record.get("label", record.get("gold_id")))
    gold_id: Optional[str] = None
    if gold is not None:
        if isinstance(gold, int) and 0 <= gold < len(options):
            gold_id = options[gold][0]
        elif str(gold) in {cid for cid, _ in options}:
            gold_id = str(gold)
        elif str(gold) in aliases:
            gold_id = options[aliases.index(str(gold))][0]
    forced = bool(record.get("forced_choice", True))
    return {
        "case_id": case_id or f"{dataset}:{family_id}",
        "family_id": family_id,
        "dataset": dataset,
        "mode": "choice",
        "query": query,
        "options": options,
        "aliases": aliases,
        "gold_id": gold_id,
        "forced_choice": forced,
    }


# ---------------------------------------------------------------------------
# End-to-end record -> DecisionExample materialization & manifest preparation
# ---------------------------------------------------------------------------


def record_dedup_keys(rec: Dict[str, Any]) -> List[str]:
    """Stable keys used for cross-source deduplication and overlap auditing."""
    keys = [f"q:{normalize_text(rec['query'])}"]
    if rec.get("state"):
        keys.append(f"state:{normalize_text(rec['state'])}")
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    for text in strings(rec.get("native_state")):
        if len(text) > 20:
            keys.append(f"state:{normalize_text(text)}")
    if rec.get("audit_group"):
        keys.append("near:" + rec["audit_group"])
    if rec.get("template_id"):
        keys.append(f"tpl:{rec['template_id']}")
    if rec.get("family_id"):
        keys.append(f"fam:{rec['family_id']}")
    return keys


def materialize_example(
    tokenizer,
    rec: Dict[str, Any],
    negative_bank: Optional[Sequence[Tuple[str, str]]] = None,
    rng: Optional[random.Random] = None,
    enable_augmentation: bool = True,
    k_proposal: Optional[Dict[int, float]] = None,
    null_rate: float = 0.15,
    max_k: int = 32,
) -> Tuple[DecisionExample, Dict[str, Any]]:
    """Convert one mapped record into a ``DecisionExample`` (with K/null augmentation)."""
    from .native import semantic_key
    mode = rec["mode"]
    if mode == "noul":
        ex = build_noul(tokenizer, rec["query"], rec["proposition"], rec.get("p_yes"))
        ex.meta.update({"rid": rec.get("rid"), "bucket": rec.get("bucket"), "domain": rec.get("domain")})
        return ex, {"realized_k": 1, "null_augmented": False}

    if mode == "score":
        ex = build_score(tokenizer, rec["rubric"], rec["query"], rec["bins"], rec.get("target"))
        ex.meta.update({"rid": rec.get("rid"), "bucket": rec.get("bucket"), "domain": rec.get("domain")})
        return ex, {"realized_k": len(ex.candidates), "null_augmented": False}

    # choice mode
    rng = rng or random.Random(0)
    is_soft = bool(rec.get("is_soft", False))
    if not enable_augmentation or is_soft:
        opts = list(rec["options"])
        ex = build_choice(
            tokenizer,
            rec["query"],
            opts,
            rec.get("target"),
            forced_choice=bool(rec.get("forced_choice", False)),
            aliases=rec.get("aliases"),
        )
        ex.meta.update({"rid": rec.get("rid"), "bucket": rec.get("bucket"), "domain": rec.get("domain")})
        return ex, {"realized_k": len(opts), "null_augmented": False}

    # Eligible hard single-positive choice record
    text_by_cid: Dict[str, str] = {}
    if rec.get("is_retrieval_pool"):
        pos_cid, pos_txt = rec["positive"]
        text_by_cid[pos_cid] = pos_txt
        for cid, txt in rec["negatives"]:
            text_by_cid[cid] = txt
        menu_ids = [pos_cid]
        pool_ids = [cid for cid, _ in rec["negatives"]]
    else:
        for cid, txt in rec["options"]:
            text_by_cid[cid] = txt
        menu_ids = [cid for cid, _ in rec["options"]]
        pos_cids = [cid for cid, p in rec["target"].items() if cid != REFERENCE_ID and p > 0.5]
        if len(pos_cids) != 1:
            ex = build_choice(tokenizer, rec["query"], rec["options"], rec["target"])
            ex.meta.update({"rid": rec.get("rid"), "bucket": rec.get("bucket")})
            return ex, {"realized_k": len(rec["options"]), "null_augmented": False}
        pos_cid = pos_cids[0]
        pool_ids = []

    bank_ids: List[str] = list(pool_ids)
    eligible = set(rec.get("eligible_negative_ids", []))
    known_positive = set(rec.get("known_positive_ids", [pos_cid]))
    seen_text = {semantic_key(t): c for c, t in text_by_cid.items()}
    bank_candidates = negative_bank or []
    if rec.get("is_retrieval_pool") and len(bank_candidates) > max_k * 4:
        bank_candidates = rng.sample(bank_candidates, max_k * 4)
    for cid, txt in bank_candidates:
        if cid in known_positive or (not rec.get("is_retrieval_pool") and cid not in eligible):
            continue
        if semantic_key(txt) in seen_text and seen_text[semantic_key(txt)] != cid:
            continue
        if cid in text_by_cid and semantic_key(text_by_cid[cid]) != semantic_key(txt):
            raise ValueError("candidate ID collision in bank")
        seen_text[semantic_key(txt)] = cid
        if cid not in text_by_cid:
            text_by_cid[cid] = txt
        bank_ids.append(cid)

    proposal = k_proposal or DEFAULT_K_PROPOSAL
    null_certified = bool(rec.get("null_certified", False))
    if rec.get("is_retrieval_pool"):
        from .augmentation import sample_target_k
        proposed = sample_target_k(1, rng, proposal)
        supplied = list(dict.fromkeys(pool_ids))
        rng.shuffle(supplied)
        extra = [cid for cid in dict.fromkeys(bank_ids) if cid not in set(supplied) and cid != pos_cid]
        rng.shuffle(extra)
        menu_ids = [pos_cid] + (supplied + extra)[:proposed-1]
        proposal = {len(menu_ids): 1.0}
        # Supplied negatives are annotated; extra bank negatives are unjudged.
        null_certified = all(cid in set(pool_ids) | eligible for cid in menu_ids if cid != pos_cid)
        bank_ids = []
    aug = augment_menu(
        menu_ids=menu_ids,
        positive_id=pos_cid,
        negative_bank=bank_ids,
        rng=rng,
        proposal=proposal,
        null_rate=null_rate,
        null_certified=null_certified,
        max_k=max_k,
    )
    opts = [(cid, text_by_cid[cid]) for cid in aug.candidate_ids]
    target = {cid: 0.0 for cid in aug.candidate_ids}
    if aug.target_id is None:
        target[REFERENCE_ID] = 1.0
    else:
        target[aug.target_id] = 1.0
        target[REFERENCE_ID] = 0.0
    ex = build_choice(tokenizer, rec["query"], opts, target)
    ex.meta.update({
        "rid": rec.get("rid"),
        "bucket": rec.get("bucket"),
        "domain": rec.get("domain"),
        "null_augmented": aug.null_augmented,
    })
    return ex, {
        "native_k": aug.native_k,
        "proposed_k": aug.proposed_k,
        "realized_k": aug.realized_k,
        "null_augmented": aug.null_augmented,
    }


@dataclass
class PreparedDatasetManifest:
    splits: Dict[str, List[str]]
    bucket_counts: Dict[str, Dict[str, int]]
    trace_summary: Dict[str, Dict[str, float]]
    candidate_k_histogram: Dict[int, int]
    removed_counts: Dict[str, int] = field(default_factory=dict)
    fallbacks_used: List[str] = field(default_factory=list)


def prepare_training_manifest(
    records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]] = (),
    fallback_records: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    num_updates: int = 3125,
    per_bucket_per_batch: int = 8,
    seed: int = 0,
) -> Tuple[PreparedDatasetManifest, List[List[Dict[str, str]]], Dict[str, List[Tuple[str, str]]]]:
    """Audit overlap, split 90/5/5, construct training-only banks, and build the trace."""
    eval_keys: Set[str] = set()
    for er in eval_records:
        for k in record_dedup_keys(er):
            eval_keys.add(k)

    all_recs = list(records)
    fallbacks_used = []
    initial = split_records(all_recs, record_dedup_keys, eval_keys=eval_keys)
    initial_by_id = {r["rid"]: r for r in all_recs}
    present = {initial_by_id[r]["bucket"] for r in initial["train"]}
    for bucket, rows in (fallback_records or {}).items():
        if bucket not in present:
            all_recs.extend(rows)
            fallbacks_used.append(bucket)
    rec_by_id = {r["rid"]: r for r in all_recs}
    if len(rec_by_id) != len(all_recs):
        raise ValueError("duplicate source record IDs; deduplicate before manifest construction")
    splits = split_records(all_recs, record_dedup_keys, eval_keys=eval_keys)
    for partition in ("train", "dev", "calib"):
        uncertain = [rid for rid in splits[partition] if rec_by_id[rid].get("audit_uncertain")]
        splits[partition] = [rid for rid in splits[partition] if rid not in set(uncertain)]
        splits["quarantine"].extend(uncertain)
    train_by_bucket = {b: [] for b in BUCKETS}
    for rid in splits["train"]:
        train_by_bucket[rec_by_id[rid]["bucket"]].append(rid)

    from .native import semantic_key
    # Construct versioned candidate negative banks strictly from the TRAINING partition
    banks_by_domain: Dict[str, List[Tuple[str, str]]] = {}
    seen_in_bank: Set[Tuple[str, str]] = set()
    for rid in splits["train"]:
        r = rec_by_id[rid]
        if r["mode"] != "choice":
            continue
        dom = r.get("domain", "general")
        opts = r.get("negatives") if r.get("is_retrieval_pool") else r.get("options", [])
        for cid, txt in opts:
            key = (dom, semantic_key(txt))
            if key not in seen_in_bank:
                seen_in_bank.add(key)
                banks_by_domain.setdefault(dom, []).append((cid, txt))

    trace = build_trace(
        train_by_bucket, num_updates=num_updates, per_bucket_per_batch=per_bucket_per_batch, seed=seed
    )
    t_stats = trace_stats(trace)

    # Freeze actual menus/targets once, before tokenization by any model.
    class TextOnly:
        def encode(self, text, **kw):
            return [0]  # never used as model input
    rng = random.Random(seed + 10000)
    k_hist = Counter()
    for batch in trace:
        for item in batch:
            rec = rec_by_id[item["rid"]]
            ex, aug = materialize_example(TextOnly(), rec,
                banks_by_domain.get(rec.get("domain", "general"), []), rng=rng)
            item["augmentation"] = aug
            if rec["mode"] == "choice":
                item["options"] = [(c.cid, c.text) for c in ex.candidates]
                item["target"] = {REFERENCE_ID: float(ex.target[0]),
                    **{c.cid: float(p) for c, p in zip(ex.candidates, ex.target[1:])}}
            k_hist[len(ex.candidates)] += 1

    bucket_counts: Dict[str, Dict[str, int]] = {}
    for s in ("train", "dev", "calib", "quarantine"):
        c = Counter(rec_by_id[rid]["bucket"] for rid in splits[s])
        bucket_counts[s] = {b: int(c.get(b, 0)) for b in BUCKETS}

    manifest = PreparedDatasetManifest(
        splits=splits,
        bucket_counts=bucket_counts,
        trace_summary=t_stats,
        candidate_k_histogram=dict(sorted(k_hist.items())),
        removed_counts={"quarantined_overlap": len(splits["quarantine"])},
        fallbacks_used=fallbacks_used,
    )
    return manifest, trace, banks_by_domain
