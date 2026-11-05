"""Original ToolRet Queries/Tools adapters, query-only single-positive routing."""
import json
from .sources import _format_tool_text
from .splits import normalize_text


def adapt_toolret_evaluation(query_rows, tool_rows):
    corpus, native_to_semantic = {}, {}
    for row in tool_rows:
        cid, text = _format_tool_text(row["documentation"])
        native_to_semantic[str(row["id"])] = cid
        entry=corpus.setdefault(cid, {"cid":cid,"text":text,"domain":row["_config"],"domains":[]})
        if row["_config"] not in entry["domains"]:
            entry["domains"].append(row["_config"])
    queries, removed = [], []
    positives_by_query = {}
    for row in query_rows:
        labels = json.loads(row["labels"]) if isinstance(row["labels"],str) else row["labels"]
        positives = set()
        for label in labels:
            if float(label.get("relevance",0)) <= 0:
                continue
            if str(label["id"]) not in native_to_semantic:
                raise ValueError("ToolRet positive ID missing from tool corpus")
            positives.add(native_to_semantic[str(label["id"])])
        key = normalize_text(row["query"])
        positives_by_query.setdefault(key,set()).update(positives)
    seen = set()
    for row in query_rows:
        key = normalize_text(row["query"])
        positives = positives_by_query[key]
        if len(positives) != 1 or key in seen:
            removed.append(row["id"])
            continue
        seen.add(key)
        positive = next(iter(positives))
        queries.append({"qid": str(row["id"]), "query": row["query"],
            "positive_id": positive, "known_positive_ids": sorted(positives),
            "domain": str(row["category"]), "source_config": row["_config"]})
    return queries, list(corpus.values()), {"excluded_multi_positive_or_duplicate": removed}
