"""Bridge to the unmodified, revision-pinned original JevBench evaluator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_official():
    root = Path(__file__).resolve().parents[3] / "third_party" / "jevbench"
    provenance = json.loads((root / "provenance.json").read_text())
    modules = {}
    for name in ("common", "score_runs"):
        file = root / (name + ".py")
        if hashlib.sha256(file.read_bytes()).hexdigest() != provenance["files"][file.name]:
            raise ValueError("Official evaluator checksum mismatch")
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get("common")
        if name == "score_runs":
            sys.modules["common"] = modules["common"]
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is not None:
                sys.modules["common"] = previous
            else:
                sys.modules.pop("common", None)
        modules[name] = module
    return modules["common"], modules["score_runs"], provenance


def official_record(case, prediction):
    native = case.example.meta["native_case"]
    out = {k: native[k] for k in ("case_id", "family_id", "slice", "task_type", "gold", "perturbation")}
    out.update(pred=None, error=prediction.get("failure_reason") if prediction.get("failed") else None,
               latency_ms=prediction.get("api_latency_ms") or prediction.get("local_gpu_ms"),
               truncated_state=native.get("truncated", False))
    if prediction.get("failed"):
        return out
    probs = prediction.get("probs")
    if case.example.mode == "choice":
        mapping = case.example.meta["native_alias_by_id"]
        out["pred"] = mapping.get(prediction["top1"])
        if probs is not None:
            out["probabilities"] = {mapping[c]: p for c, p in probs.items() if c in mapping}
    elif case.example.mode == "noul":
        p = prediction.get("p_yes")
        if p is None and probs is not None:
            p = probs.get(case.example.candidates[0].cid)
        if p is None:
            out["parse_error"] = "noul probability unavailable"
        else:
            out.update(pred=p >= .5, p_true=p)
    else:
        value = prediction.get("expected_score")
        if value is None:
            out["parse_error"] = "score unavailable"
        else:
            out.update(pred=int(round(value)), expected_level=value)
            if probs is not None:
                ids = ["__reference__"] + [c.cid for c in case.example.candidates]
                out["probabilities"] = {str(i): probs[c] for i, c in enumerate(ids)}
    return out


def evaluate_official(cases, predictions, output_dir):
    common, scorer, provenance = load_official()
    root = Path(output_dir).resolve()
    run = root / "official"
    run.mkdir(parents=True, exist_ok=True)
    by_id = {c.case_id: c for c in cases}
    rows = [official_record(by_id[p["case_id"]], p) for p in predictions]
    (run / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    (run / "meta.json").write_text(json.dumps({"evaluator": provenance, "suite": "original_12_text"}), encoding="utf-8")
    common.RUNS, common.ROOT = root, root
    old_argv = sys.argv
    try:
        sys.argv = ["score_runs.py", "--run-id", "official"]
        scorer.main()
    finally:
        sys.argv = old_argv
    return json.loads((run / "scores.json").read_text(encoding="utf-8"))
