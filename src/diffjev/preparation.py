"""Reusable preparation checks for notebook 00; no training or implicit downloads."""
from pathlib import Path
import importlib
import importlib.metadata
import json
import re
import sys

CORE_MODELS = ("llada_8b_base", "nemotron_3b", "nemotron_8b", "nemotron_14b")
CORE_DATASETS = ("openjev", "toolret", "jevbench", "toolret_queries", "toolret_tools")
PREPARED_FILES = ("records", "benchmark", "toolret_queries", "toolret_corpus", "trace", "banks", "manifest", "audit")


def check_environment(root):
    root = Path(root)
    problems = []
    if sys.version_info[:2] != (3,12):
        problems.append("Formal worker must use Python 3.12; rerun notebook 00 environment installation")
    for line in (root/"requirements-lock.txt").read_text().splitlines():
        match = re.fullmatch(r"([\w-]+)==([^\s]+)",line.strip())
        if not match:
            continue
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+")[0] != expected:
            problems.append(f"{name}: expected {expected}, found {actual}")
    expected_jeva = json.loads((root/"configs/revisions.json").read_text())["LiQiiiii/Jeva"]["revision"]
    try:
        direct = json.loads(importlib.metadata.distribution("jeva-model").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info",{}).get("commit_id") != expected_jeva:
            problems.append("Jeva runtime is not installed from the pinned Git commit")
    except importlib.metadata.PackageNotFoundError:
        problems.append("Jeva runtime is missing")
    if problems:
        raise RuntimeError("Environment mismatch:\n"+"\n".join(problems))
    # Import checks expose binary/API incompatibilities before model downloads.
    for name in ("torch", "torchvision", "torchaudio", "transformers", "peft", "datasets", "pyarrow", "jeva"):
        importlib.import_module(name)
    from .assets import environment_record
    from .workflows import save_json
    report = environment_record()
    save_json(root/"outputs/setup/environment.json",report)
    print({"python":report["python"],"packages":report["packages"],"devices":report["devices"]})
    print("Pinned environment imports passed. No GPU is required for preparation.")
    return report


def validate_prepared(directory, seed=0, updates=3125):
    """Refuse silent reuse of another seed/budget or an altered frozen trace."""
    from .data.native import digest
    directory = Path(directory)
    missing = [name for name in PREPARED_FILES if not (directory/(name+".json")).is_file()]
    if missing:
        raise ValueError("Incomplete prepared data: "+", ".join(missing))
    read = lambda name: json.loads((directory/(name+".json")).read_text(encoding="utf-8"))
    audit,trace = read("audit"),read("trace")
    if audit["seed"] != seed or len(trace) != updates:
        raise ValueError("Existing preparation uses a different seed/update budget; use a separate project copy")
    for name,key in (("trace","trace_sha256"),("banks","bank_sha256"),("manifest","manifest_sha256")):
        if digest(trace if name=="trace" else read(name)) != audit[key]:
            raise ValueError("Prepared "+name+" checksum mismatch")
    return {"seed":seed,"optimizer_updates":updates,"trace_sha256":audit["trace_sha256"]}


def prepare_all(seed=0, scaling_queries=512, restore_states=False, source_root=None):
    from . import workflows as w
    prepared = w.PREPARED
    if any((prepared/(name+".json")).exists() for name in PREPARED_FILES):
        result = validate_prepared(prepared,seed)
        print("Reusing verified frozen preparation:",result)
    else:
        if restore_states:
            print(w.restore_benchmark_states(source_root))
        print("Preparing accepted source records and frozen augmentation trace...",flush=True)
        w.prepare_data(seed=seed)
        result = validate_prepared(prepared,seed)
    scaling = prepared/"scaling.json"
    if scaling.exists():
        manifest = w.read_json(scaling)
        if manifest["seed"] != seed or manifest["ks"] != [4,16,64,256,1024,2048]:
            raise ValueError("Scaling seed/K mismatch; preserve the old manifest and use a separate project copy")
        available = len(w.read_json(prepared/"toolret_queries.json"))
        if len(manifest["queries"]) != min(scaling_queries,available):
            raise ValueError("Scaling query count mismatch; do not overwrite frozen comparison pools")
        if w.digest(manifest["candidate_text"]) != manifest["corpus_sha256"]:
            raise ValueError("Scaling corpus checksum mismatch")
        print("Reusing fixed scaling pools:",len(manifest["queries"]),"queries")
    else:
        print(w.build_scaling_manifest(seed=seed,num_queries=scaling_queries))
    w.save_json(w.ROOT/"outputs/setup/preparation.json",result)
    print("Preparation complete. Review data/processed/formal/audit.json, then open notebook 02.")
    return result


def asset_status(full_models=(), seed=0):
    from .assets import ROOT, model_specs, verify_snapshot
    from .workflows import save_json
    specs = model_specs()
    rows = []
    for name in dict.fromkeys(list(CORE_MODELS)+list(full_models)):
        path = ROOT/specs[name]["local_dir"]
        try:
            verify_snapshot(path,specs[name]["repo_id"])
            metadata = (path/"config.json").exists() and any(path.glob("tokenizer*"))
            weights = any((path/file).exists() for file in ("model.safetensors","pytorch_model.bin"))
            for index in list(path.glob("*.safetensors.index.json"))+list(path.glob("pytorch_model.bin.index.json")):
                shards = set(json.loads(index.read_text())["weight_map"].values())
                weights = bool(shards) and all((path/shard).is_file() for shard in shards)
            if name == "jeva_4b":
                # Jeva has its own config/head/backbone layout, validated on load.
                metadata = (path/"config.yaml").exists()
                weights = metadata and (path/"head.pt").exists() and any(path.rglob("*.safetensors"))
            rows.append({"model":name,"metadata":bool(metadata),"weight_files_present":bool(weights),
                         "weights_requested":name in full_models})
        except (ValueError,FileNotFoundError):
            rows.append({"model":name,"metadata":False,"weight_files_present":False,
                         "weights_requested":name in full_models})
    try:
        prepared = validate_prepared(ROOT/"data/processed/formal",seed)
        prepared["scaling_manifest_present"] = (ROOT/"data/processed/formal/scaling.json").is_file()
    except (ValueError,KeyError) as error:
        prepared = {"ready":False,"reason":str(error)}
    report = {"models":rows,"prepared":prepared,
              "note":"File/provenance checks only; native-weight parity and GPU feasibility are checked at model loading/use."}
    save_json(ROOT/"outputs/setup/readiness.json",report)
    print(json.dumps(report,indent=2))
    return report
