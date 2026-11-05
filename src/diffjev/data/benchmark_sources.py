"""Explicit downloads of only the pinned sources needed for withheld JevBench states."""
import hashlib
import json
from pathlib import Path
import re


def missing_source_plan(common, cases, source_root):
    """Resolve source files from the benchmark's own revision-pinned manifest."""
    root = Path(source_root).resolve()
    plan = {}
    for case in cases:
        if case.get("state") is not None and case.get("state") != "":
            continue
        key = common.parent_key(case["slice"])
        src = common.source_by_key(key)
        revision = src["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}",revision):
            raise ValueError(f"Unpinned benchmark source revision: {key}")
        for field,expected in (("hf_repo",src["hf_repo"]),("revision",revision)):
            if case.get(field) is not None and case[field] != expected:
                raise ValueError(f"Case/source provenance mismatch for {key}: {field}")
        for filename in src["files"]:
            target = (root/key/filename).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise ValueError("Benchmark source path escapes source root")
            if not target.is_file():
                plan[(key,filename)] = {"key":key,"repo_id":src["hf_repo"],"revision":revision,
                                        "filename":filename,"target":str(target)}
    return list(plan.values())


def download_benchmark_sources(source_root=None):
    """User-invoked only: download exact upstream files, never train/dev pools."""
    from .. import workflows as w
    from ..evaluation.official import load_official
    from .sources import load_local_dataset
    from huggingface_hub import hf_hub_download
    common,_,_ = load_official()
    common.SOURCES_PATH = w.ROOT/"data/raw/jevbench/sources.json"
    root = Path(source_root or w.ROOT/"data/raw/jevbench_sources")
    cases = load_local_dataset("jevbench",str(w.ROOT/"data"))
    plan = missing_source_plan(common,cases,root)
    completed = []
    for item in plan:
        print(f"Downloading benchmark source {item['repo_id']}@{item['revision']}: {item['filename']}",flush=True)
        path = Path(hf_hub_download(repo_id=item["repo_id"],repo_type="dataset",
            revision=item["revision"],filename=item["filename"],local_dir=str(root/item["key"])))
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024*1024),b""):
                sha.update(block)
        record = {**item,"sha256":sha.hexdigest(),"bytes":path.stat().st_size}
        provenance = root/item["key"]/"source_downloads.json"
        prior = json.loads(provenance.read_text()) if provenance.exists() else {}
        prior[item["filename"]] = record
        provenance.write_text(json.dumps(prior,indent=2)+"\n",encoding="utf-8")
        completed.append(record)
    return {"downloaded_files":len(completed),"files":completed,
            "note":"Original source text stays local; restored states must still pass the benchmark hashes."}
