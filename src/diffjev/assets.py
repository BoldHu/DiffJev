"""Pinned local assets for formal runs. No implicit downloads."""
from pathlib import Path
import json
import platform
import importlib.metadata
import torch
from .config import load_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def model_specs():
    cfg = load_yaml_config(str(ROOT / "configs/models.yaml"))
    return {k:v for group in ("anchor","size_panel","baselines") for k,v in cfg[group].items()}


def verify_snapshot(path, repo_id):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing local snapshot: {path}. See README manual asset paths.")
    lock = json.loads((ROOT / "configs/revisions.json").read_text())
    file = path / "download_provenance.json"
    if not file.exists():
        raise ValueError(f"{file} is required, including for manually uploaded snapshots")
    actual = json.loads(file.read_text())
    if actual.get("repo_id") != repo_id or actual.get("resolved_sha") != lock[repo_id]["revision"]:
        raise ValueError(f"Snapshot revision mismatch: {repo_id}")
    return {"repo_id":repo_id, "revision":actual["resolved_sha"]}


def environment_record():
    packages = {}
    for name in ("torch","transformers","datasets","huggingface_hub","accelerate","numpy","jeva"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python":platform.python_version(), "packages":packages,
            "installed_packages":{dist.metadata["Name"]:dist.version for dist in importlib.metadata.distributions()},
            "cuda":torch.version.cuda, "devices":[{"name":torch.cuda.get_device_name(i),
            "total_memory":torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())], "attention":"torch SDPA, explicit DiffJev visibility",
            "compilation":False}


def load_backbone(key, device="cuda", dtype=torch.bfloat16, verify=True):
    from .backbones.llada import LLaDAAdapter
    from .backbones.nemotron import NemotronDiffusionAdapter
    from .backbones.causal import QwenCausalAdapter
    spec = model_specs()[key]
    path = ROOT / spec["local_dir"]
    provenance = verify_snapshot(path, spec["repo_id"])
    cls, kwargs = (LLaDAAdapter, {}) if key.startswith("llada") else (
        (NemotronDiffusionAdapter, {"size":key.split("_")[1].upper()}) if key.startswith("nemotron")
        else (QwenCausalAdapter, {}))
    adapter = cls.from_local(str(path), torch_dtype=dtype, device_map=None, **kwargs).to(device)
    adapter.provenance = provenance
    cfg = adapter.hf_model.config
    adapter.max_context_tokens = int(getattr(cfg,"max_position_embeddings",getattr(cfg,"max_sequence_length",4096)))
    if verify:
        adapter.verification = {"invariants":adapter.verify_adapter(), "native_parity":adapter.verify_native_parity()}
    return adapter


def read_parquet_snapshot(name, split=None, with_config=False):
    """Read explicit split files; never infer JevBench as an imagefolder dataset."""
    import pyarrow.parquet as pq
    path = ROOT / "data/raw" / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*.parquet"))
    if split:
        files = [p for p in files if p.name.lower().startswith(split.lower())]
    if not files:
        file = path / ((split or "data")+".jsonl")
        if file.exists():
            return [json.loads(line) for line in file.read_text(encoding="utf-8").split("\n") if line.strip()]
        raise FileNotFoundError(f"No local {split} parquet/JSONL files in {path}")
    rows = []
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=2048):
            chunk = batch.to_pylist()
            if with_config:
                for row in chunk:
                    row["_config"] = file.parent.name
            rows.extend(chunk)
    return rows
