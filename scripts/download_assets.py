"""Download revision-pinned model weights and dataset snapshots (Colab / A100 server).

Never run this on the local RTX 3050 machine for full 3B/8B/14B backbones.
Records exact resolved commit SHAs into ``download_provenance.json`` inside each
target directory so every experiment is auditable.

Usage:
    python scripts/download_assets.py --list
    python scripts/download_assets.py --model llada_8b_base --yes
    python scripts/download_assets.py --model nemotron_3b --model nemotron_8b --yes
    python scripts/download_assets.py --dataset openjev --dataset toolret --dataset jevbench --yes
    python scripts/download_assets.py --all-core --yes
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffjev.config import load_yaml_config  # noqa: E402
from diffjev.utils import ensure_dirs, project_paths  # noqa: E402


def collect_models(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for group in ("anchor", "size_panel", "baselines"):
        for k, v in (cfg.get(group) or {}).items():
            if isinstance(v, dict):
                out[k] = {**v, "group": group}
    return out


def collect_datasets(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (cfg.get("training") or {}).items():
        out[k] = {**v, "group": "training"}
    for k, v in (cfg.get("fallbacks") or {}).items():
        out[f"fallback_{k}"] = {**v, "group": "fallbacks"}
    for k, v in (cfg.get("evaluation") or {}).items():
        if isinstance(v, dict) and v.get("repo_id"):
            out[k] = {**v, "group": "evaluation"}
    return out


def _write_provenance(target_dir: str, info: Dict[str, Any]) -> None:
    os.makedirs(target_dir, exist_ok=True)
    prov_path = os.path.join(target_dir, "download_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def download_model(name: str, spec: Dict[str, Any], models_root: str, rev_override: str = None,
                   metadata_only: bool = False) -> str:
    if spec.get("manual_upload_required") or not spec.get("repo_id"):
        raise RuntimeError(
            f"Model {name!r} requires manual upload to {spec.get('local_dir')} (see README.md)."
        )
    rev = rev_override or spec.get("revision")
    if not rev:
        raise RuntimeError(f"Model {name!r}: set `revision` in configs/models.yaml or pass --revision.")
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise ValueError("Model revision must be an immutable 40-character commit SHA")
    from huggingface_hub import HfApi, snapshot_download

    target = os.path.join(models_root, os.path.basename(spec["local_dir"]))
    ignore = ["*lora*", "*LoRA*"] if spec.get("skip_lora") else None
    api = HfApi()
    repo_info = api.model_info(spec["repo_id"], revision=rev)
    resolved_sha = getattr(repo_info, "sha", rev)
    # Fetch tokenizer, configuration and remote implementation without weights.
    patterns = ["*.json", "*.py", "*.model", "*.tiktoken", "*.jinja", "merges.txt", "vocab.txt"] if metadata_only else None
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=resolved_sha,
        local_dir=target,
        ignore_patterns=ignore,
        allow_patterns=patterns,
    )
    _write_provenance(
        target,
        {
            "name": name,
            "repo_id": spec["repo_id"],
            "requested_revision": rev,
            "resolved_sha": resolved_sha,
            "skip_lora": bool(spec.get("skip_lora", False)),
            "last_download_scope": "metadata_only" if metadata_only else "full_snapshot",
            "downloaded_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        },
    )
    return target


def download_dataset(name: str, spec: Dict[str, Any], raw_root: str, rev_override: str = None) -> str:
    rev = rev_override or spec.get("revision")
    if not rev:
        raise RuntimeError(f"Dataset {name!r}: set `revision` in configs/data.yaml or pass --revision.")
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise ValueError("Dataset revision must be an immutable 40-character commit SHA")
    from huggingface_hub import HfApi, snapshot_download

    target = os.path.join(raw_root, os.path.basename(spec.get("local_dir", name)))
    api = HfApi()
    ds_info = api.dataset_info(spec["repo_id"], revision=rev)
    resolved_sha = getattr(ds_info, "sha", rev)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=resolved_sha,
        local_dir=target,
    )
    _write_provenance(
        target,
        {
            "name": name,
            "repo_id": spec["repo_id"],
            "split": spec.get("split"),
            "requested_revision": rev,
            "resolved_sha": resolved_sha,
            "downloaded_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        },
    )
    return target


def main() -> None:
    paths = ensure_dirs(project_paths())
    mcfg = load_yaml_config(str(paths["configs"] / "models.yaml"))
    dcfg = load_yaml_config(str(paths["configs"] / "data.yaml"))
    models = collect_models(mcfg)
    datasets = collect_datasets(dcfg)

    ap = argparse.ArgumentParser(description="Download pinned models and datasets for DiffJev.")
    ap.add_argument("--list", action="store_true", help="List all configured models and datasets")
    ap.add_argument("--model", action="append", choices=sorted(models), help="Model key(s) to download")
    ap.add_argument("--dataset", action="append", choices=sorted(datasets), help="Dataset key(s) to download")
    ap.add_argument("--all-core", action="store_true", help="Download anchor + size panel + core datasets")
    ap.add_argument("--metadata-only", action="store_true", help="Download model tokenizer/config/code only; excludes weights")
    ap.add_argument("--revision", default=None, help="Optional explicit commit SHA override")
    ap.add_argument("--yes", action="store_true", help="Confirm downloading large files")
    args = ap.parse_args()

    if args.list or (not args.model and not args.dataset and not args.all_core):
        print("=== Configured Pretrained & Baseline Models ===")
        for k, v in models.items():
            target = paths["models"] / os.path.basename(v["local_dir"])
            print(
                f"  {k:18s} | repo={str(v.get('repo_id')):38s} | rev={str(v.get('revision')):8s} | path={target}"
            )
        print("\n=== Configured Datasets ===")
        for k, v in datasets.items():
            target = paths["raw"] / os.path.basename(v.get("local_dir", k))
            print(
                f"  {k:18s} | repo={str(v.get('repo_id')):38s} | rev={str(v.get('revision')):8s} | path={target}"
            )
        return

    if not args.yes:
        sys.exit("Refusing to start large downloads without --yes.")
    if args.metadata_only and (args.dataset or args.all_core):
        sys.exit("--metadata-only requires explicit --model entries and cannot download datasets")

    m_keys = list(args.model or [])
    d_keys = list(args.dataset or [])
    if args.all_core:
        m_keys = ["llada_8b_base", "nemotron_3b", "nemotron_8b", "nemotron_14b", "llm_as_jev_qwen", "bge_large_en"]
        d_keys = ["openjev", "toolret", "jevbench", "toolret_queries", "toolret_tools"]

    for mk in m_keys:
        out = download_model(mk, models[mk], str(paths["models"]), rev_override=args.revision, metadata_only=args.metadata_only)
        print(f"[model:{mk}] saved to {out}")
    for dk in d_keys:
        out = download_dataset(dk, datasets[dk], str(paths["raw"]), rev_override=args.revision)
        print(f"[dataset:{dk}] saved to {out}")


if __name__ == "__main__":
    main()
