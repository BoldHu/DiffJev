"""Environment helpers: paths, Google Drive / Cloud mounting, logging, artifact persistence.

Directory roots are resolved from environment variables so the same code runs
locally, on Google Colab (Drive-mounted or GCS-backed), and on the 3x A100 server:

    DIFFJEV_ROOT        project root (default: repository root; on Colab: /content/drive/MyDrive/DiffJev)
    DIFFJEV_MODELS_DIR  default  <root>/models
    DIFFJEV_DATA_DIR    default  <root>/data
    DIFFJEV_OUTPUT_DIR  default  <root>/outputs   (checkpoints, results, tables, figures, logs)
    DIFFJEV_GCS_BUCKET  optional gs://<bucket>/DiffJev for Google Cloud Storage syncing
    HF_HOME             HuggingFace cache (optional)
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def project_paths() -> Dict[str, Path]:
    root = Path(os.environ.get("DIFFJEV_ROOT", REPO_ROOT))
    paths = {
        "root": root,
        "models": Path(os.environ.get("DIFFJEV_MODELS_DIR", root / "models")),
        "data": Path(os.environ.get("DIFFJEV_DATA_DIR", root / "data")),
        "outputs": Path(os.environ.get("DIFFJEV_OUTPUT_DIR", root / "outputs")),
        "configs": REPO_ROOT / "configs",
    }
    paths["raw"] = paths["data"] / "raw"
    paths["processed"] = paths["data"] / "processed"
    paths["manifests"] = paths["data"] / "manifests"
    paths["checkpoints"] = paths["outputs"] / "checkpoints"
    paths["results"] = paths["outputs"] / "results"
    paths["tables"] = paths["outputs"] / "tables"
    paths["figures"] = paths["outputs"] / "figures"
    paths["logs"] = paths["outputs"] / "logs"
    return paths


def ensure_dirs(paths: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
    paths = paths or project_paths()
    for k in (
        "models",
        "raw",
        "processed",
        "manifests",
        "checkpoints",
        "results",
        "tables",
        "figures",
        "logs",
    ):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def mount_cloud_storage(
    drive_mount_point: str = "/content/drive",
    drive_subdir: str = "MyDrive/DiffJev",
    gcs_bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """Mount Google Drive (and optional Google Cloud Storage bucket) when on Colab.

    Sets ``DIFFJEV_ROOT`` and ``HF_HOME`` to persistent storage so models, data,
    manifests, checkpoints, logs, tables, and figures survive Colab resets.
    """
    in_colab = "google.colab" in sys.modules
    status: Dict[str, Any] = {
        "in_colab": in_colab,
        "drive_mounted": False,
        "gcs_bucket": gcs_bucket or os.environ.get("DIFFJEV_GCS_BUCKET"),
        "persistent_root": None,
    }
    if in_colab:
        try:
            from google.colab import drive  # type: ignore

            if not os.path.ismount(drive_mount_point):
                drive.mount(drive_mount_point, force_remount=False)
            status["drive_mounted"] = True
            persistent_root = os.path.join(drive_mount_point, drive_subdir)
            os.makedirs(persistent_root, exist_ok=True)
            os.environ.setdefault("DIFFJEV_ROOT", persistent_root)
            os.environ.setdefault("HF_HOME", os.path.join(persistent_root, "hf_cache"))
            status["persistent_root"] = persistent_root
        except Exception as e:
            status["drive_error"] = str(e)

        bucket = status["gcs_bucket"]
        if bucket:
            try:
                from google.colab import auth  # type: ignore

                auth.authenticate_user()
                status["gcs_authenticated"] = True
            except Exception as e:
                status["gcs_auth_error"] = str(e)

    paths = ensure_dirs(project_paths())
    status["resolved_root"] = str(paths["root"])
    return status


def sync_to_gcs(local_dir: Optional[Path] = None, gcs_uri: Optional[str] = None) -> bool:
    """Optionally sync ``outputs/`` and ``data/manifests/`` to a ``gs://`` bucket."""
    bucket = gcs_uri or os.environ.get("DIFFJEV_GCS_BUCKET")
    if not bucket:
        return False
    paths = project_paths()
    src = str(local_dir or paths["outputs"])
    dst = f"{bucket.rstrip('/')}/outputs"
    res = subprocess.run(["gsutil", "-m", "rsync", "-r", src, dst], check=False)
    return res.returncode == 0


class ExperimentLogger:
    """Structured logger that persists logs, JSONs, JSONLs, CSVs, and LaTeX tables."""

    def __init__(self, stage_name: str, paths: Optional[Dict[str, Path]] = None):
        self.stage_name = stage_name
        self.paths = ensure_dirs(paths)
        self.log_file = self.paths["logs"] / f"{stage_name}.log"
        self.jsonl_file = self.paths["logs"] / f"{stage_name}_events.jsonl"

        self.logger = logging.getLogger(f"diffjev.{stage_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def info(self, msg: str, **payload: Any) -> None:
        self.logger.info(msg)
        event = {
            "ts_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "stage": self.stage_name,
            "message": msg,
            **payload,
        }
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def save_json(self, name: str, data: Any, subdir: str = "results") -> Path:
        target_dir = self.paths.get(subdir, self.paths["results"])
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = name if name.endswith(".json") else f"{name}.json"
        out_path = target_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self.info(f"Saved JSON artifact: {out_path}")
        return out_path

    def save_jsonl(self, name: str, rows: Iterable[Dict[str, Any]], subdir: str = "results") -> Path:
        target_dir = self.paths.get(subdir, self.paths["results"])
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = name if name.endswith(".jsonl") else f"{name}.jsonl"
        out_path = target_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        self.info(f"Saved JSONL predictions: {out_path}")
        return out_path

    def save_csv(self, name: str, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> Path:
        self.paths["tables"].mkdir(parents=True, exist_ok=True)
        fname = name if name.endswith(".csv") else f"{name}.csv"
        out_path = self.paths["tables"] / fname
        if not rows:
            out_path.write_text("", encoding="utf-8")
            return out_path
        cols = list(columns) if columns else list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        self.info(f"Saved CSV table: {out_path}")
        return out_path

    def save_latex(self, name: str, latex_str: str) -> Path:
        self.paths["tables"].mkdir(parents=True, exist_ok=True)
        fname = name if name.endswith(".tex") else f"{name}.tex"
        out_path = self.paths["tables"] / fname
        out_path.write_text(latex_str, encoding="utf-8")
        self.info(f"Saved LaTeX table: {out_path}")
        return out_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def environment_report() -> Dict[str, object]:
    rep: Dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        rep["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                "bf16": torch.cuda.is_bf16_supported(),
            }
            for i in range(torch.cuda.device_count())
        ]
    for mod in ("transformers", "datasets", "huggingface_hub", "accelerate", "matplotlib"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:
            rep[mod] = None
    return rep


def load_jev_api_key() -> Optional[str]:
    """Return JEV_API_KEY from the environment or an existing JEV_API_KEY.txt.

    The value is never printed, logged or written anywhere by this project.
    """
    key = os.environ.get("JEV_API_KEY")
    if key:
        return key.strip()
    f = project_paths()["root"] / "JEV_API_KEY.txt"
    if f.is_file():
        return f.read_text(encoding="utf-8").strip() or None
    return None
