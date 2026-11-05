"""Declarative notebook 00 cells. Research/data implementations stay in src/."""


def setup_cells(markdown, code):
    return [
        markdown("## 1. Choose preparation actions\nThis is the single preparation entry point. It mounts Drive, clones or reuses your checkout, installs an isolated Python 3.12 environment, downloads selected pinned assets, restores local withheld benchmark states, and prepares frozen data/scaling manifests. It never trains a model or calls a paid API. Set the three action switches to True for a first complete setup. Leave them False to inspect/reuse existing assets. The Colab notebook kernel may be Python 3.13; research commands use the separate Python 3.12 worker."),
        code('''import os, sys, subprocess, json
from pathlib import Path
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL = "https://github.com/BoldHu/DiffJev.git"
# Optional exact code commit for a NEW clone; an existing checkout is never reset.
CODE_REVISION = None
MOUNT_DRIVE = IN_COLAB
local_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROJECT_DIR = "/content/drive/MyDrive/DiffJev" if IN_COLAB else str(local_root)
ENV_DIR = "/content/diffjev-env" if IN_COLAB else str(local_root / ".venv-formal")

INSTALL_ENV = False
DOWNLOAD_ASSETS = False
PREPARE_DATA = False

# Start with the anchor. Add panel entries when their GPU/storage runs are ready.
TRAIN_BACKBONES = ["llada_8b_base"]
# Full panel: ["llada_8b_base", "nemotron_3b", "nemotron_8b", "nemotron_14b"]
DOWNLOAD_REFERENCE_BASELINES = False  # Jeva, Qwen AR reference, BGE
EXTRA_DATASETS = []  # Only add fallback_choice/noul/score if an audit needs one.
SEED = 0
SCALING_QUERIES = 512
RESTORE_WITHHELD_STATES = True
SOURCE_ROOT = None  # default: <project>/data/raw/jevbench_sources
'''),
        markdown("## 2. Mount storage and locate the checkout\nUse PROJECT_DIR for the uploaded repository. Otherwise this cell clones it. Existing code is reused without pull/reset, so upload the revised notebook AND its scripts/src files before running. Drive stores assets, frozen manifests, checkpoints and logs; the Python environment is on the temporary Colab disk and must be recreated after a runtime reset."),
        code('''if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
ROOT = Path(PROJECT_DIR).expanduser().resolve()
if not (ROOT / "AGENTS.md").exists():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError("PROJECT_DIR is nonempty but is not this project; choose the uploaded checkout path")
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(ROOT)], check=True)
    if CODE_REVISION:
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", CODE_REVISION], check=True)
for required in ["src/diffjev/notebook_runtime.py", "scripts/prepare_workspace.py", "requirements-lock.txt"]:
    if not (ROOT / required).is_file():
        raise FileNotFoundError(f"Missing {required}; upload the updated project files, not only this notebook")
os.chdir(ROOT)
os.environ["DIFFJEV_ROOT"] = str(ROOT)
# Keep utility download paths consistent with the formal checkout-relative paths.
for key,folder in [("DIFFJEV_MODELS_DIR","models"),("DIFFJEV_DATA_DIR","data"),("DIFFJEV_OUTPUT_DIR","outputs")]:
    os.environ[key] = str(ROOT / folder)
sys.path.insert(0, str(ROOT / "src"))
from diffjev.notebook_runtime import NotebookRuntime, install_environment
print("Notebook kernel:", sys.version.split()[0])
print("Persistent project:", ROOT)
'''),
        markdown("## 3. Install the pinned worker environment\nINSTALL_ENV=True downloads Python 3.12 if necessary and installs requirements-lock.txt, including the official Jeva runtime. This does not replace Colab's kernel, so no kernel restart is required. Repeated installation reuses downloaded packages. Allow enough local disk for the environment and its package cache. No model weights or datasets are downloaded by this cell."),
        code('''if INSTALL_ENV:
    runtime = install_environment(ROOT, ENV_DIR)
else:
    worker = Path(ENV_DIR) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    runtime = NotebookRuntime(ROOT, worker)
    if not worker.exists():
        raise FileNotFoundError("Worker environment missing. Set INSTALL_ENV=True in cell 1 and rerun from there.")
    runtime.save_settings()
runtime.run([runtime.python, "scripts/prepare_workspace.py", "check"], "check_environment")
'''),
        markdown("## 4. Save the run selection and inspect the asset plan\nAll four backbone tokenizers/configs are needed for the shared context audit. The downloader now fetches those WITHOUT weights. Full weights are downloaded only for TRAIN_BACKBONES plus optional reference baselines. The five datasets include OpenJev SFT, ToolRet training, original JevBench cases, and held-out ToolRet Queries/Tools. No fallback dataset is added by default. The original release may withhold licensed benchmark state text; its manifest download is not a substitute for that text."),
        code('''CORE_MODELS = ["llada_8b_base", "nemotron_3b", "nemotron_8b", "nemotron_14b"]
if not TRAIN_BACKBONES or any(name not in CORE_MODELS for name in TRAIN_BACKBONES):
    raise ValueError("Choose one or more required backbone keys")
FULL_MODELS = list(TRAIN_BACKBONES)
if DOWNLOAD_REFERENCE_BASELINES:
    FULL_MODELS += ["jeva_4b", "llm_as_jev_qwen", "bge_large_en"]
DATASETS = ["openjev", "toolret", "jevbench", "toolret_queries", "toolret_tools"] + EXTRA_DATASETS
plan = dict(seed=SEED, backbones=TRAIN_BACKBONES, full_models=FULL_MODELS,
            datasets=DATASETS, scaling_queries=SCALING_QUERIES)
plan_path = ROOT / "outputs/setup/plan.json"
plan_path.write_text(json.dumps(plan, indent=2) + "\\n", encoding="utf-8")
runtime.python_code("""from pathlib import Path
import json, yaml
plan=json.loads(Path('outputs/setup/plan.json').read_text())
path=Path('configs/workflow.yaml')
cfg=yaml.safe_load(path.read_text())
cfg.update(seed=plan['seed'],backbones=plan['backbones'],scaling_queries=plan['scaling_queries'])
path.write_text(yaml.safe_dump(cfg,sort_keys=False))
print('Notebook run selection:',cfg['backbones'])
""", "save_run_selection")
print(json.dumps(plan, indent=2))
runtime.run([runtime.python, "scripts/download_assets.py", "--list"], "asset_plan")
'''),
        markdown("## 5. Download selected pinned assets\nSet DOWNLOAD_ASSETS=True to execute this cell's downloads. Re-running resumes/reuses snapshot files. Downloads require sufficient Drive quota and Colab disk; full panel weights alone are tens of GB. Existing Hugging Face authentication is used if configured; do not paste tokens into notebook cells. Jev API remains disabled."),
        code('''def download_arguments(option, names):
    return [part for name in names for part in [option, name]]
if DOWNLOAD_ASSETS:
    runtime.run([runtime.python, "scripts/download_assets.py", "--metadata-only", "--yes"]
                + download_arguments("--model", CORE_MODELS), "download_metadata")
    runtime.run([runtime.python, "scripts/download_assets.py", "--yes"]
                + download_arguments("--model", FULL_MODELS), "download_weights")
    runtime.run([runtime.python, "scripts/download_assets.py", "--yes"]
                + download_arguments("--dataset", DATASETS), "download_datasets")
else:
    print("Downloads disabled; existing local snapshots will be checked.")
'''),
        markdown("## 6. Restore benchmark text and prepare the shared data\nSet PREPARE_DATA=True. This runs the existing adapters, cross-source overlap audit, 90/5/5 splits, training-only banks, 3,125-update augmentation trace, and fixed 512-query random/BM25-hard scaling manifests. It is a substantial CPU/RAM operation, not training. Existing complete manifests are verified and reused; partial/mismatched preparation fails rather than silently changing comparisons.\n\nFor withheld JevBench states, upload the source files to data/raw/jevbench_sources using the layout in data/raw/jevbench/sources.json, or provide an already restored_states.jsonl. Restoration verifies state hashes. Missing licensed sources or a needed fallback partition produces an explicit error in this cell; supply those assets and rerun. Review audit.json before freezing experiments."),
        code('''if PREPARE_DATA:
    args = [runtime.python, "scripts/prepare_workspace.py", "prepare", "--seed", str(SEED),
            "--scaling-queries", str(SCALING_QUERIES)]
    if RESTORE_WITHHELD_STATES:
        args.append("--restore-states")
    if SOURCE_ROOT:
        args += ["--source-root", str(SOURCE_ROOT)]
    runtime.run(args, "prepare_data")
else:
    print("Preparation disabled; enable PREPARE_DATA for a first run.")
'''),
        markdown("## 7. Readiness report and next notebook\nThe report distinguishes tokenizer/config presence from full weight files and prepared data. It does not assert real-weight parity or GPU fit. Logs stream to outputs/setup/logs; environment, preparation and readiness reports are saved next to them. Once preparation is complete, notebook 01 is optional for rechecking/auditing. Start training in 02. Notebooks 01–07 use the same saved Python worker, so Colab's default Python/package versions do not affect formal execution. Enable reference downloads before running notebook 03's reference-model cells, and fetch the full panel before 07."),
        code('''runtime.run([runtime.python, "scripts/prepare_workspace.py", "status", "--seed", str(SEED)]
            + download_arguments("--model", FULL_MODELS), "readiness")
print("Read the report:", ROOT / "outputs/setup/readiness.json")
print("After readiness and audit review, open notebooks/02_train_diffjev.ipynb.")
'''),
    ]
