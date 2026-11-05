# DiffJev

DiffJev implements branch-parallel, permutation-equivariant slot co-denoising with a structured reference/candidate head. `AGENTS.md` and `prompt.md` define the research protocol. No research results have been established by the local fixtures.

## Environments and assets

Local invariant tests need the small core dependencies in `requirements.txt`; they use random tiny models and no network. The formal runtime uses **Python 3.12** and `requirements-lock.txt`, including the official Jeva runtime at a fixed Git revision. Notebook 00 installs an isolated Python 3.12 worker using pinned uv, even if Colab's frontend kernel uses Python 3.13. Notebooks 01–07 launch research cells through that worker. On a server, install the lock in a Python 3.12 environment and install the project with `pip install -e . --no-deps`. The formal dependency combination and native adapters must be validated with the pinned real snapshots before experiments; it has not been tested by downloading their weights here.

Set `DIFFJEV_ROOT` to the checkout when launching notebooks outside its root. On Colab, clone/upload the repository and set this path to the persistent Drive checkout. Do not assume any free Colab GPU can train an 8B model. Training processes use one GPU each; the three A100s can host independent runs. Exp 5 requires one visible A100 80GB (`CUDA_VISIBLE_DEVICES`) and bf16.

Inspect `configs/models.yaml`, `configs/data.yaml`, and `configs/revisions.json` for exact snapshots. Manual uploads must preserve all config/tokenizer/remote-code/weight files and include `download_provenance.json` with `repo_id` and `resolved_sha` matching the lock. The provided downloader writes this file; loaders are local-only and reject mismatches.

```bash
python scripts/download_assets.py --list
# Optional large downloads, only when deliberately requested on an experiment machine:
python scripts/download_assets.py --all-core --yes
python scripts/download_assets.py --model jeva_4b --yes
```

Upload paths:

- `models/GSAI-ML__LLaDA-8B-Base/`, `models/nvidia__Nemotron-Labs-Diffusion-{3B,8B,14B}/`.
- `models/Jeva-4B/` (official `config.yaml`, `head.pt`, backbone, tokenizer and adapter); Qwen/BGE/E5 paths are in `configs/models.yaml`.
- `data/raw/openjev/data/SFT.parquet` (RL excluded); `data/raw/toolret/ToolRet-Training-20w/train-*.parquet`.
- `data/raw/jevbench/cases/*.jsonl`, only the original 12 text slices. For licensed withheld states, supply `data/raw/jevbench/restored_states.jsonl` with `case_id` and original `state`. Every state hash is checked; missing states stop preparation instead of silently reducing the benchmark. The pinned release identifies upstream sources/row indices for lawful restoration.
- `data/raw/toolret_queries/<source>/queries-*.parquet` and `data/raw/toolret_tools/<category>/tools-*.parquet` are the official held-out ToolRet releases, separate from training.
- Fallback files under `data/raw/fallback_{choice,noul,score}/` are loaded only if the corresponding accepted training bucket is insufficient.

Never print, commit, or embed `JEV_API_KEY`. The API baseline reads the environment or existing `JEV_API_KEY.txt`. No API is invoked during setup/tests. Formal API evaluation requires `--allow-api` or the explicit shared notebook flag.

## Execution order

For Colab, start with the updated `00_setup.ipynb`; upload the accompanying `src/`, `scripts/`, and dependency files too. In its first configuration cell set `PROJECT_DIR` to the uploaded checkout, then enable `INSTALL_ENV`, `DOWNLOAD_ASSETS`, and `PREPARE_DATA` for first preparation. The default `TRAIN_BACKBONES = ["llada_8b_base"]` fetches anchor weights only; metadata for all four backbones is fetched for the shared context audit. Enable `DOWNLOAD_REFERENCE_BASELINES` before reference-model cells in 03, or add panel entries for later runs. Snapshot downloads are reusable; existing complete data manifests are checksum-verified and never silently regenerated. Preparation can consume significant CPU RAM/time but does not train or call paid APIs.

The notebook mounts Drive, clones only when the specified checkout is missing, installs dependencies, downloads pinned models/training/benchmark assets, optionally restores withheld benchmark text from already supplied original sources, prepares splits/augmentation/scaling manifests, and saves a readiness report. Licensed source text is not automatically fetched: use the paths from the downloaded `jevbench/sources.json`. Missing source files, fallback partitions, or incomplete prepared outputs produce explicit errors.

The worker defaults to `/content/diffjev-env`, so recreate it with `INSTALL_ENV=True` after a Colab runtime reset. Drive artifacts survive. There is no need to replace or restart Colab's kernel for this worker installation. Persisted interpreter settings and reports live in `outputs/setup/`, and subprocess output streams to `outputs/setup/logs/`. Later notebooks run each research cell in a fresh worker process; arbitrary Python variables do not persist between research cells, while saved models/manifests do. The supplied cells are self-contained.

The notebooks are generated by `python scripts/build_notebooks.py`. They contain real workflows, no synthetic fallback or fake baseline metrics. Run in this order after providing the required assets:

1. `00_setup.ipynb`: complete preparation, including environment/provenance, storage, installation/downloads, frozen data and scaling pools.
2. `01_prepare_data.ipynb`: optional audit/reuse of preparation already performed by 00; can prepare separately if skipped there.
3. `02_train_diffjev.ipynb`: pretrained anchor/size-panel adaptations, development selection, resume and calibration.
4. `03_run_baselines.ipynb`: independently train serialized DLM and same-checkpoint AR controls; load official reference models.
5. `04_exp1_exp2.ipynb`: original official JevBench evaluator and canonical-family order/alias tests.
6. `05_exp3_scaling.ipynb`: shared nested random/BM25-hard candidate sets, actual K, accuracy, survival and failures.
7. `06_exp4_ablation.ipynb`: separately trained one-factor controls, full-context T=1, and compact table.
8. `07_exp5_efficiency.ipynb`: fresh complete requests, separate operator/kernel profiles, matched-query speedups, memory, coverage and plots.

`configs/workflow.yaml` selects runs, seed and API opt-in. Run all required panel entries; optional multi-seed expansion is explicit. The same operations are available without notebooks:

```bash
python scripts/run_experiments.py prepare
python scripts/run_experiments.py manifests
python scripts/run_experiments.py train --backbone llada_8b_base
python scripts/run_experiments.py train --backbone nemotron_3b --method ar
python scripts/run_experiments.py train --backbone llada_8b_base --method serialized_dlm
python scripts/run_experiments.py train --variant full_context_t1_control
python scripts/run_experiments.py exp1-exp2 --backbone llada_8b_base
python scripts/run_experiments.py exp1-exp2 --reference jeva
python scripts/run_experiments.py exp3 --backbone nemotron_3b
python scripts/run_experiments.py exp4 --variant no_pruning
python scripts/run_experiments.py exp5 --backbone nemotron_3b
python scripts/run_experiments.py exp5 --backbone llada_8b_base --attribution
python scripts/run_experiments.py report
```

For withheld JevBench states, the optional `restore-states --source-root data/raw/jevbench_sources` command runs the pinned official source adapters on already uploaded source files and checks every resulting hash. Use the local layout in the pinned `sources.json`. Preparation also requires the pinned backbone tokenizer/config files: it applies one shared context-feasibility filter across all required trained controls before freezing the trace, without loading weights.

Repeat the controlled training/evaluation commands for Nemotron 8B/14B and the remaining required ablations; the notebooks contain those loops. Resume with `train ... --resume <run>/step_500.pt`. Evaluation requires the separately saved `calibrated.pt` for the requested method. It cannot silently fall back to fresh random or differently trained weights.

## Data and result audit

Preparation persists accepted records, splits, candidate banks, exact ordered augmentation menus/targets, rejected row reasons, revision hashes and overlap-audit settings in `data/processed/formal/`. OpenJev criteria and soft targets are preserved; ToolRet JSON tool descriptions use stable semantic IDs. OpenJev cross-record negatives require task-valid reviewed IDs in `data/manifests/audit_overrides.jsonl` (`rid`, `eligible_negative_ids`). `null_certified` authorizes null augmentation only when its semantic conditions have been audited. ToolRet small menus use supplied negatives; extra bank negatives exclude known positives and remain unjudged-negative assumptions. Actual proposed/realized K is saved instead of an idealized histogram.

Approximate near-duplicate detection is not evidence of zero contamination. Inspect the saved audit/quarantine reports before freezing comparisons. Never use evaluation queries for negative mining or temperature fitting.

Runs write `outputs/formal/<backbone>/<method>/<variant>/seed_<seed>/`. Checkpoints contain only permitted trainable tensors, optimizer/RNG state, configuration and provenance. Development selection uses final full-support decision CE, common across controls; calibration uses the separate calibration partition. Native adapter invariant/parity checks run before using real snapshots and save measured drift.

Original JevBench evaluator files are preserved under `third_party/jevbench` with commit and SHA256 provenance. `official/scores.json` contains its task metrics and family uncertainty. The original scorer has no universal scalar aggregate; project summaries are labeled separately. Score tasks retain MAE/RPS/rounding semantics. Exp 2 reports incomplete coverage rather than treating failures/missing probabilities as zero flips or drift.

Exp 5 measures tokenization through output mapping with gradients off, model eval mode, fresh caches, device synchronization and resident weights. GPU kernel-only time is profiled separately. Executed-shape FLOPs include actual projection/head/attention shapes and actual pruning/cached decoding paths. Fused causal SDPA counts triangular logical pairs; its dense bound and unclassified operators are retained for audit. These are not hardware instruction counts. Training FLOPs are unavailable without a separate training profile; actual wall time/tokens/parameter counts are reported instead of a fabricated 3x multiplier. Preserve raw profiles and inspect omissions before publication.

Failures and unsupported contexts remain in coverage. Speedups use common successful queries. Tables/plots are built only from saved results and never filled with hardcoded values. Query bootstrap does not estimate training-seed uncertainty.

## Local verification and repository policy

```bash
python -m pytest -q
python scripts/smoke_train.py
```

The smoke script and synthetic fixtures are local verification only. They never establish accuracy, calibration or speed claims. Formal notebook execution, real-weight parity across all sizes, full data audits, GPU memory feasibility and paid API responses require the actual assets/environment and were not run while implementing these fixes.

The intended remote is `https://github.com/BoldHu/DiffJev.git`. Raw data, model files, checkpoints, caches, secrets and large result files are ignored. Preserve compact configuration/ID manifests and approved result summaries separately; do not push without explicit authorization.
