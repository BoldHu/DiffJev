# AGENTS.md — DiffJev Research and Implementation Guidelines

## 1. Mission and source of truth

DiffJev studies **permutation-equivariant multi-slot co-denoising** for decision models: independent candidate branches, a pretrained diffusion language model, symmetric slot interaction, and progressive candidate pruning. Preserve this architecture when implementing or refining the project.

The companion [prompt.md](prompt.md) defines the complete research specification, equations, source references, and rationale. This file defines implementation defaults, reproducibility requirements, and the minimal experimental scope. Keep both files consistent; record substantive methodological changes explicitly.

The 2026-11-05 reconciliation in `prompt.md` removes the obsolete small-K/listwise/IIA-queue draft and aligns it with this structured-diffusion protocol. Exact asset revisions are recorded in `configs/revisions.json`; formal Python 3.12 package pins are in `requirements-lock.txt`. The original JevBench evaluator is pinned separately from its case manifest. Native-weight and GPU validation must precede research claims.

Treat order robustness, large-candidate accuracy, calibration, and efficiency as hypotheses to evaluate. Do not write fabricated results, promise superiority, or describe an unmeasured flip rate as “0.00%.” Distinguish mathematical properties, implementation tests, proposed choices, and empirical findings.

The current document-editing task does not itself authorize running experiments, downloading model weights, invoking paid APIs, or pushing repository changes.

## 2. Architectural commitments

- Keep **GSAI-ML/LLaDA-8B-Base** as the method-development and complete-ablation anchor. Add the required same-family dense **Nemotron-Labs-Diffusion 3B/8B/14B** panel for Exp 1, Exp 3, and Exp 5. Pin each official revision, tokenizer, implementation, and dependencies.
- Add no trainable neural layers, embeddings, adapters, or LoRA modules. Reuse the backbone's embeddings, transformer blocks, normalization, and vocabulary head.
- Initially train the existing final **four** transformer layers and final normalization; freeze the lower layers, input embeddings, and output head. Report trainable and total parameter counts.
- Keep candidate identity separate from candidate meaning. External aliases and array indices are metadata, not model features; preserve meaningful tool names and descriptions.
- Read every decision slot using the same verified single-token Yes/No verbalizers. Never silently substitute the first token of a multi-token verbalizer.
- Do not serialize all candidate texts into one causal option list for DiffJev. Serialized input is a comparison baseline.
- Use the same lower-layer position/mask rules and upper-layer visibility during training and inference.
- Backbones beyond the specified size panel and auxiliary losses are optional extensions, not prerequisites for completing the five core experiments.

### 2.0 Required model-size panel and adapter checks

| Nominal size | Exact checkpoint |
|---|---|
| 3B | `nvidia/Nemotron-Labs-Diffusion-3B` |
| 8B | `nvidia/Nemotron-Labs-Diffusion-8B` |
| 14B | `nvidia/Nemotron-Labs-Diffusion-14B` |

Use these instruction-tuned IDs without `-Base`. Their published context configuration is 262,144 tokens; measure actual context/memory feasibility. Treat the names as nominal size labels and count unique total/trainable parameters in the instantiated models. Do not equate MoE active parameters with dense total parameters.

Each checkpoint supports AR and diffusion modes. At each size, separately adapt AR, serialized DLM, and DiffJev from the same initial checkpoint, with the same accepted source data, augmentation trace, update budget, and final-four-layer-plus-normalization training policy. Do not evaluate AR by switching the attention mode of a checkpoint trained only as DiffJev.

Reuse existing Nemotron embeddings, `encoder.layers`, `encoder.norm`, and `diffusion_head`; do not load optional self-speculation LoRA. Its inspected diffusion attention path discards the incoming mask: ensure the actual attention kernel receives the required branch/slot/memory visibility. Its query scaling also depends on `cache_position`; reset all semantic position-dependent factors consistently, and keep KV storage addresses separate from repeated semantic slot positions. Verify each pinned adapter before using it.

Run full robustness and ablation suites on the LLaDA anchor. Do not repeat every ablation across all three sizes. Same-family scaling still does not isolate parameter count from every pretraining difference. See `prompt.md` [R11] for the official model/config/code sources.

### 2.1 Branch packing and position rules

Pack a shared query `Q` and branches `[MASK_i] + option_i`. Query tokens retain ordinary within-query positions. Every slot receives position `L_q + 1`; every branch's option positions restart at `L_q + 2`. Include task instructions in `Q` when calculating its length.

Lower-layer attention must satisfy:

| Token group | Allowed keys/values |
|---|---|
| Shared query | Query only |
| Branch i, including its slot | Query and branch i only |
| Padding | Excluded |

This gives bidirectional processing inside each branch, conditioned on a candidate-independent query. Do not let the shared query read all branches: that creates indirect cross-branch communication. Fully bidirectional query–option interaction requires separate query copies and is a different implementation choice.

At the layer boundary, retain a base slot vector `h_i_base` and static context representations. Encode independent branches in chunks when necessary; chunking must preserve their mathematical outputs. Avoid constructing a dense attention mask over all candidate text tokens.

### 2.2 Top-layer denoiser and context memory

Reuse the final four layers on each denoising pass. All active slots may attend to one another symmetrically, with tied slot positions and shared weights.

The **first pass uses compressed slot states only**. Later passes additionally permit each slot to read static query/own-option detail memory, never another option's detail tokens directly. Build detail memory only for survivors when pruning is enabled.

Cache keys/values separately for each top layer. In the static context path, shared query rows read query rows only; candidate text rows read the query and their own text. Neither reads dynamic slots. A cache is reusable within a rollout because its dependencies remain fixed; invalidate it after relevant parameter updates. A boundary KV tensor is not automatically a valid cache for every layer.

The first all-candidate slot interaction still has quadratic attention cost in candidate count. Measure its actual runtime and memory; pruning does not make the entire algorithm linear in K.

## 3. Structured slot diffusion

### 3.1 Clean states and shared probability head

For K explicit slots, legal clean vectors contain either all No labels or exactly one Yes. Their outcomes are a virtual reference and candidates `1..K`. The reference has fixed energy zero; it is not an additional learned branch.

At each pass compute `z_i = logit_i(Yes) - logit_i(No)` and normalize energies `[0, z_1, ..., z_K]` using `logsumexp`:

`P_ref = 1 / (1 + sum_j exp(z_j)); P_i = exp(z_i) * P_ref`.

For a hard corrupted state, exclude clean outcomes incompatible with already revealed labels before normalization. The Bernoulli marginal for slot i is **r_i = P_i**, not `sigmoid(z_i)` when K > 1.

For a linear vocabulary head, compute only the two existing Yes/No rows, or their exact difference after final normalization, including an existing bias. Avoid unnecessary full-vocabulary projections, and permit the same optimization for eligible baselines.

### 3.2 Forward corruption and training loss

Sample one legal clean action from the annotated target distribution, then independently replace each slot label with `[MASK]` with probability t. Keep query and option text clean. For half of corruption draws use `t = 1`; otherwise sample uniformly from `[1e-3, 1)`. Full-mask draws use first-pass slot-only visibility; other draws can read static detail memory.

Use masked binary cross-entropy on the structured marginals:

`L_mask = sum_i 1[x_t_i == MASK] * BCE(y_i, P_i) / (K * t)`.

The denominator is K, not the random number of masked slots. An example with no masked slots contributes zero. Implement boundary probabilities stably. This is a structured composite denoising objective, not a claim to reproduce the native LLaDA ELBO exactly.

Also train teacher-free rollouts from an all-mask state, using the full hard or soft task target in structured cross-entropy at **every** pass. Do not feed ground-truth feedback into this rollout.

Default objective: `L_mask + mean_passes(L_task_CE) + 0.1 * L_Brier`, where `L_Brier = sum_a (P_final[a] - target[a])**2` on the final unpruned training rollout. Brier can be disabled explicitly; IIA regularization defaults to zero. These are initial settings, not established optima. For score, cumulative Brier can replace categorical Brier. Report enabled terms. Do not add auxiliary objectives merely to enlarge the method.

### 3.3 Deterministic inference and feedback

Use **T = 3** denoiser evaluations at `t = 1, 2/3, 1/3`. Initialize each slot's categorical state `mu_i = (Yes: 0, No: 0, Mask: 1)`.

Before each evaluation, reset its boundary input to:

`h_i = h_i_base + mu_i.Yes * E(Yes) + mu_i.No * E(No) + mu_i.Mask * E(Mask) - E(Mask)`.

This parameter-free boundary feedback is part of DiffJev's proposed adapter. Do not claim it is an unchanged native DLM forward pass or repeatedly accumulate previous hidden states.

For a transition from t to s, let `beta = 1 - s/t`, using probabilities from the current pass:

`mu_i.Yes += beta * mu_i.Mask * P_i`;
`mu_i.No += beta * mu_i.Mask * (1 - P_i)`;
`mu_i.Mask *= s/t`.

Compute all updates from the same old state. Use stop-gradient through feedback probabilities initially and record this choice. Do not apply hard-state compatibility constraints to soft masses. Return the final denoiser distribution at `t = 1/3`, **before** any update to zero noise. This deterministic mean-field reverse approximation does not preserve the full stochastic joint distribution.

A reference stochastic sampler may sample one compatible legal action per reverse step, then reveal each masked coordinate independently with probability beta. It must not independently sample competing Yes actions. Keep this reference separate from the default deterministic evaluation.

### 3.4 Pruning and output scope

Disable hard pruning for `K <= 32`, `noul`, `score`, and all full-support benchmark/calibration runs. For other larger choice problems, evaluate all K slots, retain 32 for the second pass, then retain 8 for the third. Keep the virtual reference in every normalization.

Select survivors by current candidate energies. Retain all exact ties at a cutoff, allowing the nominal cap to be exceeded. Any required single-choice tie break uses immutable semantic identity, not array position or display aliases. Update soft states from the current pass, then gather survivors; do not reassign removed mass to null. Do not rescue gold candidates during evaluation.

After pruning, report probabilities on **retained support**, with active candidate IDs and an explicit probability-scope field. These are not the original full-K probabilities. Do not combine stale scores and call them an exact final posterior or claim discarded candidates have known zero probability.

Record gold survival after each stage. Use the no-pruning path for full-support calibration. Training with the default maximum of 32 candidates does not require hard-pruning gradients.

## 4. Task adapters

| Mode | Explicit slots | Reference | Returned value |
|---|---|---|---|
| choice | K semantic alternatives | None of these | Candidate probabilities and reference probability |
| noul | One positive proposition | No | `P(Yes) = sigmoid(z_1)` and its complement |
| score | M−1 ordered interval bins | Lowest score bin | All M bin probabilities and their expected score |

For score, sort the numeric bin representatives `c_0 < ... < c_(M−1)` and return `sum_m c_m * P_m`. Include the full rubric, especially the reference bin definition, in the common query. The reference means the lowest bin, not abstention. Obtain a valid CDF by cumulative sums; do not reinterpret interval-bin logits as independent threshold probabilities.

For benchmarks requiring a forced choice, normalize over candidate energies without the reference and document that adapter. Preserve native target distributions, label polarity, numeric scale, and official scoring semantics.

Normalization alone does not establish calibration. Fit any temperature on a separate calibration split and fix it before test evaluation. Use a fixed final-readout temperature per mode; do not silently change internal denoising probabilities or fit a new temperature on each test K. Select any decision threshold before testing.

## 5. Data and training protocol

Use **shenjunhao/OpenJevData-140k**, `split="SFT"`, plus **mangopy/ToolRet-Training-20w**, `split="train"`. Use one supervised structured-denoising stage, without RL. The checked raw OpenJev SFT pool has 100,345 records (86,023 hard, 14,322 soft); ToolRet has approximately 209k records (checked release: 208,826). These counts precede filtering. Exclude OpenJev's separate RL pool.

### 5.1 Sampling mixture and fixed budget

| Sampling bucket | Share | Records per effective batch |
|---|---:|---:|
| OpenJev general choice | 25% | 8 |
| OpenJev noul | 25% | 8 |
| OpenJev score | 25% | 8 |
| ToolRet single-positive tool choice | 25% | 8 |

Use effective batch size **32** and **3,125 optimizer updates**, initially **100,000 example presentations per run**, or 25,000 per bucket. Sample with replacement when necessary; this is not a claim of 100,000 unique records. One presentation supplies the loss paths for one source example. Report unique counts, repetition rates, processed tokens, and realized candidate counts.

Use AdamW, initial learning rate `2e-5`, bf16 where supported, gradient clipping at 1, and microbatch accumulation/checkpointing as needed. A development run may revise the budget before freezing comparisons. Thereafter all required sizes and controlled trained baselines replay the same accepted source/augmentation trace and optimizer-update budget, except the explicitly changed factor in a designated ablation. Match the checkpoint-selection rule; report actual compute because equal updates are not equal FLOPs.

### 5.2 Adapters, splitting, and augmentation

Freeze evaluation families, audit overlap, group duplicate source problems/templates **across sources**, then assign internal train/development/calibration splits 90/5/5. Construct candidate banks, augmented menus, and null examples after splitting. Training banks draw only from the training partition. Evaluation inputs may be used for overlap auditing, not tuning or negative mining. Quarantine uncertain records and continue with accepted records.

If a bucket is insufficient, replace only its rejected/missing records using `allenai/ai2_arc` ARC-Easy/ARC-Challenge `train` for choice, `google/boolq` `train` for noul, or `nvidia/HelpSteer2` `train` with the helpfulness 0–4 rubric for score. Audit these fallbacks too and freeze one shared manifest. Do not add every fallback by default.

- OpenJevData contains hard and soft targets and native candidate counts extending beyond the training cap. Filter examples with original K > 32; do not crop their candidates and claim the original task was retained.
- Preserve soft targets in task cross-entropy. Sample a legal hard action from that distribution only to construct a corrupted training view.
- Map noul labels by semantic polarity. Sort score bins using numeric `score_values`, with the lowest bin as reference.
- If an explicit option is semantically identical to the virtual null reference, merge its target mass into that reference and remove the duplicate slot. Do not treat every text containing “No” as null.
- ToolRet training uses the original `query`, not the generated `prompt`. Keep exactly-one-positive records after tool deduplication; do not turn other positives into negatives. Its 15 supplied negatives allow K <= 16; obtain additional unique negatives from the training bank for K = 32.
- Deduplicate across sources and held-out evaluation families using stable IDs plus normalized text checks. Keep reordered twins and near-duplicate queries in the same split.
- For eligible hard single-positive choice training, propose `K = {2,4,8,16,32}` with probabilities `{0.50,0.25,0.15,0.07,0.03}`. Preserve original OpenJev menus and renormalize over target sizes at least that large. ToolRet negatives are a retrieval pool, so sample derived menus directly. Log realized K after null augmentation and filtering.
- Draw negatives from a versioned bank with relevant schema/task meaning. Do not inject unrelated class labels, additional valid answers, or arbitrary negatives into soft-label, noul, or score tasks.
- For 15% of eligible hard single-positive choice training examples, remove the positive only when no remaining candidate is valid, producing a reference target. Do not apply this blindly to all training tasks.
- Record the actual candidate-count histogram, data mixture, training budget, random seeds, and any removed examples.

Call evaluation at K > 32 **cardinality extrapolation beyond the training cap**. Never train on hundreds of injected candidates and describe those same counts as unseen.

## 6. Exactly five core experiments

Do not add an end-to-end agent execution suite or unrelated benchmark collection to the required plan.

### Exp 1 — Benchmark performance

Use the pinned original **Leanmcp/jevbench** text suite and its official evaluator. The documented release has 12 text datasets and 8,016 cases including reordered twins, corresponding to 5,516 canonical families; verify the pinned manifest rather than assuming all rows are independent. SST5 is a score task.

Report official aggregate results and task-level breakdowns. Add probability metrics only where the model supplies comparable probabilities and the benchmark supports them. Keep full support, avoid test-set calibration, and separate API latency from local GPU latency.

Include a compact 3B/8B/14B size panel alongside the LLaDA anchor and reference baselines. The size panel does not require every Exp 4 ablation to be repeated.

### Exp 2 — Option order and external-label robustness

Use up to 1,000 fixed canonical benchmark families. Enumerate permutations for K <= 4; otherwise use the original order and at most ten seeded random permutations. Preserve semantics and remap predictions to stable candidate identities.

Test arbitrary alias renaming separately from reordering. Do not change meaningful class/tool descriptions or present semantic label-definition changes as a pure permutation test. Report semantic decision flip rate, probability drift after inverse permutation, and accuracy change.

### Exp 3 — Long-tail candidate-count scaling

Use **ToolRet** as a query-only, retrieval-derived tool-routing task, restricted to single-positive queries. Fix 512 stratified evaluation queries, or all eligible queries if fewer; use 32 or 64 for smoke runs.

Use the pinned `mangopy/ToolRet-Queries` and `mangopy/ToolRet-Tools` evaluation releases, independently of `ToolRet-Training-20w`. Parse the JSON relevance labels, deduplicate semantic tools, and exclude queries with multiple known positives after grouping duplicate queries. Preserve the official category-specific tool corpus.

Evaluate `K = {4, 16, 64, 256, 1024, 2048}` with the same queries throughout. Construct one nested random-negative pool and one nested hard-negative pool per query, retaining the annotated positive. Rank hard negatives with a frozen, declared retriever; exclude known positives and deduplicate candidate descriptions/IDs.

All models receive identical candidate sets and truncation rules. Report Hit@1, stagewise gold survival, and failures by K, including the 3B/8B/14B panel. Reuse these manifests for formal latency, peak-memory, and FLOP comparisons in Exp 5. These are supplied-candidate routing results, not full-corpus retrieval or real tool execution success.

Lower-layer caches may accelerate accuracy sweeps over repeated candidate sets. Exp 5 measures fresh full requests on 100 fixed queries; disclose any warm-cache results separately. An optional no-answer probe is a diagnostic, not another mandatory experiment.

### Exp 4 — Controlled ablations

Use one compact table: full model plus `T = 1`, no cross-slot attention, untied sequential positions, no pruning, and no candidate augmentation. Change one factor at a time; do not run a factorial grid.

Use a separately trained **full-context T = 1** control: same branches, top layers, head, training examples, decision loss, Brier setting, and calibration policy, but no corruption loss or iterative state updates. Allow its own detail memory in its one pass. A truncated first pass can be an additional diagnostic, but would confound iterations with detail access. To isolate the masked objective from ordinary recurrence, add a matched-step no-`L_mask` control only if making that stronger claim. Equal examples/updates do not imply equal computation; report compute differences.

### Exp 5 — Same-size decision latency and computational complexity

Reuse Exp 3's fixed set of up to 512 queries, its **BM25-hard negative manifests**, and `K = {4,16,64,256,1024,2048}`. Select a fixed stratified 100-query subset for detailed timing, or all eligible queries if fewer. Random-negative accuracy remains in Exp 3; a second timing sweep is not required. At each Nemotron size, compare separately adapted **serialized native AR**, **serialized one-pass DLM**, and **DiffJev T=3 with pruning**, initialized from the exact same checkpoint.

AR returns one short valid decision without chain-of-thought, using one prompt prefill and cached further answer tokens. Correctly handle multi-token candidate IDs. The timing task is top-1 choice; do not require unnecessary full probability outputs or K full prompt prefills. The serialized DLM uses one Yes/No slot per candidate and the same partition head. Do not inflate its cost by requiring repeated full-prompt passes. If additional native DLM generation is measured, allow its valid caching and charge its actual execution graph.

Reuse the LLaDA 8B serialized baseline and Exp 4 full-context T=1/no-pruning controls in one small attribution table. Do not expand this into a full size-by-ablation grid.

**Timing contract:**

- One A100 80GB, bf16, batch size 1, identical device count and software settings. Match placement separately for any multi-GPU comparison.
- Keep weights resident and kernels warmed, but clear input-dependent caches for each request. Include within-request cache construction.
- At least five warmups per configuration; 100 queries, three repetitions each. Aggregate repetitions within a query first, then report query-level p50/p95 and query-bootstrap uncertainty.
- Measure complete request wall time from tokenization/packing through transfer, branch encoding, memory prefill, all decision passes, selection, and output mapping. Synchronize every participating CUDA device before starting the wall timer and after the final GPU-dependent output operation before stopping it. Exclude model loading/initial compilation. Report GPU-only time separately and profile costs in separate runs.
- Preserve candidate content and shared truncation policy. Record actual tokens, padding, chunk sizes, attention backend, compilation, total/trainable parameters, and peak allocated GPU memory including weights.
- Allow equivalent legal caching, optimized attention, and restricted vocabulary projection. Report unsupported contexts/OOM with coverage; compute speedup only on the same successful queries, never as infinity for a failed baseline.
- Pair cost with accuracy and stagewise gold survival. Select any accuracy-matched operating point on development data. Do not present accuracy sacrificed through pruning as an unqualified efficiency gain.

**Complexity contract:** use the complete formulation in `prompt.md`. For query length q, option-text length m, b=m+1, L layers, upper k=4, and width d, define

`Phi(n,a) = Theta(n*d^2 + a*d)`,
`B(K) = Phi(q+K*b, q^2+K*b*(q+b))`,
`M(s) = Phi(q+s*m, q^2+s*m*(q+m))`,
`S(n,delta) = Phi(n, n^2+delta*n*(q+m))`.

For actual survivor counts s and u (nominally 32 and 8), the three-pass total is

`C_DiffJev = (L-k)*B(K) + k*M(s) + k*(S(K,0)+S(s,1)+S(u,1)) + C_head_feedback_selection`.

With pruning disabled, s=u=K. Charge any repeated query work and actual heterogeneous lengths/padding. A serialized one-pass DLM costs `L*Phi(N,N^2)`; causal AR prefill costs `L*Phi(N,N*(N+1)/2)` plus actual answer decoding. Use each method's actual N and attention graph.

The intended advantage is replacing cross-candidate **text-token** interaction, whose dense global term is `O(L*K^2*m^2*d)`, with local branch work and upper-layer **slot** interaction `O(k*K^2*d)`, then shrinking later passes. Total cost also contains linear-in-token projections/FFNs and memory construction. The first slot pass remains quadratic in K; never claim overall O(K), universally lower complexity than independent scoring, or a theoretical attention ratio as measured speedup.

Estimate full-request GFLOPs from actual operator shapes using two FLOPs per multiply-add. Audit profiler omissions for fused/custom kernels. Separate arithmetic from resident-memory complexity: efficient attention need not materialize an N-by-N matrix. Produce latency-versus-K and GFLOPs-versus-K plots with size panels, plus a compact accuracy/p50/p95/memory/speedup/coverage table.

## 7. Baselines and fair comparisons

Required comparisons are the available official **Jev API**, **Jeva-4B**, a pinned autoregressive **LLM-as-Jev** implementation, and a **serialized DLM baseline using the same LLaDA backbone**. Use BGE or E5 as a retrieval baseline in Exp 3.

For Exp 5, additionally require the exact-checkpoint Nemotron AR and serialized DLM controls at each of 3B/8B/14B. Do not use a hosted API, a different-size released model, or a smaller active-parameter MoE as the sole proof of matched-size efficiency.

Treat `djev` as an optional cross-backbone comparison only after verifying its public implementation and revision. Do not invent a `MetaEncoder-4B` checkpoint or substitute an unverified model silently.

Use native baseline prompting and probability APIs when available. For autoregressive alias probabilities, score complete valid labels, including multi-token labels; do not compare only their first token. Report unavailable probabilities as unavailable.

Share evaluation queries, candidate pools, semantic text, seeds, and budgets. Do not truncate away gold options to fit a baseline. Report context-limit failures explicitly. Distinguish frozen/API references from models trained on the project data, and measured results from published reference values.

## 8. Environment, notebooks, and repository workflow

Local verification uses the **RTX 3050 12 GB** in the `Jev` Conda environment. Use a tiny random/config-reduced model and toy data for masks, losses, caches, and probability tests. Do not download the full backbone or datasets for this stage or interpret smoke accuracy as research evidence.

Core experiments use Colab GPUs; larger training/scaling can use the user's **three 80 GB A100s**. Check actual GPU memory before selecting precision, batch size, and accumulation. Do not promise that an 8B backbone fits every free Colab runtime. Keep any multi-GPU implementation subordinate to the same notebook configuration.

Required notebook sequence:

| Notebook | Purpose |
|---|---|
| `00_setup.ipynb` | Colab storage/checkout, isolated Python 3.12, optional pinned downloads, shared data preparation and readiness |
| `01_prepare_data.ipynb` | Optional preparation rerun/audit: provenance, adapters, deduplication, splits, candidate manifests |
| `02_train_diffjev.ipynb` | Training, validation, checkpoint/resume |
| `03_run_baselines.ipynb` | Baseline setup and standardized adapters |
| `04_exp1_exp2.ipynb` | Benchmark and robustness evaluation |
| `05_exp3_scaling.ipynb` | Candidate-count accuracy and pruning; shared Exp 5 manifests |
| `06_exp4_ablation.ipynb` | Controlled ablations and report tables |
| `07_exp5_efficiency.ipynb` | Matched-size timing, total FLOPs, memory, and complexity comparison |

Notebooks must run top-to-bottom after checkout and import reusable code from `src/`; avoid copying the implementation into each notebook. Keep model/attention, diffusion, task adapters, data, evaluation, and baseline code modular. `tests/` contains meaningful invariant tests; `README.md` specifies execution order and manual upload paths for weights, data, and baseline assets.

Notebook 00 is the single preparation entry point; 01 remains an optional audit/reuse interface. On hosted Colab, the frontend kernel handles Drive while research commands in 00–07 run in the saved Python 3.12 worker environment. Recreate this temporary environment after runtime resets. Download only selected full models, but obtain all four backbone tokenizers/configs for shared context acceptance. Do not silently replace frozen preparation on reruns. These are execution changes, not changes to the five experiments or method.

The intended Git remote is `https://github.com/BoldHu/DiffJev.git`. Track source, notebooks, configuration, manifests, and compact results. Exclude raw datasets, checkpoints, caches, and secrets. Use `JEV_API_KEY` from the environment or an existing `JEV_API_KEY.txt` without displaying, committing, or embedding its contents in outputs.

## 9. Verification and reporting

Before expensive experiments, verify branch isolation; inverse-permutation logit/probability agreement; coupled stochastic noise; legal forward/reverse states; the K = 1 sigmoid identity; valid score distributions; cached versus recomputed memory; pruning identity/survival; and absence of label leakage in teacher-free inference.

Use numerical tolerances appropriate to the tested precision and report actual drift. Include adversarial ties and heterogeneous option lengths. Verify the trainable-parameter set and that no forbidden new parameters were introduced.

Use fixed candidate manifests and evaluation seeds. Plan three training seeds for the LLaDA anchor and key learned controls when resources permit; the required size panel initially uses one fixed seed to contain scope. Report actual counts and do not call query-bootstrap intervals training-seed uncertainty. Group evaluation uncertainty by canonical query/family, not reordered copies. Save run configurations, revisions, shared sample/augmentation traces, raw predictions, timing boundaries, measured failures, and profiler coverage.

Stop adding optional infrastructure once the defined method and five experiments are reproducible. Prefer a clear implementation and honest evidence over untested architectural claims or expanding the research scope.
