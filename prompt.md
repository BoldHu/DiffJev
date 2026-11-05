# DiffJev research specification

This document and AGENTS.md jointly specify the method and exactly five core experiments. AGENTS.md supplies operational defaults. The equations and visibility rules below are normative; neither document reports measured research results.

## Reconciliation record (2026-11-05)

The former draft described a small-K listwise scorer, an IIA queue objective, a different backbone, four experimental suites, an unverified checkpoint, and an unmeasured 0.00% flip rate. Those statements conflicted with AGENTS.md. This revision aligns the specification with the agreed structured multi-slot method, LLaDA anchor, required Nemotron panel, K<=32 training cap, and five experiments. No superiority or exact floating-point invariance is asserted. Results require the specified experiments.

## 1. Research question and constraints

DiffJev investigates permutation-equivariant, multi-slot co-denoising for typed decisions. The hypotheses concern order/alias robustness, accuracy as candidate count grows, calibration, and decision cost. Candidate IDs and display aliases are metadata; meaningful class/tool names and descriptions remain input semantics. Serialized alternatives are a baseline, not the DiffJev input format.

Use GSAI-ML/LLaDA-8B-Base for method development and complete ablations. Add the instruction-tuned dense nvidia/Nemotron-Labs-Diffusion-3B, -8B, and -14B checkpoints for Exp 1, 3, 5. Each exact initial checkpoint independently initializes DiffJev, serialized one-pass DLM, and native causal AR adaptations. Never switch a DiffJev-only trained checkpoint into AR and call it a controlled baseline. No additional trainable embeddings, layers, adapters, or LoRA are introduced by DiffJev. Train the existing final four layers and final normalization; freeze embeddings, output head, and lower layers. Count unique parameters.

Model, tokenizer, remote implementation, and data revisions are frozen in configs/revisions.json. Formal package versions are in requirements-lock.txt; record the actual environment. Nemotron's nominal sizes do not replace parameter counting, and scaling does not isolate every pretraining difference. Published context configuration is 262,144; actual feasible requests must be measured.

## 2. Parallel branches and Set-RoPE

Let Q include all common task instructions, with q tokens. Candidate i has text o_i of length m_i and a decision slot x_i, initially MASK. Input is Q plus independent branches [MASK_i]+o_i. In one-based notation, query positions are 1..q, every slot is q+1, and every option starts at q+2. Code uses zero-based equivalents. Reset every semantic position-dependent factor, including Nemotron query scaling; KV cache storage offsets remain separate.

At each lower layer: query rows read query keys only; branch i (including its slot) reads query and branch i only; padding is excluded. Query reading all branches would violate independence indirectly. A bidirectional query-option alternative would need separate query copies and is outside this implementation. Encode branches in chunks without changing their mathematical outputs. Do not construct an all-candidate-text dense mask.

At the layer boundary retain h_i_base, query states, and own-option states. All slots in an upper denoising pass may read one another symmetrically with shared weights and tied slot positions. The first pass sees compressed slots only. Later passes additionally read static query/own-option detail memory; no slot directly reads another candidate's detail tokens.

Build separate static K/V at each of the final four layers. In this static path query rows read only query, and option rows read query plus their own text, without dynamic slots. Cache dependencies therefore remain fixed during a rollout; invalidate caches after parameter changes. Build detail memory only for survivors when pruning. A single boundary K/V tensor is not a cache for all layers.

Native adapter fidelity is checked independently of these deliberate visibility changes. LLaDA uses act(ff_proj(x))*up_proj(x), followed by ff_out. Nemotron reuses its native rotary module, including YaRN inverse frequencies and cos/sin scaling. Its native query factor is 1 + beta*log(1+floor(position/original_context_length)); use semantic positions. Its diffusion path discards masks, so the adapter must pass visibility to the actual attention kernel. Native contiguous-position and long-position block parity, mask isolation, and restricted-head parity gate formal use.

## 3. Structured probabilities and diffusion

For K explicit slots, legal clean vectors are all No or exactly one Yes. They represent reference a=0 and candidates a=1..K. The reference is virtual, with fixed energy zero; it has no learned branch.

For the same verified single-token Yes/No verbalizers at every slot:

    z_i = logit_i(Yes) - logit_i(No)
    log Z = logsumexp(0,z_1,...,z_K)
    P_0 = exp(-log Z),  P_i = exp(z_i-log Z).

For hard corrupted states, remove incompatible legal outcomes before normalization: revealed No excludes that candidate, and revealed Yes forces that action. Revealed competing Yes labels are illegal. Slot Bernoulli marginals are r_i=P_i, not sigmoid(z_i) for K>1. At K=1, P_1=sigmoid(z_1).

Use the existing vocabulary rows (or their exact difference) after the existing final normalization, including head bias. No full vocabulary projection is needed for DiffJev or eligible serialized DLM controls. Numerically stable log-complement and boundary probabilities are required.

Sample a legal clean action a from the full annotated target distribution tau. Convert it to its legal binary vector y, then independently mask each coordinate with probability t, leaving query/options clean. Half the corruption draws use t=1; otherwise t is uniform on [0.001,1). Full-mask draws use first-pass slot-only visibility; other draws may read static detail memory.

    L_mask = sum_i 1[x_t_i=MASK] BCE(y_i,P_i) / (K*t).

The denominator is the original K, not the realized mask count. Zero masked coordinates contribute zero. This is a structured composite denoising loss, not the unchanged native LLaDA ELBO.

Additionally train teacher-free, all-mask rollouts. At each pass use the entire hard or soft task target in structured cross-entropy:

    L_task = mean_passes[-sum_a tau_a log P_a]
    L_Brier = sum_a (P_final[a]-tau_a)^2
    L = L_mask + L_task + 0.1 L_Brier.

Training rollouts are unpruned. Cumulative Brier is an explicit optional score variant; Brier may be explicitly disabled. IIA weight defaults to zero, without a cross-batch queue. The no-mask-loss and other optional extensions are not core requirements. Record enabled terms.

## 4. Deterministic inference and recurrent feedback

Use three denoiser evaluations at t=1,2/3,1/3. Initially mu_i=(Yes:0,No:0,Mask:1). Before each evaluation reset the boundary input:

    h_i = h_i_base + mu_i.Yes E(Yes) + mu_i.No E(No)
                     + mu_i.Mask E(Mask) - E(Mask).

This is a parameter-free proposed boundary adapter, not an unchanged native DLM pass. Do not accumulate the previous hidden state repeatedly.

For t to s, beta=1-s/t, using the same old state and current denoiser distribution:

    mu_i.Yes_new = mu_i.Yes_old + beta*mu_i.Mask_old*P_i
    mu_i.No_new = mu_i.No_old + beta*mu_i.Mask_old*(1-P_i)
    mu_i.Mask_new = mu_i.Mask_old*s/t.

Stop gradients through feedback probabilities initially, recording that choice. Soft masses do not impose hard-state compatibility constraints. Return the final denoiser distribution at t=1/3, before any zero-noise update. This mean-field approximation does not preserve the full joint stochastic law.

A separate stochastic reference may sample one compatible legal action and independently reveal each still-masked coordinate with probability beta. It must not independently sample competing Yes actions. Couple noise by semantic identity in permutation tests.

## 5. Pruning, tasks, and calibration

No hard pruning for K<=32, noul, score, or full-support benchmarks/calibration. For other larger choice tasks evaluate K slots, keep 32 for pass two and 8 for pass three, preserving all exact cutoff ties. Update soft states first, then gather survivors. Tie-break an actual single decision by immutable semantic identity, never array position/display alias. Never rescue gold candidates or transfer removed mass to null.

Always retain the virtual reference in normalization. Pruned outputs include active IDs and probability_scope=retained_support; they are not the full-K posterior. Do not splice stale scores into an exact final distribution or claim removed candidates have known zero mass. Record gold survival after each pruning stage. First-pass all-slot attention remains quadratic in K.

- choice: K semantic alternatives, reference means none of these.
- noul: one positive proposition, reference means No; return sigmoid(z_1) and its complement, preserving original polarity.
- score: M-1 ordered interval bins; reference is the lowest bin, not abstention. Sort numeric representatives c_0<...<c_(M-1); put the complete rubric and reference definition in Q. Return sum_m c_m P_m and CDF cumulative sums. Do not use independent threshold logits.

For forced-choice benchmarks normalize candidate energies without the reference, consistently for decisions and probability metrics. Preserve native scales and official scoring. Fit one final-readout temperature per mode on the separate calibration split, then freeze it for every test K. Never temperature-scale internal reverse updates implicitly, and never tune a threshold on test data.

## 6. Data and training protocol

Use only OpenJevData-140k SFT (checked 100,345 rows: 86,023 hard, 14,322 soft) and ToolRet-Training-20w train (checked 208,826 rows), before filtering. No RL stage. OpenJev state and question both enter Q; options contain id/name/criteria; answer.kind and answer.probabilities preserve positional native labels. score_values align with native options before numeric sorting. ToolRet uses query, not generated prompt; parse its lists of JSON tool descriptions in positive and negative, deduplicate, and accept exactly one positive.

Effective batch32 has eight each: OpenJev general choice, noul, score, and ToolRet choice. Train3,125 AdamW updates at initial LR2e-5, bf16 where supported, clip norm1, microbatch1/accumulation32. This is100,000 presentations, not unique records. Controlled trained baselines replay a frozen accepted-source and actual-augmentation trace. Match checkpoint selection using development full-support decision CE. Report repetition, tokens, realized K, parameters, wall compute, revisions, and RNG state. Exact resume restores CPU/CUDA/Python/NumPy and local corruption/augmentation RNGs.

Freeze JevBench and ToolRet evaluation families before grouping/deduplication across sources. Then assign internal90/5/5 train/dev/calibration. Build banks and menus afterwards from training only. Audit normalized state/query, stable IDs, reordered families, and near duplicates; quarantine uncertain cases. Near-duplicate heuristics are not proof of zero overlap. Use ARC train, BoolQ train, or HelpSteer2 train only to replace insufficient corresponding buckets, with the same audit and joint splits. Record fallback use.

Filter original OpenJev K>32 rather than cropping. Preserve soft targets for task CE; sample one action only for the corrupted view. Explicit null-equivalent options merge their mass into the reference; a word containing "No" is not a null rule. IDs are content-derived semantic identities, never local aliases such as o0; meaningful punctuation/tool parameters are preserved.

Eligible hard single-positive choice augmentation proposes K={2,4,8,16,32} with probabilities{.50,.25,.15,.07,.03}. Preserve original OpenJev menus, renormalizing over target sizes at least as large. OpenJev bank extensions require audited task-valid negatives; do not inject unrelated class labels. ToolRet derives small menus from supplied negatives, extending from its train-only bank for K=32 while excluding every known positive and duplicate. Unjudged retrieval negatives remain a dataset assumption and must be audited. Null augmentation applies at15% only when all remaining options are certified invalid, never blindly to every hard label. Persist actual ordered IDs, targets, proposed/realized K, and null decisions. Soft-label/noul/score tasks never receive arbitrary negative injection.

## 7. Exactly five core experiments

1. **Benchmark performance:** the pinned original12-text-slice JevBench,8,016 cases/5,516 canonical families. Verify the actual manifest and restore licensed withheld states with hash checks. Run its pinned official evaluator, keep full support, treat SST5 as score, separate API/local latency. Report task-level official outputs and clearly labeled project aggregates. Include LLaDA and the3B/8B/14B panel.
2. **Order and external-label robustness:** up to1,000 fixed canonical families; enumerate K<=4 permutations, otherwise original plus at most10 seeded permutations. Preserve semantic IDs; inverse-map probabilities. Test arbitrary alias renaming separately. Report semantic flips, probability drift, accuracy changes, and missing coverage. Unavailable probabilities/failures are not zero drift.
3. **Candidate-count scaling:** ToolRet-Queries/ToolRet-Tools, query-only single-positive cases,512 stratified fixed queries (all if fewer;32/64 smoke). One nested random-negative pool and one frozen BM25-hard pool per query, K={4,16,64,256,1024,2048}, same candidates/content for all models. Exclude known positives and duplicate IDs/descriptions. Report Hit@1, gold survival, actual K, context/OOM failures, including size panel. K>32 is cardinality extrapolation. No real tool execution or full-corpus retrieval success claim.
4. **Controlled ablations:** one compact anchor table: full, full-context T=1, no cross-slot attention, sequential positions, no pruning, no candidate augmentation. Change one factor at a time. Train T=1 separately with full own-detail memory, same examples/decision loss/Brier/calibration, no corruption loss/iteration. A slot-only T=1 is optional and confounded with detail access. No-pruning uses the full checkpoint; other learned controls train independently. A matched-step no-L_mask control is needed only for stronger masked-objective attribution.
5. **Same-size latency and complexity:** reuse Exp3 BM25-hard manifests, fixed stratified100 queries, all six K. At each Nemotron size compare independently adapted native AR, serialized one-pass DLM, and DiffJevT=3/pruning, from the exact same initial weights. Reuse LLaDA serialized DLM/T=1/no-pruning in a compact attribution table. No full size-by-ablation grid.

Required references: available official Jev API, released LIQIIIII/Jeva-4B via its official runtime, pinned Qwen2.5-3B-Instruct LLM-as-Jev, LLaDA serialized DLM, and BGE or E5 retrieval in Exp3. The exact-checkpoint Nemotron AR/DLM controls are additional. djev is optional after public implementation verification. No invented MetaEncoder checkpoint. Use native prompts/APIs, actual baseline tokenizers, complete multi-token AR labels, and unavailable probability markers where appropriate.

## 8. Timing and complexity contract

Use one A10080GB, bf16, batch1, identical settings. Keep one method's weights resident, warm kernels>=5 requests, clear input-dependent caches per request, and include within-request caches. Measure100 fixed queries x3 repetitions; average repetitions within query, then p50/p95 with query bootstrap. Synchronize participating CUDA devices before wall timing and after output mapping. Include tokenization, packing, transfer, branch encoding, memory construction, all passes, selection, output mapping. Exclude loading/compilation. Measure GPU kernel-only time in separate profiling runs. Log hardware, placement, tokens/padding/chunks, backend, parameters, peak allocated memory including weights, raw failures, and profiler coverage.

AR returns one short valid decision with one prefill and cached further answer tokens, no CoT or unnecessary K prefills/probabilities. Serialized DLM runs once with one Yes/No slot per candidate and the same partition head. Permit equivalent SDPA, legal caching, and restricted projection. Preserve common semantic content and truncation policy, never truncate away gold candidates. OOM/context failures count toward coverage; compare speedups only on common successful queries. Pair cost with accuracy/survival; select accuracy-matched settings only on dev.

For q common-query tokens, average m option-text tokens, b=m+1, L layers, upper k=4, width d:

    Phi(n,a) = Theta(n*d^2 + a*d)
    B(K) = Phi(q+K*b, q^2+K*b*(q+b))
    M(s) = Phi(q+s*m, q^2+s*m*(q+m))
    S(n,delta) = Phi(n, n^2+delta*n*(q+m))
    C_DiffJev = (L-k)*B(K) + k*M(s)
               + k*(S(K,0)+S(s,1)+S(u,1)) + C_head_feedback_selection.

s,u are actual survivor counts (nominal32,8); without pruning s=u=K. Charge heterogeneous lengths, padding, repeated query work, projections/FFNs and detail-memory construction. Serialized DLM costs L*Phi(N,N^2); causal AR prefill L*Phi(N,N*(N+1)/2), plus actual decoding. Use actual method-specific N.

The intended replacement is global cross-candidate text attention O(L*K^2*m^2*d) with local branch work and upper slot attention O(k*K^2*d), shrinking later passes. Do not claim total O(K), universal gains over independent scoring, or measured speedup from a theoretical attention ratio. Efficient attention need not store N^2 probabilities. Count GFLOPs from executed tensor shapes at two FLOPs/MAC, explicitly audit fused/custom omissions and distinguish dense logical SDPA bounds from hardware skipped-tile arithmetic. Publish latency/K and GFLOPs/K size panels and accuracy/p50/p95/memory/speedup/coverage tables.

## 9. Verification and execution

Use tiny random/config-reduced fixtures locally, with no full weight/data downloads or research accuracy claims. Verify native FFN/RoPE/scaling parity, branch isolation, permutation agreement with measured tolerance, coupled stochastic noise, legal diffusion states, K=1 identity, valid score CDFs, cached/recomputed memory, recurrent boundary resets, pruning ties/identity/survival, no label leakage, and allowed parameter set. Real-weight/bf16/native integration and A100 measurements remain necessary before conclusions.

The eight notebooks00_setup through07_exp5_efficiency import src/ implementations and fail clearly on missing real assets. Formal workflows never silently run mocks. Keep datasets/weights/checkpoints/secrets out of Git. Read JEV_API_KEY privately from environment or an existing key file; no API call without explicit user opt-in. Track compact configurations, ID manifests, raw predictions, and measured failures. Plan three anchor/key-control training seeds when resources permit, one initial size-panel seed. Query bootstrap is not training-seed uncertainty. Group reordered twins by canonical family.

## 10. Primary sources and revision lock

- [R1] LLaDA model/config/implementation: https://huggingface.co/GSAI-ML/LLaDA-8B-Base ; https://github.com/ML-GSAI/LLaDA
- [R2] OpenJev SFT/RL schema and source notices: https://huggingface.co/datasets/shenjunhao/OpenJevData-140k
- [R3] ToolRet training: https://huggingface.co/datasets/mangopy/ToolRet-Training-20w
- [R4] ToolRet held-out queries/tools: https://huggingface.co/datasets/mangopy/ToolRet-Queries ; https://huggingface.co/datasets/mangopy/ToolRet-Tools
- [R5] Original JevBench cases: https://huggingface.co/datasets/Leanmcp/jevbench ; original evaluator commit251421a6f7c9a801038a617b172bc0cb967df076: https://github.com/Leanmcp/jevbench
- [R6] Jeva weights/runtime: https://huggingface.co/LIQIIIII/Jeva-4B ; https://github.com/LiQiiiii/Jeva
- [R7] AR reference: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- [R8] Retrieval references: https://huggingface.co/BAAI/bge-large-en-v1.5 ; https://huggingface.co/intfloat/e5-large-v2
- [R9] Conditional fallbacks: https://huggingface.co/datasets/allenai/ai2_arc ; https://huggingface.co/datasets/google/boolq ; https://huggingface.co/datasets/nvidia/HelpSteer2
- [R10] Official Jev protocol is recorded in the pinned JevBench common.py (systemone endpoint); API availability/results must be measured separately.
- [R11] Nemotron official checkpoints/config/code: https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B ; https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B ; https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B . Each snapshot includes modeling_nemotron_labs_diffusion.py, modeling_ministral.py and its rotary/query-scaling configuration. Exact SHAs are in configs/revisions.json.
