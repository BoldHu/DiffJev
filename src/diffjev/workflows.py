"""Formal, local-asset workflows shared by the CLI and all eight notebooks.

There are no synthetic fallbacks. Expensive work runs only when the user calls
the corresponding train/evaluate function or notebook cell.
"""
from dataclasses import asdict, is_dataclass
from pathlib import Path
from collections import Counter
import copy
import gc
import json
import math
import random
import torch

from .assets import ROOT, environment_record, load_backbone, model_specs, read_parquet_snapshot, verify_snapshot
from .config import DiffJevConfig, TrainConfig, load_yaml_config
from .data.native import digest, semantic_id
from .data.sources import (map_openjev_record, map_toolret_record, map_jevbench_record,
    map_arc_record, map_boolq_record, map_helpsteer2_record, load_local_dataset,
    materialize_example, prepare_training_manifest, record_dedup_keys)
from .data.audit import audit_groups
from .data.splits import split_records, normalize_text
from .data.toolret_eval import adapt_toolret_evaluation
from .model import DiffJevModel
from .packing import Candidate
from .tasks import REFERENCE_ID, build_choice, _enc
from .training import (train_diffjev, save_trainable_checkpoint, load_trainable_checkpoint,
                       apply_freeze_policy, parameter_counts)
from .calibration import calibrate_model
from .evaluation.exp4_ablation import build_ablation_config

PREPARED = ROOT / "data/processed/formal"
CORE_ABLATIONS = ("full", "full_context_t1_control", "no_cross_slot", "untied_positions", "no_pruning", "no_augmentation")
BACKBONES = ("llada_8b_base", "nemotron_3b", "nemotron_8b", "nemotron_14b")


def clean_json(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapped(rows, mapper, source, rejects):
    result = []
    for index, row in enumerate(rows):
        try:
            mapped = list(mapper(row))
            if not mapped:
                rejects.append({"source":source,"row":index,"reason":"filtered_by_adapter"})
            for rec in mapped:
                # Full digest prevents prefix collisions in fallback source IDs.
                if rec.get("is_fallback"):
                    rec["rid"] = source+":"+digest(row)
                rec["source"] = source
                if rec["mode"] == "choice" and not rec.get("is_retrieval_pool"):
                    mapping = {cid: semantic_id(text) for cid,text in rec["options"]}
                    if len(set(mapping.values())) != len(mapping):
                        raise ValueError("duplicate semantic alternatives")
                    rec["options"] = [(mapping[c],t) for c,t in rec["options"]]
                    rec["target"] = {mapping.get(c,c):p for c,p in rec["target"].items()}
                result.append(rec)
        except (ValueError, KeyError, TypeError) as error:
            rejects.append({"source":source,"row":index,"reason":type(error).__name__+":"+str(error)})
    return result


def prepare_data(seed=0, num_updates=3125):
    """Freeze held-out families, audit all accepted sources, split, then augment."""
    cfg = load_yaml_config(str(ROOT/"configs/data.yaml"))
    provenance = {}
    for key in ("openjev","toolret","jevbench","toolret_queries","toolret_tools"):
        spec = (cfg["training"] | cfg["evaluation"])[key]
        provenance[key] = verify_snapshot(ROOT/spec["local_dir"], spec["repo_id"])
    raw_bench = load_local_dataset("jevbench", str(ROOT/"data"))
    # Missing withheld states are a hard error: do not silently shrink the test suite.
    benchmark = [map_jevbench_record(row) for row in raw_bench]
    duplicate_cases = [{"case_id":row["case_id"],"family_id":row["family_id"],"dataset":row["dataset"],
                        "groups":row["duplicate_option_groups"]}
                       for row in benchmark if row.get("duplicate_option_groups")]
    print(f"JevBench parsed: {len(benchmark)} cases; {len(duplicate_cases)} cases have duplicate option text (all retained).",flush=True)
    query_rows = read_parquet_snapshot("toolret_queries", "queries", with_config=True)
    tool_rows = read_parquet_snapshot("toolret_tools", "tools", with_config=True)
    queries, corpus, eval_audit = adapt_toolret_evaluation(query_rows, tool_rows)
    evaluation = benchmark + [{"query":q["query"],"family_id":"toolret:"+q["qid"]} for q in queries]
    rejects = []
    records = _mapped(read_parquet_snapshot("openjev","SFT"), map_openjev_record, "openjev", rejects)
    records += _mapped(read_parquet_snapshot("toolret","train"), map_toolret_record, "toolret", rejects)
    records = list({r["rid"]:r for r in records}.values())
    override_path = ROOT/"data/manifests/audit_overrides.jsonl"
    if override_path.exists():
        overrides = {r["rid"]:r for r in (json.loads(line) for line in override_path.read_text(encoding="utf-8").split("\n") if line)}
        for rec in records:
            override = overrides.get(rec["rid"],{})
            for key in ("null_certified","eligible_negative_ids","template_id","audit_uncertain"):
                if key in override:
                    rec[key] = override[key]
    # Aggregate known ToolRet positives over duplicate query rows before sampling.
    known = {}
    for rec in records:
        if rec.get("is_retrieval_pool"):
            known.setdefault(normalize_text(rec["query"]),set()).add(rec["positive"][0])
    for rec in records:
        if rec.get("is_retrieval_pool"):
            rec["known_positive_ids"] = sorted(known[normalize_text(rec["query"])])
    records = [r for r in records if not r.get("is_retrieval_pool") or len(r["known_positive_ids"])==1]
    from .data.context_audit import TrainingContextAudit
    context_audit=TrainingContextAudit()
    accepted=[]
    for rec in records:
        reason=context_audit.check(rec)
        if reason:
            rejects.append({"source":rec["source"],"rid":rec["rid"],"reason":"shared_context:"+reason})
        else:
            accepted.append(rec)
    records=accepted
    overlap_audit = audit_groups(records,evaluation)
    eval_keys = {key for r in evaluation for key in record_dedup_keys(r)}
    initial = split_records(records,record_dedup_keys,eval_keys=eval_keys)
    by_id = {r["rid"]:r for r in records}
    present = {by_id[r]["bucket"] for r in initial["train"] if not by_id[r].get("audit_uncertain")}
    fallback_used = []
    for short,bucket,mapper in (("choice","openjev_choice",map_arc_record),
                               ("noul","openjev_noul",map_boolq_record),
                               ("score","openjev_score",map_helpsteer2_record)):
        if bucket not in present:
            spec = cfg["fallbacks"][short]
            provenance["fallback_"+short] = verify_snapshot(ROOT/spec["local_dir"],spec["repo_id"])
            fallback = _mapped(read_parquet_snapshot("fallback_"+short,"train"),mapper,"fallback_"+short,rejects)
            for rec in fallback:
                reason=context_audit.check(rec)
                if reason:
                    rejects.append({"source":rec["source"],"rid":rec["rid"],"reason":"shared_context:"+reason})
                else:
                    records.append(rec)
            fallback_used.append(short)
    if fallback_used:
        overlap_audit = audit_groups(records,evaluation)
    manifest,trace,banks = prepare_training_manifest(records,evaluation,num_updates=num_updates,seed=seed)
    # Validate the actual augmented menus before freezing comparisons. Replace
    # rejected presentations within their bucket; every method replays the result.
    by_id={r["rid"]:r for r in records}
    buckets={bucket:[rid for rid in manifest.splits["train"] if by_id[rid]["bucket"]==bucket]
             for bucket in manifest.bucket_counts["train"]}
    rng=random.Random(seed+20000)
    context_replacements=[]
    for batch in trace:
        for item in batch:
            for attempt in range(100):
                rec=dict(by_id[item["rid"]])
                if "options" in item:
                    rec.update(options=item["options"],target=item["target"],is_retrieval_pool=False)
                reason=context_audit.check(rec)
                if reason is None:
                    break
                context_replacements.append({"rid":item["rid"],"reason":reason})
                rid=rng.choice(buckets[item["bucket"]])
                original=by_id[rid]
                ex,aug=materialize_example(_RawRequestEncoder(),original,banks.get(original.get("domain","general"),[]),rng=rng)
                item.clear();item.update(rid=rid,bucket=original["bucket"],augmentation=aug)
                if original["mode"]=="choice":
                    item.update(options=[(c.cid,c.text) for c in ex.candidates],target={REFERENCE_ID:float(ex.target[0]),
                        **{c.cid:float(p) for c,p in zip(ex.candidates,ex.target[1:])}})
            else:
                raise ValueError("No context-feasible presentation after 100 attempts; inspect the shared bucket audit")
    from .data.mixture import trace_stats
    manifest.trace_summary=trace_stats(trace)
    manifest.candidate_k_histogram=dict(Counter(item["augmentation"]["realized_k"] for batch in trace for item in batch))
    manifest.fallbacks_used = fallback_used
    PREPARED.mkdir(parents=True,exist_ok=True)
    for name,obj in (("records",records),("benchmark",benchmark),("toolret_queries",queries),
                     ("toolret_corpus",corpus),("trace",trace),("banks",banks),("manifest",manifest)):
        save_json(PREPARED/(name+".json"),obj)
    audit = {"sources":provenance,"seed":seed,"overlap":overlap_audit,"rejections":rejects,
             "benchmark_duplicate_options":duplicate_cases,
             "toolret_evaluation":eval_audit,"trace_sha256":digest(trace),
             "shared_context_replacements":context_replacements,
             "bank_sha256":digest(banks),"manifest_sha256":digest(clean_json(manifest)),
             "negative_policy":"OpenJev extensions require audited eligible IDs; ToolRet excludes every known positive and duplicates",
             "null_policy":"15% only for explicitly certified records", "environment":environment_record()}
    save_json(PREPARED/"audit.json",audit)
    return {"accepted":len(records),"rejected":len(rejects),"manifest":asdict(manifest)}


def restore_benchmark_states(source_root=None):
    """Rebuild withheld states with official allowlisted adapters, verify hashes.

    source_root follows the pinned sources.json local layout. Files must already
    exist; this function never downloads or uses gold fields to form inputs.
    """
    from .evaluation.official import load_official
    common,_,_ = load_official()
    common.DATA = Path(source_root or ROOT/"data/raw/jevbench_sources")
    common.SOURCES_PATH = ROOT/"data/raw/jevbench/sources.json"
    rows=load_local_dataset("jevbench",str(ROOT/"data"))
    from .data.benchmark_sources import missing_source_plan
    missing=missing_source_plan(common,rows,common.DATA)
    if missing:
        paths="\n".join(item["target"] for item in missing)
        raise FileNotFoundError("Missing original benchmark source files:\n"+paths+
            "\nDownload the pinned sources with: python scripts/prepare_workspace.py benchmark-sources"
            "\nOr add --download-benchmark-sources to prepare --restore-states."
            "\nFor a custom source directory, also pass --source-root <directory>.")
    sources={}
    restored=[]
    for case in rows:
        if case.get("state") is not None and case.get("state") != "":
            continue
        key=case["slice"]
        source=common.parent_key(key)
        if source not in sources:
            sources[source]=common.load_rows(common.source_by_key(source))
        native=common.ADAPTERS[key](sources[source][case["row_index"]],case.get("option_order"))
        state,_,_=common.truncate_state(native["state"])
        if common.content_hash(common.state_json(state)) != case["state_hash"]:
            raise ValueError(f"Restored state hash mismatch for {case['case_id']}; verify source revision")
        restored.append({"case_id":case["case_id"],"state":state})
    path=ROOT/"data/raw/jevbench/restored_states.jsonl"
    # Preserve already restored rows if called on a partial local restoration.
    prior=[json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line] if path.exists() else []
    combined={r["case_id"]:r for r in prior+restored}
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in combined.values()),encoding="utf-8")
    return {"restored":len(restored),"total_local_states":len(combined)}


def _example(tokenizer, record):
    rec = dict(record)
    if rec["mode"] == "choice" and "gold_id" in rec:
        rec["target"] = {rec["gold_id"]:1.0}
    ex,_ = materialize_example(tokenizer, rec, enable_augmentation=False)
    ex.meta.update({k:v for k,v in rec.items() if k in
                   ("state","native_state","question","native_case","native_alias_by_id","case_id","family_id","dataset","duplicate_option_groups")})
    return ex


def run_directory(backbone, method="diffjev", variant="full", seed=0):
    return ROOT/"outputs/formal"/backbone/method/variant/f"seed_{seed}"


def load_runner(backbone, method="diffjev", variant="full", seed=0, trained=True,
                device="cuda", dtype=torch.bfloat16):
    from .baselines.ar_llm import AutoregressiveJevBaseline
    from .baselines.serialized_dlm import SerializedDLMBaseline
    bb = load_backbone(backbone,device=device,dtype=dtype)
    if method == "diffjev":
        cfg,_ = build_ablation_config(None, "full" if variant=="no_pruning" else variant)
        runner = DiffJevModel(bb,cfg)
    elif method == "ar":
        runner = AutoregressiveJevBaseline(bb,max_context_tokens=bb.max_context_tokens)
    elif method == "serialized_dlm":
        runner = SerializedDLMBaseline(bb,max_context_tokens=bb.max_context_tokens)
    else:
        raise ValueError(method)
    apply_freeze_policy(bb)
    if trained:
        checkpoint_variant = "full" if variant=="no_pruning" else variant
        path = run_directory(backbone,method,checkpoint_variant,seed)/"calibrated.pt"
        load_trainable_checkpoint(str(path),runner)
    if variant=="no_pruning":
        runner.cfg.pruning_enabled = False
    runner.eval()
    return runner


def train_run(backbone="llada_8b_base", method="diffjev", variant="full", seed=0,
              resume=None, device="cuda"):
    """Train a fresh independent adaptation, select on dev, calibrate on calib."""
    if variant=="no_pruning":
        raise ValueError("No-pruning is an inference control using the full model checkpoint")
    random.seed(seed);torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    runner = load_runner(backbone,method,variant,seed,trained=False,device=device)
    tokenizer = runner.backbone.tokenizer
    records = {r["rid"]:r for r in read_json(PREPARED/"records.json")}
    manifest,trace = read_json(PREPARED/"manifest.json"),read_json(PREPARED/"trace.json")
    audit = read_json(PREPARED/"audit.json")
    if digest(trace) != audit["trace_sha256"]:
        raise ValueError("Frozen augmentation trace checksum mismatch")
    train_cfg = TrainConfig(seed=seed,optimizer_updates=len(trace))
    if any(len(b)!=train_cfg.effective_batch_size for b in trace):
        raise ValueError("Effective batch must be 32 (8 per source bucket)")
    from .evaluation.sampling import stratified_subset
    dev_records = stratified_subset([records[i] for i in manifest["splits"]["dev"]],256,
                    stratum=lambda r:r["bucket"],identity=lambda r:r["rid"])
    dev = [_example(tokenizer,r) for r in dev_records]
    calib = [_example(tokenizer,records[i]) for i in manifest["splits"]["calib"]]
    if not dev or {x.mode for x in calib} != {"choice","noul","score"}:
        raise ValueError("Separate development and all-mode calibration partitions are required")
    output = run_directory(backbone,method,variant,seed)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/"run_config.json",{"backbone":bb_provenance(runner),"method":method,
        "variant":variant,"train":train_cfg.to_dict(),"diffjev":runner.cfg.to_dict(),
        "trace_sha256":digest(trace),"environment":environment_record(),
        "verification":runner.backbone.verification,"dev_ids":[r["rid"] for r in dev_records]})
    history,summary = train_diffjev(runner,train_cfg,records,trace,tokenizer,
        read_json(PREPARED/"banks.json"),dev_examples=dev,checkpoint_dir=str(output),
        resume_from=resume,enable_augmentation=variant!="no_augmentation")
    load_trainable_checkpoint(str(output/"best.pt"),runner)
    calibration = calibrate_model(runner,calib)
    save_trainable_checkpoint(str(output/"calibrated.pt"),runner,extra_meta={"calibration":calibration,"trace_sha256":digest(trace)})
    save_json(output/"history.json",history)
    save_json(output/"calibration.json",calibration)
    return summary


def bb_provenance(runner):
    return getattr(getattr(runner,"backbone",None),"provenance",{})


def baseline_runner(name,device="cuda",allow_api=False):
    from .baselines.external import JevAPIBaseline,JevaBaseline
    from .baselines.retrieval import DenseBiEncoderBaseline
    if name=="jev_api":
        if not allow_api:
            raise ValueError("API evaluation requires explicit allow_api=True")
        return JevAPIBaseline()
    if name=="llm_as_jev":
        runner=load_runner("llm_as_jev_qwen","ar",trained=False,device=device)
        runner.requires_grad_(False)
        runner.reference_only=True
        return runner
    key = "jeva_4b" if name=="jeva" else "bge_large_en" if name=="bge" else "e5_large_v2"
    spec = model_specs()[key]
    path = ROOT/spec["local_dir"]
    verify_snapshot(path,spec["repo_id"])
    if name=="jeva":
        return JevaBaseline(str(path),device=device)
    return DenseBiEncoderBaseline(local_dir=str(path),device=device,family=name)


class _RawRequestEncoder:
    # External transports consume raw text; tensor fields intentionally stay empty.
    def encode(self,text,**kwargs):
        return []


def benchmark_cases(tokenizer):
    from .evaluation.exp1_benchmark import JevBenchCase
    cases=[]
    for rec in read_json(PREPARED/"benchmark.json"):
        ex=_example(tokenizer,rec)
        score = float(rec["native_case"]["gold"]) if ex.mode=="score" else None
        cases.append(JevBenchCase(rec["case_id"],rec["family_id"],rec["dataset"],ex,
                     gold_id=ex.gold_id if ex.mode!="score" else None,gold_score=score))
    return cases


def evaluate_benchmark(runner, output, robustness=True):
    from .preprocessing import runner_tokenizer
    from .evaluation.exp1_benchmark import run_exp1_benchmark
    from .evaluation.exp2_robustness import run_exp2_robustness
    cases=benchmark_cases(runner_tokenizer(runner) or _RawRequestEncoder())
    output=Path(output)
    report=run_exp1_benchmark(output.name,runner,cases,n_boot=2000,official_output_dir=str(output))
    save_json(output/"exp1.json",report)
    if robustness:
        families={}
        for c in sorted(cases,key=lambda c:c.case_id):
            if c.example.mode=="choice":
                families.setdefault(c.family_id,c.example)
        from .evaluation.sampling import stratified_subset
        selected = stratified_subset(list(families.items()),1000,stratum=lambda x:x[1].meta["dataset"],identity=lambda x:x[0])
        robust=run_exp2_robustness(output.name,runner,selected,n_boot=2000)
        save_json(output/"exp2.json",robust)
    return report


def build_scaling_manifest(seed=0,num_queries=512):
    """Freeze ID/text pools once; no model tokenizer influences BM25 ranking."""
    from .evaluation.exp3_scaling import build_nested_scaling_manifests
    corpus=read_json(PREPARED/"toolret_corpus.json")
    queries=read_json(PREPARED/"toolret_queries.json")
    from .evaluation.sampling import stratified_subset
    queries=stratified_subset(queries,num_queries,seed,stratum=lambda q:q["domain"],identity=lambda q:q["qid"])
    candidates={c["cid"]:Candidate(c["cid"],[],text=c["text"]) for c in corpus}
    specs=[]
    # Match the official category-specific retrieval corpus for every query.
    for domain in sorted({q["domain"] for q in queries}):
        rows=[dict(q,query_ids=[],query_text=q["query"],positive=candidates[q["positive_id"]])
              for q in queries if q["domain"]==domain]
        pool=[candidates[c["cid"]] for c in corpus if domain in c.get("domains",[c["domain"]])]
        specs+=build_nested_scaling_manifests(rows,pool,max_k=2048,num_queries=len(rows),seed=seed)
    from .evaluation.sampling import stratified_subset
    specs=stratified_subset(specs,num_queries,seed)
    catalog={c["cid"]:c["text"] for c in corpus}
    manifest={"seed":seed,"ks":[4,16,64,256,1024,2048],"retriever":{"name":"BM25","k1":1.5,"b":.75,"tokens":"lowercase whitespace"},
              "corpus_sha256":digest(catalog),"candidate_text":catalog,"queries":[{
              "qid":s.qid,"query":s.query_text,"domain":s.domain,"positive_id":s.positive.cid,
              "random_ids":[c.cid for c in s.random_negatives],"hard_ids":[c.cid for c in s.hard_negatives]} for s in specs]}
    save_json(PREPARED/"scaling.json",manifest)
    return {"queries":len(specs),"candidate_count":len(catalog),"sha256":digest(manifest)}


def scaling_specs(tokenizer=None):
    from .evaluation.exp3_scaling import ScalingQuerySpec
    manifest=read_json(PREPARED/"scaling.json")
    if digest(manifest["candidate_text"])!=manifest["corpus_sha256"]:
        raise ValueError("Scaling corpus checksum mismatch")
    cache={}
    def candidate(cid):
        if cid in cache:
            return cache[cid]
        text=manifest["candidate_text"][cid]
        cache[cid]=Candidate(cid,_enc(tokenizer,text) if tokenizer else [],text=text)
        return cache[cid]
    return [ScalingQuerySpec(q["qid"],_enc(tokenizer,q["query"]) if tokenizer else [],candidate(q["positive_id"]),
            [candidate(i) for i in q["random_ids"]],[candidate(i) for i in q["hard_ids"]],q["domain"],q["query"])
            for q in manifest["queries"]]


def evaluate_scaling(runner,output):
    from .evaluation.exp3_scaling import run_exp3_scaling
    from .preprocessing import runner_tokenizer
    specs=scaling_specs(runner_tokenizer(runner))
    reports={pool:run_exp3_scaling(Path(output).name,runner,specs,pool=pool,n_boot=2000) for pool in ("random","hard")}
    save_json(Path(output)/"exp3.json",reports)
    return reports


def evaluate_ablation(variant,seed=0,device="cuda"):
    runner=load_runner("llada_8b_base",variant=variant,seed=seed,device=device)
    output=run_directory("llada_8b_base",variant=variant,seed=seed)/"evaluation"
    benchmark=evaluate_benchmark(runner,output,robustness=True)
    scaling=evaluate_scaling(runner,output)
    row={"variant":variant,"benchmark":benchmark.aggregate,
         "scaling":scaling["hard"].by_k,"enabled_loss_terms":runner.cfg.enabled_loss_terms(),
         "training_summary":read_json(run_directory("llada_8b_base",variant="full" if variant=="no_pruning" else variant,seed=seed)/"train_summary.json")}
    save_json(output/"exp4.json",row)
    return row


def evaluate_efficiency(backbone,seed=0,device="cuda",attribution=False):
    from .evaluation.exp5_efficiency import run_exp5_efficiency_panel
    env=environment_record()
    if len(env["devices"])!=1 or "A100" not in env["devices"][0]["name"] or env["devices"][0]["total_memory"]<75*2**30:
        raise ValueError("Formal Exp 5 requires exactly one visible A100 80GB; set CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Formal Exp 5 requires bf16 support")
    methods={"diffjev":lambda:load_runner(backbone,seed=seed,device=device),
             "serialized_dlm":lambda:load_runner(backbone,"serialized_dlm",seed=seed,device=device)}
    if backbone.startswith("nemotron"):
        methods["serialized_ar"]=lambda:load_runner(backbone,"ar",seed=seed,device=device)
    if attribution:
        if backbone!="llada_8b_base":
            raise ValueError("Attribution controls are restricted to the LLaDA anchor")
        methods.update(full_context_t1_control=lambda:load_runner(backbone,variant="full_context_t1_control",seed=seed,device=device),
                       no_pruning=lambda:load_runner(backbone,variant="no_pruning",seed=seed,device=device))
    report=run_exp5_efficiency_panel(backbone,methods,scaling_specs(),n_boot=2000)
    report["environment"]=env
    output=ROOT/"outputs/formal/efficiency"/backbone
    save_json(output/"exp5.json",report)
    plot_efficiency(report,output)
    return report


def plot_efficiency(report,output):
    import matplotlib.pyplot as plt
    output=Path(output)
    for metric,label in (("wall_p50_ms","Full request p50 (ms)"),("mean_gflops","Executed-shape GFLOPs")):
        fig,ax=plt.subplots(figsize=(6,4))
        for method in sorted({r["method"] for r in report["table"]}):
            rows=[r for r in report["table"] if r["method"]==method and r["coverage"]>0]
            ax.plot([r["K"] for r in rows],[r[metric] for r in rows],"o-",label=method)
        ax.set(xscale="log",xlabel="Candidates K",ylabel=label,title=report["size"])
        ax.legend();fig.tight_layout();fig.savefig(output/(metric+".png"),dpi=180);plt.close(fig)


def collect_results():
    """Build compact tables/panels exclusively from saved measured artifacts."""
    import csv
    root=ROOT/"outputs/formal"
    tables={"exp1":[],"exp4":[],"exp5":[]}
    for path in sorted(root.rglob("exp1.json")):
        report=read_json(path)
        for dataset,entry in report["aggregate"].get("official",{}).get("slices",{}).items():
            headline=entry.get("headline",{})
            tables["exp1"].append({"run":str(path.parent.relative_to(root)),"dataset":dataset,
                "metric":headline.get("metric"),"value":headline.get("value"),"coverage":entry["coverage"]})
    for path in sorted(root.rglob("exp4.json")):
        report=read_json(path)
        summary=report["training_summary"]
        row={"variant":report["variant"],"benchmark_accuracy":report["benchmark"].get("accuracy"),
            "benchmark_brier":report["benchmark"].get("brier"),"processed_tokens":summary["processed_tokens"],
            "optimizer_updates":summary["optimizer_updates"],"training_wall_seconds":summary.get("training_wall_seconds"),
            "trainable_parameters":summary["parameter_counts"]["trainable"]}
        robust_path=path.with_name("exp2.json")
        if robust_path.exists():
            robust=read_json(robust_path)
            row.update(order_flip_rate=robust["order_permutation"]["decision_flip_rate"],
                       alias_flip_rate=robust["alias_renaming"]["decision_flip_rate"])
        for k,metrics in report["scaling"].items():
            row[f"hard_K{k}_hit_at_1"]=metrics["hit_at_1"]
            row[f"hard_K{k}_coverage"]=metrics["coverage"]
        row["K2048_gold_survival"]=report["scaling"].get("2048",{}).get("stagewise_gold_survival")
        tables["exp4"].append(row)
    panels=[]
    for path in sorted((root/"efficiency").glob("*/exp5.json")):
        report=read_json(path);panels.append(report);tables["exp5"].extend(report["table"])
    output=root/"tables"
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        if not rows:
            continue
        with (output/(name+".csv")).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader();writer.writerows(rows)
    if panels:
        import matplotlib.pyplot as plt
        for metric,label in (("wall_p50_ms","Full request p50 (ms)"),("mean_gflops","Executed-shape GFLOPs")):
            fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),squeeze=False)
            for ax,panel in zip(axes[0],panels):
                for method in sorted({r["method"] for r in panel["table"]}):
                    rows=[r for r in panel["table"] if r["method"]==method and r["coverage"]>0]
                    ax.plot([r["K"] for r in rows],[r[metric] for r in rows],"o-",label=method)
                ax.set(xscale="log",xlabel="K",ylabel=label,title=panel["size"]);ax.legend()
            fig.tight_layout();fig.savefig(output/(metric+"_size_panel.png"),dpi=180);plt.close(fig)
    save_json(output/"available_results.json",{name:len(rows) for name,rows in tables.items()})
    return {name:len(rows) for name,rows in tables.items()}
