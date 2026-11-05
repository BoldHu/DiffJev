"""Unit tests for real-backbone adapters (LLaDA, Nemotron), source mappers,
calibration, formal training loop, baselines, and Experiments 1-5."""

from __future__ import annotations

import os
import random
from types import SimpleNamespace

import torch
from torch import nn

from diffjev.backbones import LLaDAAdapter, NemotronDiffusionAdapter, default_mock_tokenizer
from diffjev.backbones.mock import RMSNorm
from diffjev.baselines import (
    AutoregressiveJevBaseline,
    BM25Retriever,
    DenseBiEncoderBaseline,
    JevAPIBaseline,
    JevaBaseline,
    SerializedDLMBaseline,
)
from diffjev.calibration import calibrate_model
from diffjev.config import DiffJevConfig, TrainConfig
from diffjev.data import (
    map_arc_record,
    map_boolq_record,
    map_helpsteer2_record,
    map_jevbench_record,
    map_openjev_record,
    map_toolret_record,
    materialize_example,
    prepare_training_manifest,
)
from diffjev.data.synthetic import synthetic_choice, synthetic_noul, synthetic_score
from diffjev.evaluation import (
    JevBenchCase,
    build_nested_scaling_manifests,
    run_exp1_benchmark,
    run_exp2_robustness,
    run_exp3_scaling,
    run_exp4_ablation_suite,
    run_exp5_efficiency_panel,
)
from diffjev.model import DiffJevModel
from diffjev.packing import Candidate
from diffjev.tasks import REFERENCE_ID
from diffjev.training import (
    apply_freeze_policy,
    load_trainable_checkpoint,
    save_trainable_checkpoint,
    train_diffjev,
)
from tests.conftest import make_backbone, make_model


class _FakeLLaDABlock(nn.Module):
    def __init__(self, d: int = 16, n_heads: int = 4, n_kv: int = 2, ffn: int = 32):
        super().__init__()
        dh = d // n_heads
        self.num_heads = n_heads
        self.num_key_value_heads = n_kv
        self.attn_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, n_heads * dh, bias=False)
        self.k_proj = nn.Linear(d, n_kv * dh, bias=False)
        self.v_proj = nn.Linear(d, n_kv * dh, bias=False)
        self.attn_out = nn.Linear(n_heads * dh, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.ff_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.ff_out = nn.Linear(ffn, d, bias=False)
        self.act = nn.SiLU()


class _FakeLLaDAModel(nn.Module):
    def __init__(self, vocab_size: int = 64, d: int = 16, n_layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=10000.0,
            mask_token_id=1,
            pad_token_id=0,
        )
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.wte = nn.Embedding(vocab_size, d)
        self.model.transformer.blocks = nn.ModuleList(
            [_FakeLLaDABlock(d=d) for _ in range(n_layers)]
        )
        self.model.transformer.ln_f = RMSNorm(d)
        self.model.transformer.ff_out = nn.Linear(d, vocab_size, bias=False)


class _FakeNemotronBlock(nn.Module):
    def __init__(self, d: int = 16, n_heads: int = 4, n_kv: int = 2, ffn: int = 32):
        super().__init__()
        dh = d // n_heads
        self.input_layernorm = RMSNorm(d)
        self.self_attn = nn.Module()
        self.self_attn.num_heads = n_heads
        self.self_attn.num_key_value_heads = n_kv
        self.self_attn.q_proj = nn.Linear(d, n_heads * dh, bias=False)
        self.self_attn.k_proj = nn.Linear(d, n_kv * dh, bias=False)
        self.self_attn.v_proj = nn.Linear(d, n_kv * dh, bias=False)
        self.self_attn.o_proj = nn.Linear(n_heads * dh, d, bias=False)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, ffn, bias=False)
        self.mlp.up_proj = nn.Linear(d, ffn, bias=False)
        self.mlp.down_proj = nn.Linear(ffn, d, bias=False)


class _FakeNemotronModel(nn.Module):
    def __init__(self, vocab_size: int = 64, d: int = 16, n_layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=10000.0,
            attn_scale_base=512.0,
            mask_token_id=1,
            pad_token_id=0,
        )
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(vocab_size, d)
        self.encoder.layers = nn.ModuleList([_FakeNemotronBlock(d=d) for _ in range(n_layers)])
        self.encoder.norm = RMSNorm(d)
        self.diffusion_head = nn.Linear(d, vocab_size, bias=True)
        # Attach a dummy LoRA attribute to verify NemotronDiffusionAdapter strips it
        self.self_spec_lora = nn.Linear(d, 4)


def test_llada_and_nemotron_adapters_verify_and_run(tok):
    llada_hf = _FakeLLaDAModel(vocab_size=tok.vocab_size).to(torch.float64)
    llada = LLaDAAdapter(llada_hf, tok)
    rep_l = llada.verify_adapter()
    assert all(rep_l.values())

    nem_hf = _FakeNemotronModel(vocab_size=tok.vocab_size).to(torch.float64)
    nem = NemotronDiffusionAdapter(nem_hf, tok, size="8B")
    rep_n = nem.verify_adapter()
    assert all(rep_n.values())

    # End-to-end DiffJev prediction & equivariance on both adapters
    ex = synthetic_choice(tok, random.Random(0), 4)
    for bb in (llada, nem):
        m = DiffJevModel(bb, DiffJevConfig())
        p1 = m.predict(ex).prob_by_id()
        p2 = m.predict(ex.permuted([3, 1, 2, 0])).prob_by_id()
        for k in p1:
            assert abs(p1[k] - p2[k]) < 1e-9


def test_source_mappers_and_manifest_preparation(tok):
    # OpenJev: filters K > 32 without cropping; merges audited null; preserves soft targets
    assert list(
        map_openjev_record(
            {"query": "q", "mode": "choice", "options": [f"o{i}" for i in range(33)], "answer": 0},
            max_train_k=32,
        )
    ) == []
    c_rec = next(
        map_openjev_record(
            {
                "id": "c1",
                "mode": "choice",
                "query": "choose w1",
                "options": ["option w1", "No way option", "None of the above"],
                "soft_target": [0.2, 0.3, 0.5],
            }
        )
    )
    # "No way option" is NOT treated as null; "None of the above" IS merged into reference
    assert len(c_rec["options"]) == 2
    assert abs(c_rec["target"][REFERENCE_ID] - 0.5) < 1e-6

    # ToolRet: uses original query, deduplicates tools, drops multi-positive records
    assert (
        list(
            map_toolret_record(
                {
                    "query": "q",
                    "positive_tools": [{"id": "a", "description": "x"}, {"id": "b", "description": "y"}],
                    "negative_tools": [{"id": "c", "description": "z"}],
                }
            )
        )
        == []
    )
    t_rec = next(
        map_toolret_record(
            {
                "id": "t1",
                "query": "choose w2",
                "prompt": "IGNORED GENERATED PROMPT",
                "positive_tools": [{"id": "p", "name": "w2", "description": "option w2"}],
                "negative_tools": [{"id": f"n{i}", "name": f"w{i+3}", "description": f"option w{i+3}"} for i in range(5)],
            }
        )
    )
    assert t_rec["query"] == "choose w2"

    # Fallbacks & JevBench
    assert len(list(map_arc_record({"id": "a1", "question": "q", "choices": {"text": ["x", "y"], "label": ["A", "B"]}, "answerKey": "B"}))) == 1
    assert len(list(map_boolq_record({"passage": "p", "question": "q", "answer": True}))) == 1
    assert len(list(map_helpsteer2_record({"prompt": "p", "response": "r", "helpfulness": 3}))) == 1
    jb = map_jevbench_record({"id": "jb1", "dataset": "sst5", "query": "movie review", "label": 4})
    assert jb is not None and jb["mode"] == "score"


def test_formal_train_loop_checkpoint_and_calibration(tok, tmp_path):
    recs = []
    for i in range(16):
        recs.extend(map_openjev_record({"id": f"c{i}", "mode": "choice", "query": f"choose w{i%8}", "options": [f"option w{i%8}", f"option w{(i+1)%8}"], "answer": 0}))
        recs.extend(map_openjev_record({"id": f"n{i}", "mode": "noul", "query": f"query w{i%8}", "proposition": f"option w{i%8}", "label": "Yes"}))
        recs.extend(map_openjev_record({"id": f"s{i}", "mode": "score", "rubric": "rubric", "query": f"query w{i%8}", "score_values": [0, 1, 2], "bins": ["bin w0", "bin w1", "bin w2"], "label": 1}))
        recs.extend(map_toolret_record({"id": f"t{i}", "query": f"choose w{i%8}", "positive_tools": [{"id": f"p{i}", "name": f"w{i%8}", "description": f"option w{i%8}"}], "negative_tools": [{"id": f"ng{i}_{j}", "name": f"w{(i+j+1)%8}", "description": f"option w{(i+j+1)%8}"} for j in range(3)]}))

    manifest, trace, banks = prepare_training_manifest(recs, num_updates=3, per_bucket_per_batch=2, seed=0)
    rec_by_id = {r["rid"]: r for r in recs}
    dev_exs = [materialize_example(tok, rec_by_id[rid], enable_augmentation=False)[0] for rid in manifest.splits["train"][:4]]

    bb = make_backbone(tok, dtype=torch.float32)
    model = DiffJevModel(bb, DiffJevConfig())
    ckpt_dir = str(tmp_path / "ckpts")
    hist, summary = train_diffjev(
        model,
        TrainConfig(effective_batch_size=8, optimizer_updates=3, learning_rate=1e-3, bf16=False),
        rec_by_id,
        trace,
        tok,
        banks_by_domain=banks,
        dev_examples=dev_exs,
        checkpoint_dir=ckpt_dir,
        save_every=2,
        eval_every=2,
    )
    assert len(hist) == 3 and summary["total_presentations"] == 24
    assert os.path.isfile(os.path.join(ckpt_dir, "step_3.pt"))

    # Load checkpoint and run calibration
    load_trainable_checkpoint(os.path.join(ckpt_dir, "step_3.pt"), model)
    cal = calibrate_model(model, dev_exs)
    assert "choice" in cal and model.cfg.readout_temperature["choice"] > 0


def test_baselines_and_experiments_end_to_end(tok):
    bb = make_backbone(tok, dtype=torch.float64)
    apply_freeze_policy(bb, 4)
    model = DiffJevModel(bb, DiffJevConfig())

    rng = random.Random(0)
    ex_c = synthetic_choice(tok, rng, K=6, forced_choice=True)
    ser = SerializedDLMBaseline(bb)
    ar = AutoregressiveJevBaseline(bb, tok)
    bm25 = BM25Retriever()
    bienc = DenseBiEncoderBaseline(embed_fn=bb.embed)
    api = JevAPIBaseline(
        tokenizer=tok,
        transport=lambda p: {"answers": {"decision": {"type":"choice",
            "choice":next(iter(p["questions"]["decision"]["criteria"])),
            "probabilities": {key:1.0/len(p["questions"]["decision"]["criteria"])
                              for key in p["questions"]["decision"]["criteria"]}}}},
    )
    assert JevaBaseline().predict(ex_c).failure_reason == "checkpoint_unavailable"

    for b in (ser, ar, bm25, bienc, api):
        res = b.predict(ex_c)
        assert not res.failed and res.top1 in [c.cid for c in ex_c.candidates]
    # Verify AR top-1 fast mode and training losses
    assert not ar.predict_top1(ex_c).failed
    assert torch.isfinite(ar.training_losses(ex_c)["total"])
    assert torch.isfinite(ser.training_losses(ex_c)["total"])

    # Run Exp 1, 2, 3, 4, 5 on tiny synthetic suites
    cases = [
        JevBenchCase("c1", "f1", "d1", ex_c, gold_id=ex_c.gold_id),
        JevBenchCase("c2", "f2", "sst5", synthetic_score(tok, rng, M=5), gold_score=2.0),
        JevBenchCase("c3", "f3", "noul_ds", synthetic_noul(tok, rng), gold_id="yes"),
    ]
    r1 = run_exp1_benchmark("diffjev", model, cases, n_boot=10)
    assert r1.num_cases == 3 and "sst5" in r1.by_dataset

    r2 = run_exp2_robustness("diffjev", model, [("f1", ex_c)], n_boot=10)
    assert r2.order_permutation["decision_flip_rate"] == 0.0
    assert r2.alias_renaming["decision_flip_rate"] == 0.0

    corpus = [Candidate(f"t_{i:03d}", tok.encode(f"option w{i % 48}")) for i in range(50)]
    q_raw = [{"qid": "q0", "query_ids": tok.encode("choose w1"), "positive": corpus[1]}]
    specs = build_nested_scaling_manifests(q_raw, corpus, max_k=40, num_queries=1)
    r3 = run_exp3_scaling("diffjev", model, specs, ks=(4, 36), pool="hard", warm_lower_cache=True, run_no_answer_probe=True, n_boot=10)
    assert 36 in r3.by_k and r3.no_answer_probe is not None

    table4 = run_exp4_ablation_suite(
        lambda cfg, sp: (DiffJevModel(bb, cfg), {"estimated_train_tflops": 0.01}),
        cases[:1],
        [("f1", ex_c)],
        specs,
        scaling_ks=(4, 36),
        variants=("full", "full_context_t1_control", "no_pruning"),
        n_boot=5,
    )
    assert len(table4) == 3

    p5 = run_exp5_efficiency_panel(
        "mock",
        {"serialized_ar": ar, "serialized_dlm": ser, "diffjev_t3_pruning": model},
        specs,
        ks=(4, 36),
        num_queries=1,
        warmups=1,
        repetitions=1,
        n_boot=5,
    )
    assert len(p5["table"]) == 6


def test_cloud_logger_tables_and_academic_figures(tok, tmp_path):
    from diffjev.utils import ExperimentLogger, mount_cloud_storage
    from diffjev.visualization import (
        plot_data_preparation_summary,
        plot_exp1_benchmark,
        plot_exp2_robustness,
        plot_exp3_scaling,
        plot_exp4_ablations,
        plot_exp5_efficiency,
        plot_training_and_calibration,
        render_and_save_table,
    )

    cloud = mount_cloud_storage(drive_mount_point=str(tmp_path / "drive"))
    assert "resolved_root" in cloud
    test_paths = {
        k: tmp_path / "cloud_root" / k
        for k in ("models", "raw", "processed", "manifests", "checkpoints", "results", "tables", "figures", "logs")
    }
    test_paths["root"] = tmp_path / "cloud_root"
    logger = ExperimentLogger("test_stage", paths=test_paths)
    logger.info("hello", step=1)
    jpath = logger.save_json("sample.json", {"ok": True})
    assert os.path.isfile(jpath)

    tbl = render_and_save_table(
        logger,
        "unit_table",
        [{"model": "DiffJev", "acc": 0.9123}],
        columns=[("model", "Model"), ("acc", "Accuracy")],
        caption="Unit test table.",
    )
    assert os.path.isfile(tbl["csv"]) and os.path.isfile(tbl["latex"])

    bb = make_backbone(tok, dtype=torch.float64)
    apply_freeze_policy(bb, 4)
    model = DiffJevModel(bb, DiffJevConfig())
    rng = random.Random(0)
    ex_c = synthetic_choice(tok, rng, K=6, forced_choice=True)
    cases = [JevBenchCase("c1", "f1", "d1", ex_c, gold_id=ex_c.gold_id)]
    r1 = run_exp1_benchmark("diffjev", model, cases, n_boot=5)
    r2 = run_exp2_robustness("diffjev", model, [("f1", ex_c)], n_boot=5)
    corpus = [Candidate(f"t_{i:03d}", tok.encode(f"option w{i % 48}")) for i in range(40)]
    q_raw = [{"qid": "q0", "query_ids": tok.encode("choose w1"), "positive": corpus[1]}]
    specs = build_nested_scaling_manifests(q_raw, corpus, max_k=36, num_queries=1)
    r3 = run_exp3_scaling("diffjev", model, specs, ks=(4, 36), pool="hard", n_boot=5)

    f0 = plot_data_preparation_summary(
        {
            "bucket_counts": {"train": {"openjev_choice": 8, "openjev_noul": 8, "openjev_score": 8, "toolret_choice": 8}},
            "candidate_k_histogram": {2: 4, 4: 4, 8: 2},
        },
        out_dir=tmp_path / "figs",
        show=False,
    )
    f1 = plot_training_and_calibration(
        [{"step": 0, "total": 1.2, "L_mask": 0.5, "L_task_CE": 0.6, "L_Brier": 0.1}],
        {"choice": {"nll_before": 0.8, "nll_after": 0.7, "temperature": 1.1}},
        out_dir=tmp_path / "figs",
        show=False,
    )
    f2 = plot_exp1_benchmark([r1], out_dir=tmp_path / "figs", show=False)
    f3 = plot_exp2_robustness([r2], out_dir=tmp_path / "figs", show=False)
    f4 = plot_exp3_scaling([r3], [r3], out_dir=tmp_path / "figs", show=False)
    f5 = plot_exp4_ablations(
        [
            {"variant": "full", "benchmark_accuracy": 0.85, "order_flip_rate": 0.0, "infer_gflops_at_kmax": 10.0},
            {"variant": "no_pruning", "benchmark_accuracy": 0.86, "order_flip_rate": 0.0, "infer_gflops_at_kmax": 25.0},
        ],
        out_dir=tmp_path / "figs",
        show=False,
    )
    f6 = plot_exp5_efficiency(
        {
            "size": "mock",
            "table": [
                {"method": "diffjev_t3_pruning", "K": 4, "coverage": 1.0, "wall_p50_ms": 5.0, "wall_p95_ms": 6.0, "mean_gflops": 10.0, "peak_memory_mb": 120.0, "speedup_vs_serialized_dlm": 1.2},
                {"method": "diffjev_t3_pruning", "K": 36, "coverage": 1.0, "wall_p50_ms": 15.0, "wall_p95_ms": 18.0, "mean_gflops": 45.0, "peak_memory_mb": 160.0, "speedup_vs_serialized_dlm": 2.3},
            ],
        },
        out_dir=tmp_path / "figs",
        show=False,
    )
    for fig in (f0, f1, f2, f3, f4, f5, f6):
        assert os.path.isfile(fig["pdf"]) and os.path.isfile(fig["png"])


