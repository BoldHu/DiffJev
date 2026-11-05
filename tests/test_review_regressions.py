"""Small regression fixtures for the methodological review; no downloads."""
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffjev.backbones import LLaDAAdapter, NemotronDiffusionAdapter
from diffjev.data.native import semantic_id
from diffjev.data.sources import map_openjev_record, map_toolret_record, map_jevbench_record, materialize_example, prepare_training_manifest
from diffjev.data.augmentation import augment_menu
from diffjev.evaluation.exp2_robustness import apply_alias_scheme
from diffjev.evaluation.exp3_scaling import ScalingQuerySpec
from diffjev.evaluation.exp5_efficiency import _measure_single_request
from diffjev.evaluation.operations import OperationCounter
from diffjev.tasks import build_choice, REFERENCE_ID
from diffjev.config import TrainConfig
from diffjev.training import train_diffjev
from tests.conftest import make_model
from tests.test_adapters_baselines_experiments import _FakeLLaDAModel, _FakeNemotronModel


def test_llada_native_ffn_order(tok):
    native = _FakeLLaDAModel(tok.vocab_size).double()
    adapter = LLaDAAdapter(native,tok)
    block = adapter.layer_modules()[0]
    block.attn_out.weight.data.zero_()
    x = torch.randn(1,3,16,dtype=torch.float64)
    pos = torch.arange(3)[None]
    k,v = adapter.layer_kv(0,x,pos)
    actual = adapter.layer_forward(0,x,pos,k,v,torch.ones(1,3,3,dtype=torch.bool))
    expected = x+block.ff_out(block.act(block.ff_proj(block.ff_norm(x)))*block.up_proj(block.ff_norm(x)))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_nemotron_reuses_native_rotary_amplitude_and_semantic_scaling(tok):
    native = _FakeNemotronModel(tok.vocab_size).double()
    native.config.rope_parameters={"rope_type":"yarn","llama_4_scaling_beta":.1,
                                   "original_max_position_embeddings":16}
    class Rotary(nn.Module):
        def forward(self,x,positions):
            self.positions=positions.clone()
            angle=positions[...,None].double().expand(-1,-1,4)/17
            return angle.cos()*1.23,angle.sin()*1.23
    native.encoder.rotary_emb=Rotary()
    adapter=NemotronDiffusionAdapter(native,tok,size="3B")
    parts=adapter._inspect_block(0)
    pos=torch.tensor([[0,15,16,32,32]])
    x=torch.ones(1,4,5,4,dtype=torch.float64)
    actual=adapter._rope(x,pos,parts)
    cos,sin=native.encoder.rotary_emb(x,pos)
    expected=x*cos[:,None]+torch.cat((-x[...,2:],x[...,:2]),-1)*sin[:,None]
    torch.testing.assert_close(actual,expected)
    scaled=adapter._apply_semantic_query_scaling(parts,x,pos)
    expected_scale=1+.1*torch.log1p(torch.floor(pos.float()/16))
    torch.testing.assert_close(scaled,x*expected_scale[:,None,:,None].double())
    torch.testing.assert_close(actual[:,:,3],actual[:,:,4])


def _native_openjev(task="choice"):
    return {"state":"A complete problem context.","question":"Decide according to the supplied conditions.",
        "task":task,"category":"test","options":[{"id":"o0","name":"First","criteria":"Must have x > 0"},
        {"id":"o1","name":"Second","criteria":"Must have x < 0"}],
        "score_values":None,"answer":{"kind":"soft","probabilities":[.25,.75]}}


def test_native_openjev_preserves_context_soft_mass_and_numeric_bins(tok):
    raw=_native_openjev()
    rec=list(map_openjev_record(raw))[0]
    ex,_=materialize_example(tok,rec,enable_augmentation=False)
    assert raw["state"] in ex.meta["query_text"]
    assert "x > 0" in ex.candidates[0].text
    torch.testing.assert_close(ex.target,torch.tensor([0,.25,.75],dtype=torch.float64))
    raw["task"]="score";raw["score_values"]=[4,0]
    rec=list(map_openjev_record(raw))[0]
    ex,_=materialize_example(tok,rec,enable_augmentation=False)
    assert ex.bin_values==[0.,4.]
    torch.testing.assert_close(ex.target,torch.tensor([.75,.25],dtype=torch.float64))
    assert "x < 0" in ex.meta["query_text"] and "x > 0" in ex.meta["query_text"]
    raw["split"]="RL"
    assert list(map_openjev_record(raw))==[]


def test_native_noul_polarity_and_toolret_json_schema(tok):
    raw=_native_openjev("noul")
    raw["options"][0]["name"]="No";raw["options"][1]["name"]="Yes"
    rec=list(map_openjev_record(raw))[0]
    assert rec["p_yes"]==.75
    tool=lambda name,expr:json.dumps({"name":name,"description":expr,"parameters":{"x":{"type":"number"}}})
    raw={"id":"train_1","query":"The original query","prompt":"Do not feed this prompt",
         "positive":[tool("calculate","x+1")],"negative":[tool("calculate","x-1"),tool("calculate","x-1")]}
    rec=list(map_toolret_record(raw))[0]
    assert rec["query"]==raw["query"] and len(rec["negatives"])==1
    assert rec["positive"][0]!=rec["negatives"][0][0]
    assert semantic_id("x+1")!=semantic_id("x-1")


def test_native_benchmark_alias_remapping_and_withheld_state():
    state="A clinical case"
    raw={"case_id":"one","family_id":"family","slice":"medqa_usmle","task_type":"choice",
         "question":{"type":"choice","instructions":"Choose","criteria":{"a":"Treatment one","b":"Treatment two"}},
         "state":state,"state_hash":hashlib.sha256(json.dumps(state,separators=(",", ":")).encode()).hexdigest()[:16],"gold":"b"}
    first=map_jevbench_record(raw)
    second=map_jevbench_record(dict(raw,question={**raw["question"],"criteria":{"a":"Treatment two","b":"Treatment one"}},gold="a"))
    assert first["gold_id"]==second["gold_id"]
    raw.pop("state")
    with pytest.raises(ValueError,match="withheld"):
        map_jevbench_record(raw)


def test_unique_augmentation_large_aliases_and_scaling_k(tok):
    aug=augment_menu(["p"],"p",["n","n","m"],random.Random(1),proposal={4:1},null_rate=0)
    assert len(aug.candidate_ids)==len(set(aug.candidate_ids))==3
    ex=build_choice(tok,"query",[(str(i),"option "+str(i)) for i in range(77)],{ "0":1},forced_choice=True)
    for scheme in ("alpha_upper","alpha_reverse","numeric_bracket"):
        changed=apply_alias_scheme(ex,scheme)
        assert len({c.alias for c in changed.candidates})==77
        assert [(c.cid,c.text) for c in changed.candidates]==[(c.cid,c.text) for c in ex.candidates]
    spec=ScalingQuerySpec("q",ex.query_ids,ex.candidates[0],ex.candidates[1:4],ex.candidates[1:4])
    with pytest.raises(ValueError,match="requested K=16"):
        spec.build_example(16)


def test_scaling_manifest_replays_complete_content(tok,tmp_path):
    from diffjev.evaluation.exp3_scaling import save_scaling_manifest,load_scaling_manifest
    ex=build_choice(tok,"query",[("a","w1"),("b","w2")],{"a":1},forced_choice=True)
    spec=ScalingQuerySpec("q",ex.query_ids,ex.candidates[0],ex.candidates[1:],ex.candidates[1:],query_text="query")
    path=tmp_path/"manifest.json"
    save_scaling_manifest([spec],str(path))
    assert load_scaling_manifest(str(path)) == [spec]
    record=json.loads(path.read_text())
    record["candidates"]["a"]["text"]="changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="checksum"):
        load_scaling_manifest(str(path))


def test_failed_robustness_is_unavailable_not_zero(tok):
    from diffjev.baselines.base import BaselineAdapter,BaselineResult
    from diffjev.evaluation.exp2_robustness import run_exp2_robustness
    class Unavailable(BaselineAdapter):
        def predict(self,example,**kwargs):
            return BaselineResult("unavailable",example.mode,[],None,failed=True,failure_reason="context_limit")
    ex=build_choice(tok,"query",[("a","w1"),("b","w2")],{"a":1},forced_choice=True)
    report=run_exp2_robustness("unavailable",Unavailable(),[("q",ex)],n_boot=0)
    for section in (report.order_permutation,report.alias_renaming):
        assert section["max_prob_drift_overall"] is None
        assert section["decision_flip_rate"] is None
        assert section["mean_accuracy_delta"] is None


def test_timing_has_no_grad_and_forced_probs_are_consistent(tok):
    model=make_model(tok)
    ex=build_choice(tok,"query",[("a","w1"),("b","w2")],{"a":1},forced_choice=True)
    original=model.encode
    observed=[]
    def checked(packed):
        observed.append(torch.is_grad_enabled())
        return original(packed)
    model.encode=checked
    measurement=_measure_single_request(model,ex,"a",None,4096)
    assert not measurement["failed"] and observed==[False]
    assert not model.training
    output=model.predict(ex,full_support=True)
    assert REFERENCE_ID not in output.prob_by_id()
    assert abs(sum(output.prob_by_id().values())-1)<1e-10


def test_operation_count_uses_actual_rectangular_shapes():
    a=torch.ones(3,7);b=torch.ones(7,11)
    with OperationCounter() as counter:
        _=a@b
    assert sum(counter.flops.values())==2*3*7*11


def test_interrupted_training_resume_matches_uninterrupted(tok,tmp_path):
    records={"r":{"rid":"r","mode":"choice","bucket":"openjev_choice","domain":"test",
                  "query":"query","options":[("a","w1"),("b","w2")],"target":{"a":1.,REFERENCE_ID:0.},"is_soft":False}}
    trace=[[{"rid":"r","bucket":"openjev_choice","options":[("a","w1"),("b","w2")],
              "target":{"a":1.,REFERENCE_ID:0.},"augmentation":{"realized_k":2,"null_augmented":False}}] for _ in range(3)]
    cfg=TrainConfig(seed=71,optimizer_updates=3,effective_batch_size=1,bf16=False)
    full=make_model(tok,seed=5)
    train_diffjev(full,cfg,records,trace,tok)
    interrupted=make_model(tok,seed=5)
    def stop(row):
        raise RuntimeError("interrupted after durable checkpoint")
    with pytest.raises(RuntimeError,match="interrupted"):
        train_diffjev(interrupted,cfg,records,trace,tok,checkpoint_dir=str(tmp_path),save_every=1,log_fn=stop)
    resumed=make_model(tok,seed=5)
    train_diffjev(resumed,cfg,records,trace,tok,resume_from=str(tmp_path/"step_1.pt"))
    for (name,a),(_,b) in zip(full.named_parameters(),resumed.named_parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0,msg=name)


def test_formal_notebooks_never_import_synthetic_or_mock():
    root=Path(__file__).resolve().parents[1]
    for path in (root/"notebooks").glob("*.ipynb"):
        nb=json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"]=="code":
                source="".join(cell["source"])
                compile(source,str(path),"exec")
                assert "build_mock" not in source and "synthetic" not in source
    pins=json.loads((root/"configs/revisions.json").read_text())
    assert all(len(p["revision"])==40 for p in pins.values())


def test_official_evaluator_bridge_preserves_native_ids_and_score(tmp_path,tok):
    from diffjev.workflows import _example
    from diffjev.evaluation.exp1_benchmark import JevBenchCase
    from diffjev.evaluation.official import evaluate_official
    records=[
        {"case_id":"c","family_id":"fc","slice":"medqa_usmle","task_type":"choice","perturbation":"original",
         "state":{"vignette":"case"},"question":{"type":"choice","instructions":"choose","criteria":{"a":"one","b":"two"}},"gold":"b"},
        {"case_id":"s","family_id":"fs","slice":"sst5","task_type":"score","perturbation":"original",
         "state":{"sentence":"movie"},"question":{"type":"score","instructions":"rate","criteria":["bad","good"]},"gold":1},
        {"case_id":"n","family_id":"fn","slice":"atbench500","task_type":"noul","perturbation":"original",
         "state":{"trajectory":"safe"},"question":{"type":"noul","instructions":"unsafe?","criteria":{"true":"unsafe","false":"safe"}},"gold":False},
    ]
    cases,predictions=[],[]
    for raw in records:
        mapped=map_jevbench_record(raw);ex=_example(tok,mapped)
        case=JevBenchCase(raw["case_id"],raw["family_id"],raw["slice"],ex)
        cases.append(case)
        probabilities={REFERENCE_ID:float(ex.target[0]),**{c.cid:float(p) for c,p in zip(ex.candidates,ex.target[1:])}}
        predictions.append({"case_id":case.case_id,"top1":ex.gold_id,"probs":probabilities,
                            "expected_score":1.0 if ex.mode=="score" else None,
                            "p_yes":0.0 if ex.mode=="noul" else None,"failed":False})
    result=evaluate_official(cases,predictions,tmp_path)
    assert result["slices"]["medqa_usmle"]["metrics"]["accuracy"]==1
    assert result["slices"]["sst5"]["metrics"]["mae_expected_level"]==0
    assert result["slices"]["atbench500"]["metrics"]["accuracy_at_0.5"]==1


def test_jeva_loader_uses_official_local_runtime(monkeypatch,tmp_path,tok):
    import sys
    from diffjev.baselines.external import JevaBaseline
    calls={}
    class Jeva:
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            calls.update(path=path,**kwargs)
            return SimpleNamespace(model=SimpleNamespace(tokenizer=tok))
    monkeypatch.setitem(sys.modules,"jeva",SimpleNamespace(Jeva=Jeva))
    baseline=JevaBaseline(str(tmp_path),device="cpu")
    assert baseline.model is not None
    assert calls["local_files_only"] is True and calls["cache"]=="none"
    assert calls["path"]==str(tmp_path)


def test_bge_e5_use_own_text_tokenization_and_pooling(tok):
    from diffjev.baselines.retrieval import DenseBiEncoderBaseline
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        def __init__(self):self.seen=[]
        def __call__(self,texts,**kwargs):
            self.seen.extend(texts)
            assert kwargs["truncation"] is False
            return Batch(input_ids=torch.tensor([[1,2,0] for _ in texts]),
                         attention_mask=torch.tensor([[1,1,0] for _ in texts]))
    class Model:
        def __call__(self,**inputs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[1.,0.],[0.,2.],[100.,100.]]]).expand(len(inputs["input_ids"]),-1,-1))
    for family in ("bge","e5"):
        runner=DenseBiEncoderBaseline(family=family)
        runner.tokenizer=Tokenizer();runner._hf_model=Model()
        encoded=runner._encode_texts(["original text"],query=True)
        assert "original text" in runner.tokenizer.seen[0]
        expected=torch.tensor([[1.,0.]]) if family=="bge" else torch.nn.functional.normalize(torch.tensor([[.5,1.]]),dim=-1)
        torch.testing.assert_close(encoded,expected)
