"""Duplicate benchmark alternatives retain native label identity and support."""
import copy

import pytest
import torch

from diffjev.data.sources import map_jevbench_record
from diffjev.workflows import _example
from diffjev.evaluation.official import official_record
from diffjev.evaluation.exp1_benchmark import JevBenchCase
from diffjev.evaluation.exp2_robustness import apply_alias_scheme
from tests.conftest import make_model


def duplicate_case():
    return {"case_id":"fixture-original","family_id":"fixture-family","slice":"medmcqa",
        "task_type":"choice","perturbation":"original","state":{"question":"toy question"},
        "question":{"type":"choice","instructions":"Choose","criteria":{"a":"w1","b":"w1","c":"w2"}},
        "option_order":[0,1,2],"gold":"b"}


def test_duplicate_options_preserve_gold_and_full_native_distribution(tok):
    raw=duplicate_case()
    rec=map_jevbench_record(raw)
    ids=[cid for cid,_ in rec["options"]]
    assert len(ids)==len(set(ids))==3
    assert [text for _,text in rec["options"]]==["w1","w1","w2"]
    assert rec["gold_id"]==ids[1]
    assert rec["duplicate_option_groups"][0]["source_option_indices"]==[0,1]
    changed=map_jevbench_record(dict(raw,gold="a"))
    assert changed["options"]==rec["options"]  # identity must not depend on gold
    ex=_example(tok,rec)
    assert ex.candidates[0].token_ids==ex.candidates[1].token_ids
    torch.testing.assert_close(ex.target,torch.tensor([0.,0.,1.,0.],dtype=torch.float64))
    case=JevBenchCase(raw["case_id"],raw["family_id"],raw["slice"],ex)
    out=official_record(case,{"top1":ids[1],"probs":dict(zip(ids,[.2,.5,.3])),"failed":False})
    assert out["gold"]==out["pred"]=="b"
    assert out["probabilities"]=={"a":.2,"b":.5,"c":.3}


def test_duplicate_source_ids_survive_reordered_twins_and_aliases(tok):
    first=map_jevbench_record(duplicate_case())
    twin=copy.deepcopy(duplicate_case())
    twin.update(case_id="fixture-reordered",option_order=[2,1,0],gold="renamed_y")
    twin["question"]["criteria"]={"renamed_x":"w2","renamed_y":"w1","renamed_z":"w1"}
    second=map_jevbench_record(twin)
    assert second["options"]==[first["options"][i] for i in [2,1,0]]
    assert second["gold_id"]==first["gold_id"]
    ex=_example(tok,first)
    renamed=apply_alias_scheme(ex,"numeric_bracket")
    assert [(c.cid,c.text) for c in renamed.candidates]==[(c.cid,c.text) for c in ex.candidates]
    model=make_model(tok)
    a=model.predict(ex,full_support=True)
    b=model.predict(ex.permuted([2,1,0]),full_support=True)
    for cid,p in a.prob_by_id().items():
        assert b.prob_by_id()[cid]==pytest.approx(p,abs=1e-10)
    assert a.top1==b.top1


def test_duplicate_case_without_source_order_is_not_arbitrarily_relabelled():
    raw=duplicate_case();raw.pop("option_order")
    with pytest.raises(ValueError,match="fixture-original.*option_order"):
        map_jevbench_record(raw)


def test_normalized_collision_keeps_original_option_text():
    raw=duplicate_case();raw["question"]["criteria"]["b"]=" w1 "
    rec=map_jevbench_record(raw)
    assert rec["duplicate_option_groups"][0]["match"]=="normalized_text_collision"
    assert rec["options"][1][1]==" w1 "
