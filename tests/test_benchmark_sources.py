"""Small local fixtures for missing-source recovery; no network or real data."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from diffjev.data.benchmark_sources import missing_source_plan,download_benchmark_sources


def source_fixture():
    source={"key":"sst5","hf_repo":"fixture/sst5","revision":"a"*40,"files":["test.jsonl"]}
    common=SimpleNamespace(parent_key=lambda key:key,source_by_key=lambda key:source)
    cases=[{"slice":"sst5","case_id":"one","hf_repo":source["hf_repo"],"revision":source["revision"]},
           {"slice":"sst5","case_id":"two"}]
    return common,cases,source


def test_missing_source_plan_deduplicates_and_reuses(tmp_path):
    common,cases,source=source_fixture()
    plan=missing_source_plan(common,cases,tmp_path)
    assert len(plan)==1 and plan[0]["revision"]=="a"*40
    target=Path(plan[0]["target"]);target.parent.mkdir();target.write_text('{}\n')
    assert missing_source_plan(common,cases,tmp_path)==[]
    target.unlink()
    assert missing_source_plan(common,[dict(c,state={"sentence":"restored"}) for c in cases],tmp_path)==[]
    source["revision"]="main"
    with pytest.raises(ValueError,match="Unpinned"):
        missing_source_plan(common,cases,tmp_path)


def test_download_sources_uses_manifest_revision_only(monkeypatch,tmp_path):
    from diffjev import workflows as w
    from diffjev.data import sources
    from diffjev.evaluation import official
    common,cases,_=source_fixture()
    monkeypatch.setattr(w,"ROOT",tmp_path)
    monkeypatch.setattr(sources,"load_local_dataset",lambda *a:cases)
    monkeypatch.setattr(official,"load_official",lambda:(common,None,None))
    calls=[]
    def fake_download(**kw):
        calls.append(kw)
        path=Path(kw["local_dir"])/kw["filename"]
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{"text":"toy sentence","label":1}\n')
        return str(path)
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(hf_hub_download=fake_download))
    report=download_benchmark_sources()
    assert report["downloaded_files"]==1
    assert calls[0]["repo_type"]=="dataset" and calls[0]["revision"]=="a"*40
    assert download_benchmark_sources()["downloaded_files"]==0
    assert len(calls)==1
    assert (tmp_path/"data/raw/jevbench_sources/sst5/source_downloads.json").exists()


def test_restored_jsonl_trailing_blank_lines_are_accepted(tmp_path):
    from diffjev.data.sources import load_local_dataset
    path=tmp_path/"raw/jevbench";cases=path/"cases";cases.mkdir(parents=True)
    names=["aegis2","aegis2_response","atbench500","banking77","jailbreak_classification",
           "medmcqa","medqa_usmle","mmlu_pro","prompt_injections","pubmedqa","scienceqa_text","sst5"]
    # Match only release cardinality checks; all content is an artificial fixture.
    for j,name in enumerate(names):
        rows=[{"case_id":str(i),"family_id":str(i%5516)} for i in range(j,8016,len(names))]
        (cases/(name+".jsonl")).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (path/"restored_states.jsonl").write_text('{"case_id":"0","state":{"sentence":"toy"}}\n\n')
    loaded=load_local_dataset("jevbench",str(tmp_path))
    assert next(row for row in loaded if row["case_id"]=="0")["state"]=={"sentence":"toy"}


def test_restore_keeps_official_hash_check(monkeypatch,tmp_path):
    from diffjev import workflows as w
    _,cases,source=source_fixture()
    source["format"]="jsonl"
    manifest=tmp_path/"data/raw/jevbench";manifest.mkdir(parents=True)
    (manifest/"sources.json").write_text(json.dumps({"sources":[source]}))
    upstream=tmp_path/"data/raw/jevbench_sources/sst5";upstream.mkdir(parents=True)
    (upstream/"test.jsonl").write_text('{"text":"toy sentence","label":1}\n')
    expected={"sentence":"toy sentence"}
    cases=cases[:1]
    cases[0].update(row_index=0,option_order=[0,1,2,3,4],
        state_hash=hashlib.sha256(json.dumps(expected,separators=(",",":")).encode()).hexdigest()[:16])
    monkeypatch.setattr(w,"ROOT",tmp_path)
    monkeypatch.setattr(w,"load_local_dataset",lambda *a:cases)
    assert w.restore_benchmark_states()["restored"]==1
    path=manifest/"restored_states.jsonl"
    saved=path.read_text()
    assert json.loads(saved)["state"]==expected
    cases[0]["state_hash"]="0"*16
    with pytest.raises(ValueError,match="hash mismatch"):
        w.restore_benchmark_states()
    assert path.read_text()==saved
