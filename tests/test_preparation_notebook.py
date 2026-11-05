"""Preparation orchestration tests: no package installs or asset downloads."""
import fnmatch
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from diffjev.notebook_runtime import NotebookRuntime
from diffjev.preparation import validate_prepared,PREPARED_FILES
from diffjev.data.native import digest


def test_worker_logs_and_propagates_failure(tmp_path):
    worker=NotebookRuntime(tmp_path,sys.executable)
    worker.python_code("import os; print('worker output'); assert os.environ['DIFFJEV_ROOT']", "success")
    assert "worker output" in (worker.logs/"success.log").read_text()
    with pytest.raises(RuntimeError,match="failed"):
        worker.python_code("raise RuntimeError('fixture failure')","failure")
    assert "fixture failure" in (worker.logs/"failure.log").read_text()


def test_prepared_manifest_reuse_checks_identity_and_integrity(tmp_path):
    values={name:{} for name in PREPARED_FILES}
    values.update(trace=[[{"rid":"one"}],[{"rid":"two"}]],banks={"domain":[]},manifest={"splits":{}})
    values["audit"]={"seed":0,**{key:digest(values[name]) for name,key in
        (("trace","trace_sha256"),("banks","bank_sha256"),("manifest","manifest_sha256"))}}
    for name,value in values.items():
        (tmp_path/(name+".json")).write_text(json.dumps(value))
    assert validate_prepared(tmp_path,updates=2)["optimizer_updates"]==2
    with pytest.raises(ValueError,match="different"):
        validate_prepared(tmp_path,seed=1,updates=2)
    (tmp_path/"banks.json").write_text('{"altered":true}')
    with pytest.raises(ValueError,match="checksum"):
        validate_prepared(tmp_path,updates=2)


def test_metadata_download_cannot_fetch_weight_files(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("download_assets_for_test",root/"scripts/download_assets.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    hub=SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=lambda *a,**kw:SimpleNamespace(sha="a"*40)),
                        snapshot_download=lambda **kw:calls.append(kw))
    monkeypatch.setitem(sys.modules,"huggingface_hub",hub)
    cfg={"repo_id":"fixture/model","revision":"a"*40,"local_dir":"models/fixture"}
    module.download_model("fixture",cfg,str(tmp_path),metadata_only=True)
    patterns=calls[0]["allow_patterns"]
    for file in ("model.safetensors","model-00001-of-00004.safetensors","pytorch_model.bin","head.pt"):
        assert not any(fnmatch.fnmatch(file,pattern) for pattern in patterns)
    for file in ("config.json","tokenizer.json","tokenizer.model","modeling_backbone.py"):
        assert any(fnmatch.fnmatch(file,pattern) for pattern in patterns)
    assert json.loads((tmp_path/"fixture/download_provenance.json").read_text())["last_download_scope"]=="metadata_only"


def test_setup_notebook_bootstraps_before_research_imports():
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/"notebooks/00_setup.ipynb").read_text(encoding="utf-8"))
    sources=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
    for source in sources:
        compile(source,"notebook00","exec")
    assert all("from diffjev import workflows" not in s for s in sources)
    assert "INSTALL_ENV = False" in sources[0] and "DOWNLOAD_ASSETS = False" in sources[0]
    assert "PREPARE_DATA = False" in sources[0]
    text="\n".join(sources)
    assert text.index("drive.mount") < text.index("install_environment(ROOT")
    assert "--metadata-only" in text and '"--restore-states"' in text
    for path in (root/"notebooks").glob("0[1-7]*.ipynb"):
        cells=json.loads(path.read_text())["cells"]
        code=["".join(c["source"]) for c in cells if c["cell_type"]=="code"]
        assert "NotebookRuntime" in code[0]
        assert all(c.startswith("run_python(") for c in code[1:])
