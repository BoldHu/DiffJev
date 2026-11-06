"""Offline loader regressions; optional tiny native round-trip, never 8B weights."""
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from diffjev.backbones.llada import LLaDAAdapter, _compatible_llada_class


class LegacyWrapper(nn.Module):
    """The relevant legacy lifecycle: no post_init, no tie_weights keywords."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)

    def get_expanded_tied_weights_keys(self, all_submodels=False):
        return {}

    def post_init(self):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys()
        self.tie_weights(recompute_mapping=False)

    def tie_weights(self):
        if self.config.weight_tying:
            self.head.weight = self.embedding.weight


def test_legacy_lifecycle_bridge_preserves_parameters_and_independent_head():
    config = SimpleNamespace(weight_tying=False, tie_word_embeddings=True)
    torch.manual_seed(9)
    native = LegacyWrapper(config)
    torch.manual_seed(9)
    compatible = _compatible_llada_class(LegacyWrapper)(config)
    assert not hasattr(native, "all_tied_weights_keys")  # no global patch
    assert compatible.all_tied_weights_keys == {}
    compatible.tie_weights(missing_keys=set(), recompute_mapping=False)
    assert compatible.head.weight is not compatible.embedding.weight
    assert native.state_dict().keys() == compatible.state_dict().keys()
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, compatible.state_dict()[name], rtol=0, atol=0)


def test_unverified_tied_config_is_rejected():
    with pytest.raises(ValueError, match="weight_tying=False"):
        _compatible_llada_class(LegacyWrapper)(SimpleNamespace(weight_tying=True))


def test_v4_class_is_unchanged():
    class OldWrapper(nn.Module):
        pass
    assert _compatible_llada_class(OldWrapper) is OldWrapper


@pytest.mark.parametrize("problem", [None, "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_local_loader_uses_bridge_and_rejects_inexact_checkpoints(tmp_path, monkeypatch, problem):
    calls = []
    config = SimpleNamespace(model_type="llada", weight_tying=False,
                             auto_map={"AutoModel": "modeling_llada.LLaDAModelLM"})

    class Native(LegacyWrapper):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            model = cls(kwargs["config"])
            # v5 finalization: both accesses failed with the legacy wrapper.
            assert not (set() - model.all_tied_weights_keys.keys())
            model.tie_weights(missing_keys=set(), recompute_mapping=False)
            return model, {problem: ["bad.weight"]} if problem else {}

    class Metadata:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"trust_remote_code": True, "local_files_only": True}
            return config

    def resolve(class_ref, path, **kwargs):
        assert class_ref == "modeling_llada.LLaDAModelLM"
        assert kwargs == {"local_files_only": True}
        return Native

    hf, dynamic = ModuleType("transformers"), ModuleType("transformers.dynamic_module_utils")
    hf.AutoConfig = hf.AutoTokenizer = Metadata
    dynamic.get_class_from_dynamic_module = resolve
    monkeypatch.setitem(sys.modules, "transformers", hf)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)

    class CaptureAdapter(LLaDAAdapter):
        def __init__(self, model, tokenizer, **kwargs):
            nn.Module.__init__(self)
            self.hf_model = model

    if problem:
        with pytest.raises(RuntimeError, match="did not load exactly"):
            CaptureAdapter.from_local(str(tmp_path))
    else:
        adapter = CaptureAdapter.from_local(str(tmp_path))
        assert isinstance(adapter.hf_model, Native)
    assert calls[0]["dtype"] == torch.bfloat16
    assert calls[0]["local_files_only"] is True
    assert calls[0]["output_loading_info"] is True


def test_tiny_native_hf5_checkpoint_roundtrip(tmp_path):
    """Runs with formal HF packages + local pinned config/code, on CPU only.

    DIFFJEV_LLADA_SOURCE may point at a metadata-only pinned snapshot. No network
    calls, tokenizer, pretrained weights or large model allocation are needed.
    """
    hf = pytest.importorskip("transformers")
    if not hasattr(hf.PreTrainedModel, "get_expanded_tied_weights_keys"):
        pytest.skip("Requires the formal Transformers 5 loader")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import save_file

    source = Path(os.environ.get("DIFFJEV_LLADA_SOURCE", "models/GSAI-ML__LLaDA-8B-Base"))
    if not all((source / name).is_file() for name in
               ("config.json", "configuration_llada.py", "modeling_llada.py")):
        pytest.skip("Pinned local LLaDA config/code is not installed")
    config = hf.AutoConfig.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
    config.d_model, config.n_heads, config.n_kv_heads, config.n_layers = 16, 4, 4, 2
    config.mlp_hidden_size, config.embedding_size, config.vocab_size = 32, 64, 64
    config.max_sequence_length, config.mask_token_id = 32, 63
    config.pad_token_id = config.eos_token_id = 62
    native_class = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM", str(source), local_files_only=True
    )
    native = native_class(config, init_params=True).eval()
    save_file(native.state_dict(), str(tmp_path / "model.safetensors"))
    # Reproduce the user's failure using real upstream code at tiny dimensions.
    with pytest.raises(AttributeError, match="all_tied_weights_keys"):
        native_class.from_pretrained(str(tmp_path), config=config, local_files_only=True)
    loaded, info = _compatible_llada_class(native_class).from_pretrained(
        str(tmp_path), config=config, local_files_only=True, dtype=torch.float32,
        output_loading_info=True,
    )
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert loaded.all_tied_weights_keys == {}
    assert loaded.get_input_embeddings().weight is not loaded.get_output_embeddings().weight
    assert native.state_dict().keys() == loaded.state_dict().keys()
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(native(ids).logits, loaded(ids).logits, rtol=0, atol=0)
