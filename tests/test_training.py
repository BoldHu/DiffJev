"""Gradient propagation, freeze policy, no new parameters, verbalizers, smoke training."""

import random

import pytest
import torch

from diffjev.backbones import VerbalizerError, resolve_single_token
from diffjev.config import DiffJevConfig
from diffjev.data.synthetic import synthetic_choice, synthetic_mixture, synthetic_noul, synthetic_score
from diffjev.model import DiffJevModel
from diffjev.training import apply_freeze_policy, parameter_counts, train_synthetic, unique_parameters
from tests.conftest import make_backbone


def _frozen_model(tok, **cfg):
    bb = make_backbone(tok, dtype=torch.float32)
    counts = apply_freeze_policy(bb, 4)
    return DiffJevModel(bb, DiffJevConfig(**cfg)), counts


def test_freeze_policy_and_counts(tok):
    m, counts = _frozen_model(tok)
    bb = m.backbone
    expected = sum(p.numel() for i in bb.top_layer_indices(4) for p in bb.layer_modules()[i].parameters())
    expected += sum(p.numel() for p in bb.final_norm_module().parameters())
    assert counts["trainable"] == expected
    assert not any(p.requires_grad for p in bb.input_embedding_module().parameters())
    assert not any(p.requires_grad for p in bb.output_head_module().parameters())
    for i in bb.lower_layer_indices(4):
        assert not any(p.requires_grad for p in bb.layer_modules()[i].parameters())


def test_no_new_parameters(tok):
    bb = make_backbone(tok)
    before = parameter_counts(bb)["total"]
    m = DiffJevModel(bb, DiffJevConfig())
    assert parameter_counts(m)["total"] == before
    assert {id(p) for p in unique_parameters(m)} == {id(p) for p in unique_parameters(bb)}


@pytest.mark.parametrize("t", [1.0, 0.5])
def test_gradients_reach_exactly_the_trainable_set(tok, t):
    m, _ = _frozen_model(tok)
    ex = synthetic_choice(tok, random.Random(0), 5)
    K = 5
    losses = m.training_losses(ex, torch.Generator().manual_seed(0), t=t, noise=torch.linspace(0.05, 0.95, K))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    for n, p in m.backbone.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, n
            assert torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None, n
    top_grad = sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.requires_grad)
    assert top_grad > 0


def test_losses_for_all_modes(tok):
    m, _ = _frozen_model(tok)
    rng = random.Random(0)
    g = torch.Generator().manual_seed(0)
    for ex in (synthetic_noul(tok, rng), synthetic_score(tok, rng), synthetic_choice(tok, rng, 3, null_prob=1.0)):
        L = m.training_losses(ex, g)
        assert set(["L_task_CE", "L_Brier", "total"]) <= set(L)
        assert torch.isfinite(L["total"])


def test_full_context_t1_control_config(tok):
    m, _ = _frozen_model(tok, num_passes=1, use_mask_loss=False, first_pass_detail_memory=True)
    ex = synthetic_choice(tok, random.Random(0), 4)
    L = m.training_losses(ex, torch.Generator().manual_seed(0))
    assert "L_mask" not in L
    _, ro = m.predict(ex, return_rollout=True)
    assert len(ro.passes) == 1 and ro.passes[0].used_memory


def test_verbalizers_must_be_single_token(tok):
    assert resolve_single_token(tok, "Yes") == tok.tok2id["Yes"]
    with pytest.raises(VerbalizerError):
        resolve_single_token(tok, "Maybe")


def test_iia_nonzero_rejected():
    with pytest.raises(NotImplementedError):
        DiffJevConfig(iia_weight=0.1)


def test_smoke_training_reduces_task_ce(tok):
    torch.manual_seed(0)
    m, _ = _frozen_model(tok)
    data = synthetic_mixture(tok, 6, seed=1, ks=(2, 3))
    hist = train_synthetic(m, data, num_updates=40, effective_batch=6, lr=3e-3, seed=0)
    first = sum(h["L_task_CE"] for h in hist[:8]) / 8
    last = sum(h["L_task_CE"] for h in hist[-8:]) / 8
    assert all(torch.isfinite(torch.tensor(h["total"])) for h in hist)
    assert last < first

