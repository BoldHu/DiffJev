"""Permutation equivariance (inference and coupled training), label leakage."""

import random

import torch

from diffjev.data.synthetic import synthetic_choice
from tests.conftest import make_model


def _perm(K, seed):
    p = list(range(K))
    random.Random(seed).shuffle(p)
    return p


def _assert_same_maps(a, b, tol):
    assert set(a) == set(b)
    for k in a:
        assert abs(a[k] - b[k]) < tol, (k, a[k], b[k])


def test_inference_equivariance_no_pruning(tok):
    m = make_model(tok)
    for seed, K in [(0, 2), (1, 4), (2, 7), (3, 16)]:
        ex = synthetic_choice(tok, random.Random(seed), K, max_len=6)
        base = m.predict(ex)
        for ps in range(3):
            out = m.predict(ex.permuted(_perm(K, 10 * seed + ps)))
            _assert_same_maps(base.prob_by_id(), out.prob_by_id(), 1e-9)
            assert out.top1 == base.top1


def test_inference_equivariance_with_pruning(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(5), 40, max_len=4)
    base = m.predict(ex)
    assert base.probability_scope == "retained_support"
    out = m.predict(ex.permuted(_perm(40, 1)))
    _assert_same_maps(base.prob_by_id(), out.prob_by_id(), 1e-9)
    for a, b in zip(base.survivor_ids_per_pass, out.survivor_ids_per_pass):
        assert set(a) == set(b)


def test_sequential_positions_break_equivariance(tok):
    m = make_model(tok, position_mode="sequential")
    ex = synthetic_choice(tok, random.Random(0), 4, max_len=6)
    base = m.predict(ex).prob_by_id()
    out = m.predict(ex.permuted([3, 2, 1, 0])).prob_by_id()
    assert max(abs(base[k] - out[k]) for k in base) > 1e-6


def test_training_loss_equivariance_with_coupled_noise(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(2), 6, max_len=5)
    K = 6
    u = torch.rand(K, generator=torch.Generator().manual_seed(0))
    action = int(torch.argmax(ex.target))
    perm = _perm(K, 4)
    ex_p = ex.permuted(perm)
    action_p = 0 if action == 0 else perm.index(action - 1) + 1
    for t in (1.0, 0.5):
        la = m.training_losses(ex, torch.Generator().manual_seed(1), t=t, noise=u, action=action)
        lb = m.training_losses(ex_p, torch.Generator().manual_seed(1), t=t, noise=u[perm], action=action_p)
        for k in ("L_mask", "L_task_CE", "L_Brier", "total"):
            assert abs(float(la[k]) - float(lb[k])) < 1e-5, k


def test_no_label_leakage_in_inference(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(9), 5)
    a = m.predict(ex).prob_by_id()
    ex.target = torch.flip(ex.target, [0])
    b = m.predict(ex).prob_by_id()
    _assert_same_maps(a, b, 0.0 + 1e-15)


def test_alias_renaming_does_not_change_output(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(4), 4)
    a = m.predict(ex).prob_by_id()
    for i, c in enumerate(ex.candidates):
        c.alias = "XYZW"[i]
    b = m.predict(ex).prob_by_id()
    _assert_same_maps(a, b, 1e-15)

