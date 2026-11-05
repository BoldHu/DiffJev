"""Forward corruption, mask recovery, mean-field reverse updates, reference sampler."""

import random

import torch

from diffjev.data.synthetic import synthetic_choice
from diffjev.diffusion import (
    MASK,
    NO,
    YES,
    clean_labels,
    corrupt,
    initial_soft_state,
    is_legal_hard_state,
    mean_field_update,
    sample_t,
    slot_inputs,
    stochastic_reference_step,
)
from diffjev.losses import mask_loss
from diffjev.probability import compatibility_mask, structured_log_probs
from tests.conftest import make_model


def test_t_sampling_distribution():
    g = torch.Generator().manual_seed(0)
    ts = [sample_t(g) for _ in range(2000)]
    full = sum(t == 1.0 for t in ts) / len(ts)
    assert 0.45 < full < 0.55
    assert all(1e-3 <= t <= 1.0 for t in ts)


def test_clean_states_are_legal():
    assert clean_labels(0, 4).tolist() == [NO] * 4
    assert clean_labels(3, 4).tolist() == [NO, NO, YES, NO]


def test_coupled_corruption_commutes_with_permutation():
    lab = clean_labels(2, 5)
    u = torch.rand(5, generator=torch.Generator().manual_seed(1))
    perm = [4, 0, 3, 1, 2]
    assert torch.equal(corrupt(lab, 0.5, u)[perm], corrupt(lab[perm], 0.5, u[perm]))
    assert (corrupt(lab, 1.0, u) == MASK).all()


def test_mask_loss_zero_without_masked_slots():
    lab = clean_labels(2, 4)
    state = lab.clone()
    lp = structured_log_probs(torch.randn(4, dtype=torch.float64), compat=compatibility_mask(state))
    assert float(mask_loss(lp, lab, state, 0.3)) == 0.0


def test_revealed_yes_recovers_masked_slots():
    """A revealed Yes forces every masked slot to No: masked loss is ~0."""
    lab = clean_labels(2, 4)
    state = torch.tensor([MASK, YES, MASK, MASK])
    lp = structured_log_probs(torch.randn(4, dtype=torch.float64), compat=compatibility_mask(state))
    assert torch.isclose(lp.exp()[2], torch.tensor(1.0, dtype=torch.float64))
    assert float(mask_loss(lp, lab, state, 0.75)) < 1e-12


def test_mask_loss_denominator_is_k():
    lab = clean_labels(1, 3)
    state = torch.tensor([MASK, NO, NO])
    z = torch.zeros(3, dtype=torch.float64)
    lp = structured_log_probs(z, compat=compatibility_mask(state))
    # compatible outcomes: ref and cand 1, each P = 0.5 -> BCE(1, 0.5) = log 2
    expected = torch.log(torch.tensor(2.0, dtype=torch.float64)) / (3 * 0.5)
    assert torch.isclose(mask_loss(lp, lab, state, 0.5), expected)


def test_mean_field_update_and_schedule():
    K = 4
    mu = initial_soft_state(K, dtype=torch.float64)
    p = torch.tensor([0.1, 0.5, 0.2, 0.0], dtype=torch.float64)
    mu1 = mean_field_update(mu, p, 1.0, 2 / 3)
    assert torch.allclose(mu1.sum(1), torch.ones(K, dtype=torch.float64))
    assert torch.allclose(mu1[:, YES], p / 3)
    assert torch.allclose(mu1[:, MASK], torch.full((K,), 2 / 3, dtype=torch.float64))
    mu2 = mean_field_update(mu1, p, 2 / 3, 1 / 3)
    assert torch.allclose(mu2[:, MASK], torch.full((K,), 1 / 3, dtype=torch.float64))
    assert torch.allclose(mu2.sum(1), torch.ones(K, dtype=torch.float64))


def test_boundary_feedback_reset_identity():
    d = 8
    h = torch.randn(3, d, dtype=torch.float64)
    e = [torch.randn(d, dtype=torch.float64) for _ in range(3)]
    x = slot_inputs(h, initial_soft_state(3, dtype=torch.float64), *e)
    assert torch.allclose(x, h, atol=1e-12)
    mu = torch.tensor([[1.0, 0, 0]] * 3, dtype=torch.float64)
    assert torch.allclose(slot_inputs(h, mu, *e), h + e[0] - e[2], atol=1e-12)


def test_rollout_final_state_and_schedule(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(0), 5)
    _, ro = m.predict(ex, return_rollout=True)
    assert [round(r.t, 6) for r in ro.passes] == [1.0, round(2 / 3, 6), round(1 / 3, 6)]
    assert [r.used_memory for r in ro.passes] == [False, True, True]
    # final distribution returned before any update to zero noise
    assert torch.allclose(ro.final_mu[:, MASK], torch.full((5,), 1 / 3, dtype=torch.float64))


def test_full_mask_view_uses_slot_only_visibility(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(1), 3)
    bd = m.encode(m.pack(ex))
    _, _, mem = m.denoiser.corrupted_view(bd, torch.full((3,), MASK))
    assert mem is None
    _, _, mem2 = m.denoiser.corrupted_view(bd, torch.tensor([MASK, NO, MASK]))
    assert mem2 is not None


def test_reference_sampler_never_two_yes():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        K = 5
        state = torch.full((K,), MASK)
        for beta in (1 / 3, 1 / 2, 1.0):
            lp = structured_log_probs(torch.randn(K, generator=g, dtype=torch.float64), compat=compatibility_mask(state))
            state, _ = stochastic_reference_step(state, lp, beta, g)
            assert is_legal_hard_state(state)
        assert not (state == MASK).any()

