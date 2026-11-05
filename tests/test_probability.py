"""Structured probability normalisation, K = 1 identity, score readout."""

import random

import pytest
import torch

from diffjev.diffusion import MASK, NO, YES
from diffjev.probability import (
    compatibility_mask,
    forced_choice_log_probs,
    log_one_minus_marginals,
    score_readout,
    structured_log_probs,
)
from diffjev.tasks import build_score
from tests.conftest import make_model


def test_partition_normalisation_sums_to_one():
    z = torch.randn(7, dtype=torch.float64)
    lp = structured_log_probs(z)
    p = lp.exp()
    assert torch.isclose(p.sum(), torch.tensor(1.0, dtype=torch.float64))
    p_ref = 1.0 / (1.0 + z.exp().sum())
    assert torch.isclose(p[0], p_ref)
    assert torch.allclose(p[1:], z.exp() * p_ref)


def test_extreme_energies_are_stable():
    z = torch.tensor([800.0, -800.0, 0.0])
    lp = structured_log_probs(z)
    assert torch.isfinite(lp).all()
    assert torch.isclose(lp.exp().sum(), torch.tensor(1.0))
    l1m = log_one_minus_marginals(lp)
    assert torch.isfinite(l1m).all()


def test_k1_sigmoid_identity():
    z = torch.tensor([1.7], dtype=torch.float64)
    p = structured_log_probs(z).exp()
    assert torch.isclose(p[1], torch.sigmoid(z[0]))


def test_marginal_is_not_sigmoid_for_k_gt_1():
    z = torch.tensor([1.0, 1.0], dtype=torch.float64)
    p = structured_log_probs(z).exp()
    assert not torch.isclose(p[1], torch.sigmoid(z[0]))


def test_log_one_minus_matches_direct():
    z = torch.randn(5, dtype=torch.float64)
    lp = structured_log_probs(z)
    assert torch.allclose(log_one_minus_marginals(lp), torch.log1p(-lp[1:].exp()), atol=1e-12)


def test_compatibility_exclusion():
    st = torch.tensor([MASK, NO, MASK])
    c = compatibility_mask(st)
    assert c.tolist() == [True, True, False, True]
    st2 = torch.tensor([MASK, YES, MASK])
    assert compatibility_mask(st2).tolist() == [False, False, True, False]
    with pytest.raises(ValueError):
        compatibility_mask(torch.tensor([YES, YES]))
    lp = structured_log_probs(torch.randn(3, dtype=torch.float64), compat=c)
    assert lp.exp()[2].item() < 1e-300
    assert torch.isclose(lp.exp().sum(), torch.tensor(1.0, dtype=torch.float64))


def test_forced_choice_excludes_reference():
    z = torch.randn(4, dtype=torch.float64)
    p = forced_choice_log_probs(z).exp()
    assert torch.allclose(p, torch.softmax(z, 0))


def test_score_readout_and_adapter(tok):
    bins = [(3.0, "bin w3"), (0.0, "bin w0"), (1.0, "bin w1"), (4.0, "bin w4"), (2.0, "bin w2")]
    ex = build_score(tok, "rubric", "query w5", bins, {2.0: 0.6, 3.0: 0.4})
    assert ex.bin_values == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [c.cid for c in ex.candidates] == ["bin:1", "bin:2", "bin:3", "bin:4"]
    assert torch.allclose(ex.target, torch.tensor([0, 0, 0.6, 0.4, 0], dtype=torch.float64))
    # reference bin description is part of the shared query
    assert tok.encode("bin w0")[1] in ex.query_ids
    lp = structured_log_probs(torch.randn(4, dtype=torch.float64))
    p, e, cdf = score_readout(lp, torch.tensor(ex.bin_values, dtype=torch.float64))
    assert torch.isclose(e, (p * torch.arange(5.0, dtype=torch.float64)).sum())
    assert (cdf[1:] >= cdf[:-1]).all() and torch.isclose(cdf[-1], torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError):
        score_readout(lp, torch.tensor([0.0, 2.0, 1.0, 3.0, 4.0], dtype=torch.float64))


def test_model_noul_and_score_outputs(tok):
    from diffjev.data.synthetic import synthetic_noul, synthetic_score

    m = make_model(tok)
    rng = random.Random(0)
    n = m.predict(synthetic_noul(tok, rng))
    assert n.probability_scope == "full_support"
    assert abs(n.candidate_probs[0].item() - n.p_yes) < 1e-9
    assert abs(n.reference_prob + n.p_yes - 1.0) < 1e-9
    s = m.predict(synthetic_score(tok, rng))
    assert s.score_probs.shape[0] == 5
    assert abs(float(s.score_probs.sum()) - 1.0) < 1e-6
    assert 0.0 <= s.expected_score <= 4.0

