"""Data utilities, metrics and the analytic complexity model."""

import random
from collections import Counter

import pytest

from diffjev.complexity import diffjev_cost, serialized_dlm_cost
from diffjev.data.augmentation import augment_menu, sample_target_k
from diffjev.data.mixture import build_trace, trace_stats
from diffjev.data.splits import assign_split, normalize_text, split_records
from diffjev.evaluation.metrics import decision_flip_rate, hit_at_1, max_probability_drift, permutations_for
from diffjev.tasks import REFERENCE_ID, build_choice, merge_null_equivalent


def test_split_is_deterministic_grouped_and_quarantines():
    recs = [{"rid": f"r{i}", "q": f"Question {i % 50}!"} for i in range(400)]
    keys = lambda r: [normalize_text(r["q"])]
    a = split_records(recs, keys, eval_keys={normalize_text("Question 3")})
    b = split_records(recs, keys, eval_keys={normalize_text("Question 3")})
    assert a == b
    assert sorted(a["quarantine"]) == sorted(r["rid"] for r in recs if r["q"] == "Question 3!")
    where = {rid: s for s, ids in a.items() for rid in ids}
    for qi in range(50):
        splits = {where[r["rid"]] for r in recs if r["q"] == f"Question {qi}!"}
        assert len(splits) == 1  # duplicates/twins share a split
    assert assign_split("x") in ("train", "dev", "calib")


def test_k_proposal_respects_native_menu():
    rng = random.Random(0)
    c = Counter(sample_target_k(2, rng) for _ in range(5000))
    assert 0.45 < c[2] / 5000 < 0.55
    assert all(sample_target_k(5, rng) >= 8 for _ in range(200))
    assert sample_target_k(40, rng) == 40


def test_augment_menu_preserves_menu_and_uniqueness():
    rng = random.Random(0)
    bank = [f"n{i}" for i in range(100)]
    for _ in range(100):
        a = augment_menu(["p", "x1", "x2"], "p", bank, rng)
        assert {"x1", "x2"} <= set(a.candidate_ids)
        assert len(set(a.candidate_ids)) == len(a.candidate_ids)
        assert a.target_id == "p" and "p" in a.candidate_ids
    with pytest.raises(ValueError):
        augment_menu([f"c{i}" for i in range(33)], "c0", bank, rng)
    nulls = [augment_menu(["p", "x"], "p", bank, rng, null_rate=1.0, null_certified=True) for _ in range(10)]
    assert all(n.target_id is None and "p" not in n.candidate_ids for n in nulls)


def test_merge_null_equivalent(tok):
    ex = build_choice(tok, "q", [("a", "w1"), ("none", "w2"), ("b", "w3")], {"none": 0.7, "a": 0.3})
    m = merge_null_equivalent(ex, ["none"])
    assert [c.cid for c in m.candidates] == ["a", "b"]
    assert m.target.tolist() == pytest.approx([0.7, 0.3, 0.0])
    assert REFERENCE_ID not in [c.cid for c in m.candidates]


def test_mixture_trace():
    ids = {b: [f"{b}_{i}" for i in range(50)] for b in ("openjev_choice", "openjev_noul", "openjev_score", "toolret_choice")}
    tr = build_trace(ids, num_updates=20, seed=0)
    assert len(tr) == 20 and all(len(b) == 32 for b in tr)
    st = trace_stats(tr)
    assert all(v["presentations"] == 160 and v["unique"] == 50 for v in st.values())
    assert build_trace(ids, num_updates=5, seed=0) == tr[:5]


def test_metrics():
    assert len(permutations_for(3)) == 6
    assert len(permutations_for(6)) == 11
    assert decision_flip_rate([["a", "a"], ["a", "b"]]) == 0.5
    assert max_probability_drift([{"a": 0.5}, {"a": 0.7}]) == pytest.approx(0.2)
    assert hit_at_1(["a", "b"], ["a", "c"]) == 0.5


def test_complexity_pruning_reduces_later_passes():
    full = diffjev_cost(1024, 64, 32, 4096, 32)
    pruned = diffjev_cost(1024, 64, 32, 4096, 32, s=32, u=8)
    assert pruned < full
    assert serialized_dlm_cost(1000, 64, 2) > 0

