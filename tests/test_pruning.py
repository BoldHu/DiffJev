"""Progressive pruning: survivor selection, ties, ID mapping, gold survival, lazy memory."""

import random

import torch

from diffjev.data.synthetic import synthetic_choice
from diffjev.pruning import pruning_active, select_survivors, top1_by_identity
from diffjev.tasks import REFERENCE_ID
from tests.conftest import make_model


def test_pruning_activation_rules():
    assert not pruning_active(32, "choice", True, 32, False)
    assert pruning_active(33, "choice", True, 32, False)
    assert not pruning_active(100, "score", True, 32, False)
    assert not pruning_active(100, "noul", True, 32, False)
    assert not pruning_active(100, "choice", True, 32, True)  # full-support run
    assert not pruning_active(100, "choice", False, 32, False)


def test_ties_at_cutoff_are_retained():
    z = torch.tensor([3.0, 1.0, 2.0, 2.0, 2.0, 0.0])
    keep = select_survivors(z, 2)
    assert keep.tolist() == [0, 2, 3, 4]  # nominal cap exceeded by exact ties
    assert select_survivors(z, 10).tolist() == list(range(6))


def test_tie_break_by_semantic_identity():
    s = torch.tensor([0.5, 0.5, 0.1])
    assert top1_by_identity(s, ["tool_b", "tool_a", "tool_c"]) == "tool_a"
    assert top1_by_identity(s, ["tool_a", "tool_b", "tool_c"]) == "tool_a"


def test_pruned_rollout_id_mapping_and_scope(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(5), 40, max_len=4)
    packed = m.pack(ex)
    out, ro = m.predict(ex, return_rollout=True)
    sizes = [len(s) for s in out.survivor_ids_per_pass]
    assert sizes[0] == 40 and sizes[1] >= 32 and sizes[2] >= 8
    assert out.probability_scope == "retained_support"
    # survivors of pass 2 are exactly the top-32 energies of pass 1, by identity
    z1 = ro.passes[0].z
    top = set(packed.candidate_ids[i] for i in torch.topk(z1, 32).indices.tolist())
    assert set(out.survivor_ids_per_pass[1]) == top
    # final ids/probabilities consistent with the final pass's active indices
    fin = ro.final
    assert out.candidate_ids == [packed.candidate_ids[i] for i in fin.active.tolist()]
    assert abs(float(out.candidate_probs.sum()) + out.reference_prob - 1.0) < 1e-6
    # survivors are nested
    assert set(out.survivor_ids_per_pass[2]) <= set(out.survivor_ids_per_pass[1])
    # static detail memory built only for survivors, lazily
    assert ro.memory.build_log[0] == ro.passes[1].active.tolist()
    assert ro.memory.build_log[1] == []
    assert len(ro.memory.built_candidates) == sizes[1]
    assert REFERENCE_ID in out.prob_by_id()


def test_gold_survival_recorded_without_rescue(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(5), 40, max_len=4)
    out0, ro = m.predict(ex, return_rollout=True)
    final_ids = set(out0.candidate_ids)
    dropped = [c.cid for c in ex.candidates if c.cid not in final_ids][0]
    out = m.predict(ex, gold_id=dropped)
    assert out.gold_survival[0] is True
    assert False in out.gold_survival
    assert dropped not in out.candidate_ids  # no rescue


def test_no_pruning_paths_keep_full_support(tok):
    m = make_model(tok)
    ex = synthetic_choice(tok, random.Random(5), 40, max_len=4)
    out = m.predict(ex, full_support=True)
    assert out.probability_scope == "full_support"
    assert len(out.candidate_ids) == 40
    m2 = make_model(tok, pruning_enabled=False)
    assert len(m2.predict(ex).candidate_ids) == 40

