import random

import pytest
import torch

from diffjev.data.synthetic import synthetic_choice
from diffjev.packing import Candidate, pack_branches


def test_tied_set_rope_positions(tok):
    q = [5, 6, 7]
    cands = [Candidate("a", [8]), Candidate("b", [9, 10, 11]), Candidate("c", [12, 13])]
    p = pack_branches(q, cands, mask_token_id=1, pad_token_id=0)
    assert p.query_pos.tolist() == [0, 1, 2]
    # every slot at L_q (0-indexed == L_q + 1 one-indexed)
    assert p.slot_pos.tolist() == [3, 3, 3]
    # option positions restart at L_q + 1 (0-indexed) for every branch
    for i, n in enumerate([1, 3, 2]):
        assert p.branch_pos[i, 1 : 1 + n].tolist() == list(range(4, 4 + n))
        assert p.branch_valid[i].sum() == 1 + n
    assert (p.branch_ids[:, 0] == 1).all()  # slot token is [MASK]
    assert p.branch_ids[0, 2:].tolist() == [0, 0]  # padding


def test_sequential_positions_are_untied(tok):
    cands = [Candidate("a", [8]), Candidate("b", [9, 10])]
    p = pack_branches([5, 6], cands, 1, 0, position_mode="sequential")
    assert p.slot_pos.tolist() == [2, 4]


def test_aliases_are_metadata_only(tok):
    cands1 = [Candidate("a", [8], alias="A"), Candidate("b", [9], alias="B")]
    cands2 = [Candidate("a", [8], alias="Z"), Candidate("b", [9], alias="Q")]
    p1 = pack_branches([5], cands1, 1, 0)
    p2 = pack_branches([5], cands2, 1, 0)
    assert torch.equal(p1.branch_ids, p2.branch_ids)
    assert torch.equal(p1.branch_pos, p2.branch_pos)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        pack_branches([5], [Candidate("a", [8]), Candidate("a", [9])], 1, 0)


def test_select_preserves_identity(tok):
    ex = synthetic_choice(tok, random.Random(0), 5)
    from diffjev.packing import pack_branches as pb

    p = pb(ex.query_ids, ex.candidates, tok.mask_token_id, tok.pad_token_id)
    sub = p.select([3, 0])
    assert sub.candidate_ids == [p.candidate_ids[3], p.candidate_ids[0]]
    assert torch.equal(sub.branch_ids[0], p.branch_ids[3])

