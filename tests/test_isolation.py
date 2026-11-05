"""Attention isolation and exact chunking of the lower-layer branch encoder."""

import random

import torch

from diffjev.data.synthetic import synthetic_choice
from diffjev.encoder import build_query_memory, encode_branches, extend_option_memory
from diffjev.masks import lower_layer_dense_mask
from diffjev.packing import Candidate, pack_branches, to_dense_layout
from tests.conftest import make_backbone


def _packed(tok, K=5, seed=0):
    ex = synthetic_choice(tok, random.Random(seed), K, max_len=6)
    return pack_branches(ex.query_ids, ex.candidates, tok.mask_token_id, tok.pad_token_id)


def test_dense_mask_rules():
    seg = torch.tensor([0, 0, 1, 1, 2, -1])
    m = lower_layer_dense_mask(seg)
    assert m[0].tolist() == [True, True, False, False, False, False]  # query -> query only
    assert m[2].tolist() == [True, True, True, True, False, False]  # branch 1 -> query + own
    assert m[4].tolist() == [True, True, False, False, True, False]  # branch 2
    assert not m[:5, 5].any()  # padding excluded as key


def test_chunked_encoder_matches_dense_reference(tok):
    bb = make_backbone(tok)
    packed = _packed(tok)
    k_top = 4
    lower = bb.lower_layer_indices(k_top)
    lay = to_dense_layout(packed, tok.pad_token_id, n_pad=3)
    allowed = lower_layer_dense_mask(lay.segment_ids)[None]
    h = bb.run_layers_dense(bb.embed(lay.input_ids)[None], lay.position_ids[None], allowed, lower)[0]
    for chunk in (1, 2, 64):
        bd = encode_branches(bb, packed, k_top, chunk_size=chunk)
        assert torch.allclose(bd.query_h, h[: packed.query_len], atol=1e-10)
        assert torch.allclose(bd.h_slot_base, h[lay.slot_index], atol=1e-10)


def test_branch_isolation_under_perturbation(tok):
    bb = make_backbone(tok)
    q = [10, 11, 12]
    a = [Candidate("a", [13, 14]), Candidate("b", [15]), Candidate("c", [16, 17, 18])]
    b = [Candidate("a", [13, 14]), Candidate("b", [20, 21, 22, 23]), Candidate("c", [16, 17, 18])]
    pa = pack_branches(q, a, tok.mask_token_id, tok.pad_token_id)
    pb = pack_branches(q, b, tok.mask_token_id, tok.pad_token_id)
    ba = encode_branches(bb, pa, 4)
    bbd = encode_branches(bb, pb, 4)
    # query never reads branches; other branches unaffected
    assert torch.allclose(ba.query_h, bbd.query_h, atol=1e-12)
    for i in (0, 2):
        assert torch.allclose(ba.h_slot_base[i], bbd.h_slot_base[i], atol=1e-12)
    assert not torch.allclose(ba.h_slot_base[1], bbd.h_slot_base[1])


def test_static_memory_matches_dense_and_lazy_build(tok):
    bb = make_backbone(tok)
    packed = _packed(tok, K=4, seed=3)
    bd = encode_branches(bb, packed, 4)
    top = bb.top_layer_indices(4)
    mem_all = extend_option_memory(bb, bd, build_query_memory(bb, bd, 4), [0, 1, 2, 3], 4)
    mem_lazy = build_query_memory(bb, bd, 4)
    extend_option_memory(bb, bd, mem_lazy, [2], 4, chunk_size=1)
    extend_option_memory(bb, bd, mem_lazy, [0, 2, 3, 1], 4, chunk_size=2)
    assert mem_lazy.build_log == [[2], [0, 3, 1]]
    for j in range(4):
        for li in range(len(top)):
            assert torch.allclose(mem_all.option_k[j][li], mem_lazy.option_k[j][li], atol=1e-12)
    # dense recomputation of the static path for candidate j
    q = bd.query_h.shape[0]
    for j in range(4):
        n = int(bd.option_valid[j].sum())
        h = torch.cat([bd.query_h, bd.option_h[j, :n]])[None]
        pos = torch.cat([bd.query_pos, bd.option_pos[j, :n]])[None]
        allowed = torch.zeros(1, q + n, q + n, dtype=torch.bool)
        allowed[0, :q, :q] = True
        allowed[0, q:, :] = True
        for li, l in enumerate(top):
            k, v = bb.layer_kv(l, h, pos)
            assert torch.allclose(k[0, :, :q], mem_all.query_k[li][0], atol=1e-10)
            assert torch.allclose(k[0, :, q:], mem_all.option_k[j][li][:, :n], atol=1e-10)
            assert torch.allclose(v[0, :, q:], mem_all.option_v[j][li][:, :n], atol=1e-10)
            h = bb.layer_forward(l, h, pos, k, v, allowed)


def test_slot_never_reads_other_option_detail(tok):
    """Without cross-slot attention, slot i's z (memory pass) is invariant to option j."""
    from diffjev.config import DiffJevConfig
    from diffjev.denoiser import SlotDenoiser
    from diffjev.diffusion import initial_soft_state

    bb = make_backbone(tok)
    cfg = DiffJevConfig(cross_slot_attention=False)
    den = SlotDenoiser(bb, cfg)
    q = [10, 11]
    a = [Candidate("a", [13, 14]), Candidate("b", [15]), Candidate("c", [16])]
    b = [Candidate("a", [13, 14]), Candidate("b", [30, 31, 32]), Candidate("c", [16])]
    zs = []
    for cands in (a, b):
        bd = encode_branches(bb, pack_branches(q, cands, tok.mask_token_id, tok.pad_token_id), 4)
        act = torch.arange(3)
        mem = den.ensure_memory(bd, None, act)
        x = den.boundary_inputs(bd, act, initial_soft_state(3, dtype=torch.float64))
        zs.append(den.slot_pass(x, bd, act, mem))
    assert torch.allclose(zs[0][[0, 2]], zs[1][[0, 2]], atol=1e-12)

