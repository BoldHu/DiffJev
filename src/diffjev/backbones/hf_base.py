"""Shared implementation for HuggingFace diffusion-LM adapters (LLaDA, Nemotron).

Provides:
- Local-only loading from revision-pinned directories (never triggers unrequested
  network downloads during local tests).
- Strict single-token Yes/No verbalizer resolution.
- Block-level introspection and execution of ``layer_kv`` and ``layer_forward``
  that:
    1. Apply RoPE and any position-dependent query scaling using DiffJev's
       explicit **semantic** positions ``pos`` (keeping KV storage addresses
       separate from repeated slot positions).
    2. Route DiffJev's explicit boolean visibility mask ``allowed`` directly
       into the attention computation so native paths that discard masks are
       bypassed while reusing the exact pretrained weights.
- ``verify_adapter()`` checklist required by AGENTS.md section 2.0 before any
  real backbone is used in experiments.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import BackboneAdapter, resolve_single_token
from .rope import apply_rope


def _resolve_attr(root: nn.Module, candidates: Sequence[str]) -> nn.Module:
    for path in candidates:
        obj = root
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and obj is not None:
            return obj
    raise AttributeError(
        f"Could not resolve any of {list(candidates)} on {type(root).__name__}"
    )


@dataclass
class BlockParts:
    """Resolved submodules and geometry of one pretrained transformer block."""

    attn_norm: nn.Module
    q_proj: Optional[nn.Module]
    k_proj: Optional[nn.Module]
    v_proj: Optional[nn.Module]
    fused_qkv: Optional[nn.Module]
    o_proj: nn.Module
    q_norm: Optional[nn.Module]
    k_norm: Optional[nn.Module]
    ffn_norm: nn.Module
    gate_proj: nn.Module
    up_proj: nn.Module
    down_proj: nn.Module
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    pos_attn_scale_base: Optional[float] = None


class HFDiffusionAdapterBase(BackboneAdapter):
    """Base adapter for pretrained HuggingFace diffusion language models."""

    #: Candidate dotted paths for each structural component (first match wins)
    PATH_CANDIDATES: Dict[str, Tuple[str, ...]] = {
        "embed": (
            "model.transformer.wte",
            "encoder.embed_tokens",
            "model.embed_tokens",
            "transformer.wte",
            "embed_tokens",
        ),
        "layers": (
            "model.transformer.blocks",
            "encoder.layers",
            "model.layers",
            "transformer.blocks",
            "layers",
        ),
        "final_norm": (
            "model.transformer.ln_f",
            "encoder.norm",
            "model.norm",
            "transformer.ln_f",
            "norm",
        ),
        "head": (
            "model.transformer.ff_out",
            "diffusion_head",
            "lm_head",
            "transformer.ff_out",
        ),
    }
    YES_CANDIDATES: Sequence[str] = ("Yes", " Yes")
    NO_CANDIDATES: Sequence[str] = ("No", " No")
    DEFAULT_MASK_TOKEN_ID: Optional[int] = None
    MODEL_ID: str = ""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer,
        yes: Optional[str] = None,
        no: Optional[str] = None,
        mask_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.yes_token_id = self._resolve_verbalizer(yes, self.YES_CANDIDATES)
        self.no_token_id = self._resolve_verbalizer(no, self.NO_CANDIDATES)
        if self.yes_token_id == self.no_token_id:
            raise ValueError("Yes and No verbalizers resolved to the same token id")
        self.mask_token_id = self._resolve_mask_token_id(mask_token_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(getattr(hf_model, "config", None), "pad_token_id", None)
        self.pad_token_id = int(pad_id) if pad_id is not None else self.mask_token_id
        self._block_cache: Dict[int, BlockParts] = {}

    def _resolve_verbalizer(self, explicit: Optional[str], candidates: Sequence[str]) -> int:
        if explicit is not None:
            return resolve_single_token(self.tokenizer, explicit)
        errors = []
        for c in candidates:
            try:
                return resolve_single_token(self.tokenizer, c)
            except ValueError as e:
                errors.append(str(e))
        raise ValueError("no single-token verbalizer found: " + "; ".join(errors))

    def _resolve_mask_token_id(self, explicit: Optional[int]) -> int:
        if explicit is not None:
            return int(explicit)
        tok_mask = getattr(self.tokenizer, "mask_token_id", None)
        if tok_mask is not None:
            return int(tok_mask)
        cfg = getattr(self.hf_model, "config", None)
        cfg_mask = getattr(cfg, "mask_token_id", None) if cfg is not None else None
        if cfg_mask is not None:
            return int(cfg_mask)
        if self.DEFAULT_MASK_TOKEN_ID is not None:
            vocab_n = int(self.input_embedding_module().weight.shape[0])
            if 0 <= self.DEFAULT_MASK_TOKEN_ID < vocab_n:
                return int(self.DEFAULT_MASK_TOKEN_ID)
        raise ValueError(
            "Could not resolve mask_token_id from tokenizer or config; pass mask_token_id explicitly."
        )

    # ---- loading from pinned local snapshot ----
    @classmethod
    def from_local(
        cls,
        local_dir: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map=None,
        **kwargs,
    ) -> "HFDiffusionAdapterBase":
        """Load from an already-downloaded, revision-pinned local directory.

        Always passes ``local_files_only=True`` so it never triggers an implicit
        network download.
        """
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(
                f"Local model directory {local_dir!r} does not exist. "
                "Download the pinned revision first via scripts/download_assets.py (see README.md)."
            )
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        try:
            model = AutoModel.from_pretrained(
                local_dir,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        return cls(model, tok, **kwargs)

    # ---- structural accessors ----
    def _resolve_component(self, key: str) -> nn.Module:
        return _resolve_attr(self.hf_model, self.PATH_CANDIDATES[key])

    @property
    def num_layers(self) -> int:
        return len(self.layer_modules())

    @property
    def hidden_size(self) -> int:
        return int(self.input_embedding_module().weight.shape[1])

    def layer_modules(self) -> List[nn.Module]:
        return list(self._resolve_component("layers"))

    def final_norm_module(self) -> nn.Module:
        return self._resolve_component("final_norm")

    def input_embedding_module(self) -> nn.Module:
        return self._resolve_component("embed")

    def output_head_module(self) -> nn.Module:
        try:
            return self._resolve_component("head")
        except AttributeError:
            # Tied output head fallback when weight tying reuses input embeddings
            return self.input_embedding_module()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.input_embedding_module()(ids)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return self.final_norm_module()(h)

    # ---- block introspection & primitives ----
    def _inspect_block(self, layer: int) -> BlockParts:
        if layer in self._block_cache:
            return self._block_cache[layer]
        blk = self.layer_modules()[layer]
        cfg = getattr(self.hf_model, "config", None)

        attn_mod = getattr(blk, "self_attn", getattr(blk, "attn", blk))
        mlp_mod = getattr(blk, "mlp", getattr(blk, "ffn", blk))

        attn_norm = _resolve_attr(
            blk, ("attn_norm", "input_layernorm", "ln_1", "norm1", "operator_norm")
        )
        ffn_norm = _resolve_attr(
            blk,
            (
                "ff_norm",
                "ffn_norm",
                "post_attention_layernorm",
                "ln_2",
                "norm2",
                "pre_ff_layernorm",
            ),
        )

        q_proj = getattr(attn_mod, "q_proj", None)
        k_proj = getattr(attn_mod, "k_proj", None)
        v_proj = getattr(attn_mod, "v_proj", None)
        fused_qkv = getattr(attn_mod, "att_proj", getattr(attn_mod, "qkv_proj", None))
        if q_proj is None and fused_qkv is None:
            raise AttributeError(f"Layer {layer} has neither q_proj nor att_proj/qkv_proj")

        o_proj = _resolve_attr(attn_mod, ("o_proj", "attn_out", "out_proj", "dense"))
        q_norm = getattr(attn_mod, "q_norm", None)
        k_norm = getattr(attn_mod, "k_norm", None)

        gate_proj = _resolve_attr(mlp_mod, ("gate_proj", "ff_proj", "w1"))
        up_proj = _resolve_attr(mlp_mod, ("up_proj", "w3"))
        down_proj = _resolve_attr(mlp_mod, ("down_proj", "ff_out", "w2"))

        num_heads = int(
            getattr(attn_mod, "num_heads", None)
            or getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_heads", None)
            or 32
        )
        num_kv_heads = int(
            getattr(attn_mod, "num_key_value_heads", None)
            or getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "n_kv_heads", None)
            or num_heads
        )
        if q_proj is not None:
            head_dim = int(q_proj.weight.shape[0] // num_heads)
        else:
            total_out = int(fused_qkv.weight.shape[0])
            head_dim = int(total_out // (num_heads + 2 * num_kv_heads))

        rope_theta = float(
            getattr(attn_mod, "rope_theta", None)
            or getattr(cfg, "rope_theta", None)
            or 10000.0
        )
        scale_base = getattr(cfg, "attn_scale_base", getattr(attn_mod, "attn_scale_base", None))
        parts = BlockParts(
            attn_norm=attn_norm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            fused_qkv=fused_qkv,
            o_proj=o_proj,
            q_norm=q_norm,
            k_norm=k_norm,
            ffn_norm=ffn_norm,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            pos_attn_scale_base=float(scale_base) if scale_base is not None else None,
        )
        self._block_cache[layer] = parts
        return parts

    def _project_qkv(
        self, parts: BlockParts, x: torch.Tensor, need_q: bool = True
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        B, n, _ = x.shape
        H, Hkv, dh = parts.num_heads, parts.num_kv_heads, parts.head_dim
        if parts.fused_qkv is not None:
            qkv = parts.fused_qkv(x)
            q_dim = H * dh
            kv_dim = Hkv * dh
            q_raw, k_raw, v_raw = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            q = q_raw.view(B, n, H, dh).transpose(1, 2) if need_q else None
        else:
            q = parts.q_proj(x).view(B, n, H, dh).transpose(1, 2) if need_q else None
            k_raw = parts.k_proj(x)
            v_raw = parts.v_proj(x)
        k = k_raw.view(B, n, Hkv, dh).transpose(1, 2)
        v = v_raw.view(B, n, Hkv, dh).transpose(1, 2)
        if q is not None and parts.q_norm is not None:
            q = parts.q_norm(q)
        if parts.k_norm is not None:
            k = parts.k_norm(k)
        return q, k, v

    def _apply_semantic_query_scaling(
        self, parts: BlockParts, q: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Apply any position-dependent query scaling strictly from semantic ``pos``.

        In Nemotron-Labs-Diffusion, long-context query scaling depends on
        ``cache_position`` in the native implementation. Deriving it from
        ``pos`` resets the factor consistently for every branch and slot.
        """
        if parts.pos_attn_scale_base is None:
            return q
        scale = (
            1.0
            + torch.log(pos.to(torch.float64).clamp(min=1.0) + 1.0)
            / math.log(parts.pos_attn_scale_base)
        ).to(q.dtype)
        return q * scale[:, None, :, None]

    def _rope(self, x, pos, parts):
        return apply_rope(x, pos, parts.rope_theta)

    def attention_visibility(self, n, device, causal=False):
        return "causal" if causal else None

    def _activation(self, layer, x):
        block = self.layer_modules()[layer]
        mlp = getattr(block, "mlp", getattr(block, "ffn", block))
        return getattr(mlp, "act_fn", F.silu)(x)

    def _project_query(self, parts, x):
        # KV is supplied by the caller; do not project it a second time.
        if parts.q_proj is None:
            return self._project_qkv(parts, x, need_q=True)[0]
        q = parts.q_proj(x).view(x.shape[0], x.shape[1], parts.num_heads,
                                    parts.head_dim).transpose(1, 2)
        return parts.q_norm(q) if parts.q_norm is not None else q

    def layer_kv(
        self, layer: int, h: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = self._inspect_block(layer)
        x = parts.attn_norm(h)
        _, k, v = self._project_qkv(parts, x, need_q=False)
        k = self._rope(k, pos, parts)
        return k, v

    def layer_forward(
        self,
        layer: int,
        h: torch.Tensor,
        pos: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        parts = self._inspect_block(layer)
        B, n, _ = h.shape
        x = parts.attn_norm(h)
        q = self._project_query(parts, x)
        assert q is not None
        q = self._rope(q, pos, parts)
        q = self._apply_semantic_query_scaling(parts, q, pos)

        rep = parts.num_heads // parts.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None, :, :] if isinstance(allowed, torch.Tensor) else None,
            dropout_p=0.0, is_causal=isinstance(allowed, str) and allowed == "causal",
        ).transpose(1, 2).reshape(B, n, parts.num_heads * parts.head_dim)
        h = h + parts.o_proj(o)

        y = parts.ffn_norm(h)
        h = h + parts.down_proj(self._activation(layer, parts.gate_proj(y)) * parts.up_proj(y))
        return h

    # ---- adapter verification checklist (AGENTS.md section 2.0) ----
    @torch.no_grad()
    def verify_native_parity(self):
        """Compare an unmasked native block with our primitive execution.

        Nemotron is additionally checked beyond the original context length, where
        both YaRN and query scaling matter. Run before any formal use of a snapshot.
        """
        import inspect
        self.eval()
        block = self.layer_modules()[0]
        signature = inspect.signature(block.forward).parameters
        device = self.input_embedding_module().weight.device
        ids = torch.arange(4, device=device)[None] + 10
        h = self.embed(ids)
        offsets = [0]
        params = getattr(self.hf_model.config, "rope_parameters", None)
        if params and "llama_4_scaling_beta" in params:
            offsets += [params["original_max_position_embeddings"] * 2]
        drift = {}
        for offset in offsets:
            pos = torch.arange(4, device=device)[None] + offset
            kw = {}
            if "position_ids" in signature:
                kw["position_ids"] = pos
            if "cache_position" in signature:
                kw["cache_position"] = pos[0]
            if "position_embeddings" in signature:
                enc = getattr(self.hf_model, "encoder", getattr(self.hf_model, "model", None))
                kw["position_embeddings"] = enc.rotary_emb(h, pos)
            if "attention_bias" in signature:
                kw["attention_bias"] = torch.zeros(1, 1, 4, 4, device=device, dtype=h.dtype)
            if "attention_mask" in signature:
                kw["attention_mask"] = torch.zeros(1, 1, 4, 4, device=device, dtype=h.dtype)
            native = block(h, **kw)
            if isinstance(native, tuple):
                native = native[0]
            k, v = self.layer_kv(0, h, pos)
            ours = self.layer_forward(0, h, pos, k, v, torch.ones(1,4,4,device=device,dtype=torch.bool))
            error = float((ours.float()-native.float()).abs().max())
            drift[str(offset)] = error
            tol = 0.05 if h.dtype in (torch.bfloat16, torch.float16) else 2e-5
            if not torch.allclose(ours.float(), native.float(), atol=tol, rtol=tol):
                raise RuntimeError(f"Native block parity failed at offset {offset}: {error}")
        return {"max_abs_drift_by_position_offset": drift, "dtype": str(h.dtype)}

    @torch.no_grad()
    def verify_adapter(self) -> Dict[str, bool]:
        """Run the mandatory adapter verification suite before using the model."""
        emb = self.input_embedding_module()
        vocab_size, _ = emb.weight.shape
        device = emb.weight.device

        # 1) Single-token verbalizers and mask token validity
        verbalizers_ok = (
            0 <= self.yes_token_id < vocab_size
            and 0 <= self.no_token_id < vocab_size
            and self.yes_token_id != self.no_token_id
            and 0 <= self.mask_token_id < vocab_size
        )

        # 2) No forbidden new parameters or LoRA modules loaded
        hf_ids = {id(p) for p in self.hf_model.parameters()}
        self_ids = {id(p) for p in self.parameters()}
        no_extra_params = self_ids == hf_ids
        no_lora = not any("lora" in name.lower() for name, _ in self.hf_model.named_parameters())

        # 3) Explicit visibility mask respected by layer_forward
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device) % vocab_size
        pos = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        h0 = self.embed(ids)
        k0, v0 = self.layer_kv(0, h0, pos)
        allowed = torch.tensor(
            [[[True, True, False], [True, True, False], [True, True, True]]],
            dtype=torch.bool,
            device=device,
        )
        out_a = self.layer_forward(0, h0, pos, k0, v0, allowed)
        k_pert, v_pert = k0.clone(), v0.clone()
        k_pert[:, :, 2, :] += 10.0
        v_pert[:, :, 2, :] -= 10.0
        out_b = self.layer_forward(0, h0, pos, k_pert, v_pert, allowed)
        mask_respected = bool(torch.allclose(out_a[:, :2], out_b[:, :2], atol=1e-5))

        # 4) Semantic position reset (KV storage address decoupled from semantic pos)
        h_two = h0[:, :2].clone()
        h_two[:, 1] = h_two[:, 0]
        tied_pos = torch.tensor([[5, 5]], dtype=torch.long, device=device)
        kt, _ = self.layer_kv(0, h_two, tied_pos)
        pos_reset_ok = bool(torch.allclose(kt[:, :, 0], kt[:, :, 1], atol=1e-6))

        # 5) Restricted Yes/No head matches full-vocabulary head difference
        h_norm = self.final_norm(out_a)
        z_restricted = self.yes_no_logit_diff(h_norm)
        full = self.full_logits(h_norm)
        z_full = full[..., self.yes_token_id] - full[..., self.no_token_id]
        tol = 5e-2 if h0.dtype in (torch.float16, torch.bfloat16) else 1e-5
        head_match = bool(torch.allclose(z_restricted, z_full, atol=tol))

        report = {
            "single_token_verbalizers": verbalizers_ok,
            "no_extra_parameters": no_extra_params,
            "no_lora_loaded": no_lora,
            "explicit_mask_kernel": mask_respected,
            "semantic_position_reset": pos_reset_ok,
            "restricted_head_exact": head_match,
        }
        if not all(report.values()):
            failed = [k for k, v in report.items() if not v]
            raise RuntimeError(f"{type(self).__name__} failed verification checks: {failed}")
        return report
