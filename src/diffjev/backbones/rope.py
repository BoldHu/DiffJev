"""Rotary position embedding helpers shared by adapters.

Positions are explicit *semantic* positions (Set-RoPE repeats the slot
position across branches). Storage location of keys never enters here.
"""

from __future__ import annotations

import torch


def rope_cos_sin(pos: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """cos/sin tables of shape ``[*pos.shape, head_dim]`` (rotate-half layout)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=pos.device, dtype=torch.float64) / half))
    ang = pos.to(torch.float64)[..., None] * inv_freq  # [..., half]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply RoPE to ``x [B, H, n, d_h]`` at positions ``pos [B, n]``."""
    cos, sin = rope_cos_sin(pos, x.shape[-1], theta, x.dtype)  # [B, n, d_h]
    cos = cos[:, None]
    sin = sin[:, None]
    return x * cos + rotate_half(x) * sin

