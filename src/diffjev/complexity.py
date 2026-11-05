"""Analytic cost model and operator-shape GFLOPs estimator (AGENTS.md Exp 5 contract).

1. Asymptotic ``Phi(n, a)`` formulation from ``AGENTS.md`` lines 226-238:
    Phi(n, a) = n * d^2 + a * d
    B(K)      = Phi(q + K*b, q^2 + K*b*(q + b))            lower-layer branches
    M(s)      = Phi(q + s*m, q^2 + s*m*(q + m))            top-layer static memory
    S(n, dl)  = Phi(n, n^2 + dl * n * (q + m))             one slot pass
    C_DiffJev = (L-k)*B(K) + k*M(s) + k*(S(K,0) + S(s,1) + S(u,1)) + C_head

These are asymptotic cost units, not FLOPs or measured speedups.
Executed-operation accounting lives in evaluation/operations.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence


def phi(n: float, a: float, d: float) -> float:
    return n * d * d + a * d


def branch_cost(K: int, q: int, m: int, d: int) -> float:
    b = m + 1
    return phi(q + K * b, q * q + K * b * (q + b), d)


def memory_cost(s: int, q: int, m: int, d: int) -> float:
    return phi(q + s * m, q * q + s * m * (q + m), d)


def slot_pass_cost(n: int, delta: int, q: int, m: int, d: int) -> float:
    return phi(n, n * n + delta * n * (q + m), d)


def diffjev_cost(
    K: int,
    q: int,
    m: int,
    d: int,
    L: int,
    k: int = 4,
    s: Optional[int] = None,
    u: Optional[int] = None,
    head: float = 0.0,
) -> float:
    s_eff = K if s is None else s
    u_eff = K if u is None else u
    return (
        (L - k) * branch_cost(K, q, m, d)
        + k * memory_cost(s_eff, q, m, d)
        + k
        * (
            slot_pass_cost(K, 0, q, m, d)
            + slot_pass_cost(s_eff, 1, q, m, d)
            + slot_pass_cost(u_eff, 1, q, m, d)
        )
        + head
    )


def serialized_dlm_cost(N: int, d: int, L: int) -> float:
    return L * phi(N, N * N, d)


def causal_ar_prefill_cost(N: int, d: int, L: int, answer_tokens: int = 1) -> float:
    prefill = L * phi(N, N * (N + 1) / 2, d)
    decode = sum(L * phi(1, N + t + 1, d) for t in range(answer_tokens))
    return prefill + decode


