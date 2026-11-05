"""Baselines for DiffJev (AGENTS.md section 7 and Exp 5):

- ``SerializedDLMBaseline``     : serialized 1-pass DLM with 1 Yes/No slot per candidate
- ``AutoregressiveJevBaseline`` : causal LLM-as-Jev / exact-checkpoint Nemotron AR
                                  (1-prefill cached KV, complete multi-token label scoring)
- ``BM25Retriever``             : frozen Okapi BM25 hard-negative ranker and baseline
- ``DenseBiEncoderBaseline``    : BGE / E5 dense bi-encoder retrieval baseline
- ``JevAPIBaseline``            : commercial Jev API client (API latency separated)
- ``JevaBaseline``              : single-pass set baseline Jeva-4B
- ``OptionalDjevBaseline``      : optional djev baseline (requires revision verification)
"""

from .ar_llm import AutoregressiveJevBaseline, CausalSerializedPrompt, pack_causal_prompt
from .base import BaselineAdapter, BaselineResult
from .external import JevAPIBaseline, JevaBaseline, OptionalDjevBaseline
from .retrieval import BM25Retriever, DenseBiEncoderBaseline
from .serialized_dlm import SerializedDLMBaseline, SerializedPrompt, pack_serialized_dlm

__all__ = [
    "BaselineAdapter",
    "BaselineResult",
    "SerializedDLMBaseline",
    "SerializedPrompt",
    "pack_serialized_dlm",
    "AutoregressiveJevBaseline",
    "CausalSerializedPrompt",
    "pack_causal_prompt",
    "BM25Retriever",
    "DenseBiEncoderBaseline",
    "JevAPIBaseline",
    "JevaBaseline",
    "OptionalDjevBaseline",
]

