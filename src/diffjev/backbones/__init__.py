"""Backbone adapters.

- ``base.BackboneAdapter``              : interface used by all core DiffJev modules
- ``mock.MockBackbone``                 : tiny random transformer (local tests)
- ``hf_base.HFDiffusionAdapterBase``    : shared HF diffusion adapter implementation
- ``llada.LLaDAAdapter``                : GSAI-ML/LLaDA-8B-Base adapter
- ``nemotron.NemotronDiffusionAdapter`` : nvidia/Nemotron-Labs-Diffusion 3B/8B/14B adapter
"""

from .base import BackboneAdapter, VerbalizerError, resolve_single_token
from .hf_base import BlockParts, HFDiffusionAdapterBase
from .llada import LLaDAAdapter
from .mock import MockBackbone, MockConfig, MockTokenizer, build_mock_backbone, default_mock_tokenizer
from .nemotron import NEMOTRON_CHECKPOINTS, NemotronDiffusionAdapter, strip_optional_lora

__all__ = [
    "BackboneAdapter",
    "VerbalizerError",
    "resolve_single_token",
    "BlockParts",
    "HFDiffusionAdapterBase",
    "LLaDAAdapter",
    "NEMOTRON_CHECKPOINTS",
    "NemotronDiffusionAdapter",
    "strip_optional_lora",
    "MockBackbone",
    "MockConfig",
    "MockTokenizer",
    "build_mock_backbone",
    "default_mock_tokenizer",
]
