import pytest
import torch

from diffjev.backbones.mock import MockConfig, build_mock_backbone, default_mock_tokenizer
from diffjev.config import DiffJevConfig
from diffjev.model import DiffJevModel

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tok():
    return default_mock_tokenizer()


def make_backbone(tok, dtype=torch.float64, num_layers=6, seed=0, **kw):
    cfg = MockConfig(num_layers=num_layers, seed=seed, **kw)
    return build_mock_backbone(cfg, tok, dtype=dtype)


def make_model(tok, dtype=torch.float64, num_layers=6, seed=0, **cfg_kw):
    bb = make_backbone(tok, dtype=dtype, num_layers=num_layers, seed=seed)
    return DiffJevModel(bb, DiffJevConfig(**cfg_kw))


@pytest.fixture
def model(tok):
    return make_model(tok)

