"""Offline regressions for chat-template return contracts; no model downloads."""
from collections import UserDict

import pytest
import torch

from diffjev.baselines.ar_llm import pack_causal_prompt
from diffjev.data.context_audit import TrainingContextAudit
from diffjev.tasks import build_choice


class BatchEncodingFixture(UserDict):
    """Model HF's string-key access and its distinct integer-index behavior."""
    def __getitem__(self, key):
        if isinstance(key, int):
            raise AssertionError("BatchEncoding was indexed as a token sequence")
        return super().__getitem__(key)


class ChatTokenizer:
    chat_template = "offline test template"
    eos_token_id = 101

    def __init__(self, base, result):
        self.base, self.result = base, result

    def encode(self, text, **kwargs):
        return self.base.encode(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return self.result


@pytest.mark.parametrize("kind", ["batch_encoding", "dict", "flat", "nested", "tensor", "batched_tensor"])
def test_chat_prompt_preserves_exact_ids_and_complete_answer_labels(tok, kind):
    ids = [7, 9, 11, 13]
    values = {
        "batch_encoding": BatchEncodingFixture(input_ids=ids, attention_mask=[1]*4),
        "dict": {"input_ids": ids}, "flat": ids, "nested": [ids],
        "tensor": torch.tensor(ids), "batched_tensor": {"input_ids": torch.tensor([ids])},
    }
    tokenizer = ChatTokenizer(tok, values[kind])
    example = build_choice(tokenizer, "query", [("a", "w1"), ("b", "w2")],
                           aliases=["[12]", "[34]"])
    packed = pack_causal_prompt(example, tokenizer, 3, 4, torch.device("cpu"))
    assert packed.prompt_ids.dtype == torch.long
    assert packed.prompt_ids.tolist() == [ids]
    assert packed.position_ids.tolist() == [[0, 1, 2, 3]]
    assert packed.candidate_ids == ["a", "b"]
    assert packed.label_token_seqs == [tok.encode(s) + [101] for s in ["[12]", "[34]"]]
    assert packed.ref_token_seq == tok.encode("[0]") + [101]
    assert tokenizer.kwargs == dict(tokenize=True, return_dict=True, padding=False,
        truncation=False, add_generation_prompt=True, enable_thinking=False)
    assert "[12]: w1" in tokenizer.messages[0]["content"]


@pytest.mark.parametrize("ids", [[], [[1], [2]], [1.5], [True], [-1]])
def test_invalid_chat_ids_fail_without_truncating_or_flattening_requests(tok, ids):
    tokenizer = ChatTokenizer(tok, {"input_ids": ids})
    example = build_choice(tokenizer, "query", [("a", "w1")])
    with pytest.raises(ValueError, match="one nonempty sequence of integer input_ids"):
        pack_causal_prompt(example, tokenizer, 3, 4, torch.device("cpu"))


def test_context_audit_counts_prompt_ids_and_complete_answer(tok):
    tokenizer = ChatTokenizer(tok, BatchEncodingFixture(input_ids=list(range(100))))
    audit = TrainingContextAudit.__new__(TrainingContextAudit)
    record = {"id":"fixture", "source":"openjev", "mode":"choice", "query":"query",
              "options":[("a","w1"),("b","w2")], "gold_id":"a", "target":{"a":1.0}}
    # Each label is one digit + terminator + EOS: 100 prompt + 3 answer tokens.
    audit.entries = [("nemotron_3b", tokenizer, 103)]
    assert audit.check(record) is None
    audit.entries = [("nemotron_3b", tokenizer, 102)]
    assert audit.check(record) == "nemotron_3b:ar:103>102"
