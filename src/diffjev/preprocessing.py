"""Shared semantic content policy; every runner tokenizes it independently."""
from dataclasses import replace
from .packing import Candidate
from .tasks import _enc


def retokenize_example(example, tokenizer, max_option_len=None):
    """Tokenization belongs inside a measured full request.

    Formal runs preserve the pinned semantic content and report context failures.
    Token truncation here is only for same-tokenizer diagnostics and is recorded.
    """
    if tokenizer is None or "query_text" not in example.meta:
        if max_option_len is None:
            return example
        return replace(example, candidates=[replace(c, token_ids=c.token_ids[:max_option_len]) for c in example.candidates])
    candidates = []
    for c in example.candidates:
        if c.text is None:
            raise ValueError("Missing candidate semantic text")
        tokens = _enc(tokenizer, c.text)
        candidates.append(Candidate(c.cid, tokens[:max_option_len] if max_option_len else tokens, c.alias, c.text))
    return replace(example, query_ids=_enc(tokenizer, example.meta["query_text"]), candidates=candidates)


def runner_tokenizer(runner):
    return getattr(runner, "tokenizer", None) or getattr(getattr(runner, "backbone", None), "tokenizer", None)
