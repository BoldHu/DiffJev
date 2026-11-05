"""Retrieval baselines and hard-negative pool ranker for Exp 3 / Exp 5.

- ``BM25Retriever``: deterministic Okapi BM25 used both as the declared frozen
  hard-negative ranker in Exp 3 / Exp 5 and as a lexical baseline.
- ``DenseBiEncoderBaseline``: dense bi-encoder retrieval baseline using a pinned
  local checkpoint (``BAAI/bge-large-en-v1.5`` or ``intfloat/e5-large-v2``) or a
  local token-embedding encoder in offline test environments.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..pruning import top1_by_identity
from ..tasks import DecisionExample
from .base import BaselineAdapter, BaselineResult


class BM25Retriever(BaselineAdapter):
    """Frozen, deterministic Okapi BM25 ranker over token IDs or whitespace tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, name: str = "bm25"):
        self.k1 = k1
        self.b = b
        self.name = name

    def score_documents(
        self, query_tokens: Sequence[int], doc_tokens_list: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        N = len(doc_tokens_list)
        if N == 0:
            return torch.zeros(0, dtype=torch.float64)
        lengths = [max(1, len(d)) for d in doc_tokens_list]
        avgdl = sum(lengths) / N
        df: Counter = Counter()
        tf_per_doc: List[Counter] = []
        for d in doc_tokens_list:
            c = Counter(d)
            tf_per_doc.append(c)
            for term in c.keys():
                df[term] += 1

        q_terms = list(query_tokens)
        scores = torch.zeros(N, dtype=torch.float64)
        for i, c in enumerate(tf_per_doc):
            dl = lengths[i]
            s = 0.0
            for term in q_terms:
                f = c.get(term, 0)
                if f == 0:
                    continue
                n_q = df.get(term, 0)
                idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                s += idf * (f * (self.k1 + 1.0)) / denom
            scores[i] = s
        return scores

    def rank_hard_negatives(
        self,
        query_tokens: Sequence[int],
        negative_pool: Sequence[Tuple[str, Sequence[int]]],
    ) -> List[str]:
        """Rank candidate negative IDs in descending BM25 score (ties broken by ID)."""
        if not negative_pool:
            return []
        docs = [toks for _, toks in negative_pool]
        scores = self.score_documents(query_tokens, docs).tolist()
        ranked = sorted(
            zip([cid for cid, _ in negative_pool], scores),
            key=lambda x: (-x[1], x[0]),
        )
        return [cid for cid, _ in ranked]

    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        t0 = time.perf_counter()
        docs = [c.token_ids for c in example.candidates]
        cids = [c.cid for c in example.candidates]
        scores = self.score_documents(example.query_ids, docs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        top1 = top1_by_identity(scores, cids)
        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=cids,
            top1=top1,
            prob_by_id=None,  # BM25 supplies uncalibrated lexical scores, not probabilities
            local_gpu_ms=elapsed_ms,
            total_tokens=len(example.query_ids) + sum(len(d) for d in docs),
        )


class DenseBiEncoderBaseline(BaselineAdapter):
    """Dense bi-encoder retrieval baseline (BGE / E5 or local embedding table)."""

    SUPPORTED_CHECKPOINTS = {
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
    }

    def __init__(
        self,
        embed_fn=None,
        name: str = "bge_biencoder",
        local_dir: Optional[str] = None,
        device: str = "cpu",
        family: str = "bge",
    ):
        self.name = name
        self.device = device
        self._hf_model = None
        self._embed_fn = embed_fn
        if family not in self.SUPPORTED_CHECKPOINTS:
            raise ValueError("retrieval family must be bge or e5")
        self.family = family
        self.tokenizer = None
        if local_dir is not None:
            if not os.path.isdir(local_dir):
                raise FileNotFoundError(
                    f"Bi-encoder directory {local_dir!r} not found. Download pinned revision first."
                )
            from transformers import AutoModel, AutoTokenizer

            self._hf_model = AutoModel.from_pretrained(local_dir, local_files_only=True).to(device)
            self._hf_model.eval()
            self._hf_model.requires_grad_(False)
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

    @torch.no_grad()
    def _encode_texts(self, texts, query=False):
        if self.family == "e5":
            texts = [("query: " if query else "passage: ") + t for t in texts]
        elif query:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        vectors = []
        for start in range(0, len(texts), 32):
            inputs = self.tokenizer(texts[start:start+32], padding=True, truncation=False,
                                    return_tensors="pt").to(self.device)
            if int(inputs["attention_mask"].sum(-1).max()) > 512:
                raise ValueError("context_limit: retrieval input exceeds native 512-token limit")
            h = self._hf_model(**inputs).last_hidden_state.float()
            if self.family == "bge":
                vec = h[:, 0]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1)
                vec = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(F.normalize(vec, p=2, dim=-1).cpu())
        return torch.cat(vectors)

    @torch.no_grad()
    def _encode_token_ids(self, token_ids_list: Sequence[Sequence[int]]) -> torch.Tensor:
        if self._embed_fn is not None:
            vecs = [self._embed_fn(torch.tensor(list(t), dtype=torch.long)).float().mean(0) for t in token_ids_list]
            mat = torch.stack(vecs, dim=0)
            return F.normalize(mat, p=2, dim=-1)
        if self._hf_model is not None:
            raise ValueError("pretrained retrieval requires raw text, not another model's token IDs")
        raise RuntimeError("DenseBiEncoderBaseline requires either embed_fn or local_dir")

    @torch.no_grad()
    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        t0 = time.perf_counter()
        cids = [c.cid for c in example.candidates]
        all_seqs = [example.query_ids] + [c.token_ids for c in example.candidates]
        if self._hf_model is not None:
            if "query_text" not in example.meta or any(c.text is None for c in example.candidates):
                raise ValueError("retrieval baseline requires original semantic text")
            try:
                embs = torch.cat([self._encode_texts([example.meta["query_text"]], query=True),
                                   self._encode_texts([c.text for c in example.candidates])])
            except ValueError as error:
                if "context_limit" not in str(error):
                    raise
                return BaselineResult(self.name, example.mode, cids, None, failed=True,
                                      failure_reason="context_limit")
        else:
            embs = self._encode_token_ids(all_seqs)
        q_emb = embs[0:1]
        c_embs = embs[1:]
        sims = (q_emb @ c_embs.T)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        top1 = top1_by_identity(sims, cids)
        return BaselineResult(
            baseline_name=self.name,
            mode=example.mode,
            candidate_ids=cids,
            top1=top1,
            prob_by_id=None,  # Cosine similarity is not a calibrated probability distribution
            local_gpu_ms=elapsed_ms,
            total_tokens=sum(len(s) for s in all_seqs),
        )

