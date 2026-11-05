"""Shared training-context acceptance using pinned tokenizers, without weights."""
import json
from pathlib import Path
import torch
from ..assets import ROOT, model_specs, verify_snapshot
from ..baselines.ar_llm import pack_causal_prompt
from .sources import materialize_example


class TrainingContextAudit:
    def __init__(self):
        from transformers import AutoTokenizer
        self.entries=[]
        for name,spec in model_specs().items():
            if name != "llada_8b_base" and not name.startswith("nemotron"):
                continue
            path=ROOT/spec["local_dir"]
            print(f"[prepare] Loading context-audit tokenizer: {name} (local files only)",flush=True)
            verify_snapshot(path,spec["repo_id"])
            config=json.loads((path/"config.json").read_text(encoding="utf-8"))
            limit=int(config.get("max_position_embeddings",config.get("max_sequence_length",4096)))
            tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=True)
            self.entries.append((name,tokenizer,limit))
            print(f"[prepare] Context-audit tokenizer ready: {name}; context limit={limit}",flush=True)

    def check(self,record):
        for name,tokenizer,limit in self.entries:
            example,_=materialize_example(tokenizer,record,enable_augmentation=False)
            n=len(example.query_ids)+sum(1+len(c.token_ids) for c in example.candidates)
            if n > limit:
                return f"{name}:serialized_dlm:{n}>{limit}"
            if name.startswith("nemotron"):
                # Both prompts are independently adapted from the same checkpoint.
                # Raw-text AR packing tokenizes complete answer strings itself;
                # these fixture-only fallback IDs are never used on this path.
                packed=pack_causal_prompt(example,tokenizer,0,0,torch.device("cpu"))
                length=packed.prompt_ids.shape[-1]+max(map(len,packed.label_token_seqs+[packed.ref_token_seq]))
                if length > limit:
                    return f"{name}:ar:{length}>{limit}"
        return None
