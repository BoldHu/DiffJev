"""Synthetic smoke run: forward, loss, backward and a few AdamW updates on the
tiny random mock backbone. Produces NO research evidence.

    python scripts/smoke_train.py --updates 20 --device cuda
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from diffjev.backbones.mock import MockConfig, build_mock_backbone, default_mock_tokenizer  # noqa: E402
from diffjev.config import DiffJevConfig  # noqa: E402
from diffjev.data.synthetic import synthetic_choice, synthetic_mixture  # noqa: E402
from diffjev.model import DiffJevModel  # noqa: E402
from diffjev.training import apply_freeze_policy, train_synthetic  # noqa: E402
from diffjev.utils import environment_report, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--examples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    print(json.dumps(environment_report(), indent=1))
    tok = default_mock_tokenizer()
    bb = build_mock_backbone(MockConfig(seed=args.seed), tok).to(args.device)
    counts = apply_freeze_policy(bb, 4)
    print("parameters:", counts)
    model = DiffJevModel(bb, DiffJevConfig())
    print("enabled loss terms:", model.cfg.enabled_loss_terms())
    data = synthetic_mixture(tok, args.examples, seed=args.seed)
    train_synthetic(
        model, data, args.updates, effective_batch=args.batch, lr=args.lr, seed=args.seed,
        log_fn=lambda h: print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}),
    )
    import random

    ex = synthetic_choice(tok, random.Random(123), 48, max_len=4)
    out = model.predict(ex, gold_id=ex.gold_id)
    rec = out.to_record()
    rec["probs"] = {k: round(v, 4) for k, v in list(rec["probs"].items())[:5]}
    print("pruned K=48 inference:", json.dumps(rec)[:800])


if __name__ == "__main__":
    main()

