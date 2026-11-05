"""Notebook preparation entry point. Does not download or train models."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    from diffjev.preparation import check_environment,prepare_all,asset_status
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["check","prepare","status"])
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--scaling-queries",type=int,default=512)
    parser.add_argument("--restore-states",action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--model",action="append",default=[])
    args=parser.parse_args()
    if args.stage=="check":
        check_environment(ROOT)
    elif args.stage=="prepare":
        prepare_all(args.seed,args.scaling_queries,args.restore_states,args.source_root)
    else:
        asset_status(args.model,args.seed)

if __name__=="__main__":
    main()
