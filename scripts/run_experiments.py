"""Formal DiffJev workflows. No synthetic data, auto downloads, or implicit API calls."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from diffjev import workflows as w


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","restore-states","manifests","train","exp1-exp2","exp3","exp4","exp5","report"])
    parser.add_argument("--backbone",default="llada_8b_base",choices=w.BACKBONES)
    parser.add_argument("--method",default="diffjev",choices=["diffjev","ar","serialized_dlm"])
    parser.add_argument("--variant",default="full",choices=w.CORE_ABLATIONS)
    parser.add_argument("--reference",choices=["jeva","llm_as_jev","jev_api","bge","e5"])
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--source-root")
    parser.add_argument("--allow-api",action="store_true")
    parser.add_argument("--attribution",action="store_true")
    args=parser.parse_args()
    if args.stage=="prepare":
        result=w.prepare_data(seed=args.seed)
    elif args.stage=="restore-states":
        result=w.restore_benchmark_states(args.source_root)
    elif args.stage=="manifests":
        result=w.build_scaling_manifest(seed=args.seed)
    elif args.stage=="train":
        result=w.train_run(args.backbone,args.method,args.variant,args.seed,args.resume,args.device)
    elif args.stage=="exp4":
        result=w.evaluate_ablation(args.variant,args.seed,args.device)
    elif args.stage=="exp5":
        result=w.evaluate_efficiency(args.backbone,args.seed,args.device,args.attribution)
    elif args.stage=="report":
        result=w.collect_results()
    else:
        if args.reference:
            runner=w.baseline_runner(args.reference,args.device,args.allow_api)
            output=w.ROOT/"outputs/formal/references"/args.reference
        else:
            runner=w.load_runner(args.backbone,args.method,args.variant,args.seed,device=args.device)
            output=w.run_directory(args.backbone,args.method,args.variant,args.seed)/"evaluation"
        if args.stage=="exp1-exp2":
            result=w.evaluate_benchmark(runner,output,robustness=args.backbone=="llada_8b_base")
        else:
            result=w.evaluate_scaling(runner,output)
    print("Completed:",args.stage)
    # Large predictions/profiles are saved to disk, not dumped into notebook output.
    if isinstance(result,dict):
        print("Artifacts:",list(result)[:12])

if __name__=="__main__":
    main()
