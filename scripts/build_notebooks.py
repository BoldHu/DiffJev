"""Generate the eight formal notebooks; never execute experiments."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
def markdown(text):
    return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code(text):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(keepends=True)}
BOOT = """import os, sys, gc
from pathlib import Path
ROOT = Path(os.environ.get('DIFFJEV_ROOT', Path.cwd())).resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'AGENTS.md').exists(), 'Set DIFFJEV_ROOT to the checked-out repository'
sys.path.insert(0, str(ROOT / 'src'))
from diffjev import workflows as w
from diffjev.config import load_yaml_config
cfg = load_yaml_config(str(ROOT / 'configs/workflow.yaml'))
SEED, DEVICE = cfg['seed'], cfg['device']
"""

def build():
    notebooks = [
    ('00_setup', 'Environment and pinned assets', [
      markdown('Use Python 3.12 for the formal environment. Installation and downloads below are explicit opt-in operations. Upload complete pinned snapshots to the paths in configs/models.yaml and configs/data.yaml; include download_provenance.json. No model weights are fetched by importing DiffJev.'),
      code("INSTALL = False\nif INSTALL:\n    import subprocess\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT/'requirements-lock.txt')])"),
      code("from diffjev.assets import environment_record, model_specs\nw.save_json(ROOT/'outputs/formal/environment.json', environment_record())\nprint(environment_record())\nfor name,spec in model_specs().items():\n    print(name, spec['revision'], ROOT/spec['local_dir'])"),
      code("# Optional large downloads; edit only on the intended experiment machine.\nDOWNLOAD = False\nif DOWNLOAD:\n    import subprocess\n    subprocess.check_call([sys.executable, str(ROOT/'scripts/download_assets.py'), '--all-core', '--yes'])"),
      markdown('Before Exp 5, expose exactly one A100 80GB with CUDA_VISIBLE_DEVICES. Larger training runs may be scheduled independently on the three GPUs. This code does not silently shard model layers across devices.')]),
    ('01_prepare_data','Real data preparation, audits, and frozen menus',[
      markdown('Requires the pinned OpenJev SFT, ToolRet training, original JevBench text cases, ToolRet Queries/Tools snapshots, and the four backbone tokenizer/config snapshots for shared context acceptance (no weights are loaded during preparation). Restore withheld JevBench states from their licensed source with w.restore_benchmark_states(); hashes are checked. Records with ambiguous labels are quarantined. Review data/processed/formal/audit.json before freezing comparisons.'),
      code("summary = w.prepare_data(seed=SEED)\nprint(summary['accepted'], summary['rejected'])"),
      code("print(w.build_scaling_manifest(seed=SEED, num_queries=cfg['scaling_queries']))")]),
    ('02_train_diffjev','Independent pretrained DiffJev adaptations',[
      markdown('These cells perform real supervised training. The saved source/augmentation trace, 3,125 updates, final-four-layer-plus-normalization policy, development checkpoint rule, and separate calibration split are shared. Set RUNS to the machines/runs allocated to this session. Three anchor seeds are planned; the size panel initially uses seed 0.'),
      code("RUNS = cfg['backbones']\nfor backbone in RUNS:\n    summary = w.train_run(backbone, seed=SEED, device=DEVICE)\n    print(backbone, summary)\n    gc.collect()"),
      markdown('Resume a specific interrupted adaptation with w.train_run(..., resume=".../step_500.pt"). CPU/CUDA/Python/NumPy RNG, corruption generator, augmentation generator, counters, and trace identity are restored.')]),
    ('03_run_baselines','Train real matched controls and load reference baselines',[
      markdown('Every trained control starts again from the original pinned checkpoint. The released Jeva model uses its official runtime; the AR reference uses Qwen2.5-3B-Instruct. Missing assets cause an explicit error. API calls remain disabled unless enable_paid_api is deliberately enabled in the shared configuration.'),
      code("for backbone in cfg['backbones']:\n    methods = ['serialized_dlm'] + (['ar'] if backbone.startswith('nemotron') else [])\n    for method in methods:\n        print(w.train_run(backbone, method=method, seed=SEED, device=DEVICE))\n        gc.collect()"),
      code("for name in cfg['references'] + ['bge']:\n    runner = w.baseline_runner(name, device=DEVICE)\n    print(name, type(runner).__name__)\n    del runner\n    gc.collect()")]),
    ('04_exp1_exp2','Original benchmark and canonical-family robustness',[
      code("for backbone in cfg['backbones']:\n    runner = w.load_runner(backbone, seed=SEED, device=DEVICE)\n    output = w.run_directory(backbone, seed=SEED)/'evaluation'\n    w.evaluate_benchmark(runner, output, robustness=backbone=='llada_8b_base')\n    del runner\n    gc.collect()"),
      code("runner = w.load_runner('llada_8b_base', method='serialized_dlm', seed=SEED, device=DEVICE)\nw.evaluate_benchmark(runner, w.run_directory('llada_8b_base','serialized_dlm',seed=SEED)/'evaluation')\ndel runner\ngc.collect()"),
      code("names = cfg['references'] + (['jev_api'] if cfg['enable_paid_api'] else [])\nfor name in names:\n    runner = w.baseline_runner(name, device=DEVICE, allow_api=cfg['enable_paid_api'])\n    w.evaluate_benchmark(runner, ROOT/'outputs/formal/references'/name)\n    del runner\n    gc.collect()"),
      markdown('The original revision-pinned official evaluator writes official/scores.json. Project summaries are labeled separately because the original scorer defines task-specific headlines, not a single universal aggregate. Probability metrics use the same candidate-only normalization as forced-choice predictions.')]),
    ('05_exp3_scaling','Fixed nested ToolRet candidate pools',[
      code("# Reuse the frozen manifests from notebook 01; no negative mining on test outcomes.\nfor backbone in cfg['backbones']:\n    methods = ['diffjev','serialized_dlm'] + (['ar'] if backbone.startswith('nemotron') else [])\n    for method in methods:\n        runner = w.load_runner(backbone,method=method,seed=SEED,device=DEVICE)\n        w.evaluate_scaling(runner,w.run_directory(backbone,method,seed=SEED)/'evaluation')\n        del runner\n        gc.collect()"),
      code("for name in cfg['references'] + ['bge']:\n    runner=w.baseline_runner(name,device=DEVICE)\n    w.evaluate_scaling(runner,ROOT/'outputs/formal/references'/name)\n    del runner\n    gc.collect()"),
      markdown('K > 32 is cardinality extrapolation. Reports include actual K, Hit@1, coverage, raw query predictions, and stagewise gold survival. These are supplied-candidate routing results, not full-corpus retrieval or tool execution.')]),
    ('06_exp4_ablation','Separately trained one-factor controls',[
      code("for variant in cfg['anchor_ablations']:\n    print(w.train_run('llada_8b_base',variant=variant,seed=SEED,device=DEVICE))\n    gc.collect()"),
      code("for variant in w.CORE_ABLATIONS:\n    w.evaluate_ablation(variant,seed=SEED,device=DEVICE)\n    gc.collect()\nw.collect_results()"),
      markdown('The T=1 control has full own-detail access in its only pass, no corruption loss, and no iterative feedback. No-pruning reuses the full trained model. No-augmentation replays the same source trace with native menus. No factorial grid or extra experiment is introduced.')]),
    ('07_exp5_efficiency','Same-checkpoint timing and executed-operation accounting',[
      markdown('Requires one visible A100 80GB, bf16, batch size 1. Each method is loaded and released separately. Every timed request starts from raw semantic text with input caches cleared; five warmups precede 100 fixed stratified queries and three repetitions. Operator/CUDA profiles run separately from wall timing. Inspect unclassified operator coverage before treating logical FLOP totals as complete.'),
      code("for backbone in ['nemotron_3b','nemotron_8b','nemotron_14b']:\n    w.evaluate_efficiency(backbone,seed=SEED,device=DEVICE)\n    gc.collect()\nw.evaluate_efficiency('llada_8b_base',seed=SEED,device=DEVICE,attribution=True)"),
      code("w.collect_results()"),
      markdown('Results include accuracy, survival, p50/p95, query-bootstrap uncertainty, resident-weight peak memory, matched-successful-query speedups, failures, and latency/GFLOP plots. CPU fixtures and theoretical attention ratios are never presented as measured A100 speedups.')])]
    for name,title,cells in notebooks:
        nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
            'cells':[markdown('# '+title+'\n\nFormal DiffJev workflow; see AGENTS.md and prompt.md.'),code(BOOT)]+cells}
        for i,cell in enumerate(nb['cells']):cell['id']=f'{name}-{i}'
        (ROOT/'notebooks'/f'{name}.ipynb').write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Generated eight formal notebooks (not executed).')

if __name__=='__main__':build()
