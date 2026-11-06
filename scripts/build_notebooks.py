"""Generate the eight formal notebooks; never execute experiments."""
from pathlib import Path
import json
from setup_notebook_cells import setup_cells

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
if not (ROOT / 'AGENTS.md').exists() and Path('/content/drive/MyDrive/DiffJev/AGENTS.md').exists():
    ROOT = Path('/content/drive/MyDrive/DiffJev')
assert (ROOT / 'AGENTS.md').exists(), 'Mount Drive and set DIFFJEV_ROOT to the checkout used by notebook 00'
sys.path.insert(0, str(ROOT / 'src'))
from diffjev.notebook_runtime import NotebookRuntime
runtime = NotebookRuntime(ROOT)
runtime.python_code("from diffjev.preparation import check_environment; from pathlib import Path; check_environment(Path.cwd())", label='__NOTEBOOK_NAME___environment')
def run_python(source):
    prefix = '''import gc
from pathlib import Path
from diffjev import workflows as w
from diffjev.config import load_yaml_config
ROOT = w.ROOT
cfg = load_yaml_config(str(ROOT / 'configs/workflow.yaml'))
SEED, DEVICE = cfg['seed'], cfg['device']
'''
    runtime.python_code(prefix + source, label='__NOTEBOOK_NAME__')
print('Experiment worker:', runtime.python)
"""

def build():
    notebooks = [
    ('00_setup', 'Complete Colab preparation', setup_cells(markdown,code)),
    ('01_prepare_data','Real data preparation, audits, and frozen menus',[
      markdown('Requires the pinned OpenJev SFT, ToolRet training, original JevBench text cases, ToolRet Queries/Tools snapshots, and the four backbone tokenizer/config snapshots for shared context acceptance (no weights are loaded during preparation). Restore withheld JevBench states from their licensed source with w.restore_benchmark_states(); hashes are checked. Records with ambiguous labels are quarantined. Review data/processed/formal/audit.json before freezing comparisons.'),
      markdown('Notebook 00 already includes preparation. This notebook optionally verifies/reuses the same frozen artifacts; it does not overwrite them.'),
      code("from diffjev.preparation import prepare_all\nprepare_all(seed=SEED, scaling_queries=cfg['scaling_queries'])"),
      code("audit = w.read_json(w.PREPARED/'audit.json')\nprint('Sources:', audit['sources'])\nprint('Rejected records:',len(audit['rejections']))\nprint('Trace:',audit['trace_sha256'])")]),
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
      markdown('Runs only configs/workflow.yaml backbones. With llada_8b_base alone, this is the anchor attribution comparison (DiffJev, same-backbone serialized DLM, full-context T=1 and no-pruning). The required Nemotron same-checkpoint AR/DLM/DiffJev size panel is deferred, not completed or replaced by the Qwen reference. Complete notebook 06 before the anchor timing run.'),
      code("selected = list(dict.fromkeys(cfg['backbones']))\nif not selected or any(b not in w.BACKBONES for b in selected):\n    raise ValueError('Select valid backbones in configs/workflow.yaml')\nprint('Efficiency backbones:', selected)\nprint('Deferred required backbones:', [b for b in w.BACKBONES if b not in selected])\nfor backbone in selected:\n    w.evaluate_efficiency(backbone, seed=SEED, device=DEVICE, attribution=backbone=='llada_8b_base')\n    gc.collect()"),
      code("w.collect_results()"),
      markdown('Results include accuracy, survival, p50/p95, query-bootstrap uncertainty, resident-weight peak memory, matched-successful-query speedups, failures, and latency/GFLOP plots. CPU fixtures and theoretical attention ratios are never presented as measured A100 speedups.')])]
    for name,title,cells in notebooks:
        if name != '00_setup':
            cells = [code('run_python(r\"\"\"\n'+''.join(cell['source'])+'\n\"\"\")')
                     if cell['cell_type']=='code' else cell for cell in cells]
            cells = [code(BOOT.replace('__NOTEBOOK_NAME__',name))]+cells
        nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
            'cells':[markdown('# '+title+'\n\nFormal DiffJev workflow; see AGENTS.md and prompt.md.')]+cells}
        for i,cell in enumerate(nb['cells']):cell['id']=f'{name}-{i}'
        (ROOT/'notebooks'/f'{name}.ipynb').write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Generated eight formal notebooks (not executed).')

if __name__=='__main__':build()
