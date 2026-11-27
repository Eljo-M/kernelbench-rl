"""Extract public website evidence from saved artifacts, without running experiments."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    return math.exp(sum(map(math.log, values))/len(values)) if values else None


def build(repo, out):
    sys.path.insert(0, str(repo))
    import numpy as np
    from kernels.op_registry import OP_REGISTRY
    baseline_path = repo/'experiments/rtx5060/full30/baselines.jsonl'
    paired_path = repo/'experiments/heldout_comparison/comparison.json'
    env_path = repo/'experiments/rtx5060/full30/environment.json'
    rows = [json.loads(line) for line in baseline_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 180 or len({r['op_name'] for r in rows}) != 30:
        raise ValueError('Expected the recorded final 180-row, 30-operator baseline suite')
    seen = {(r['op_name'], r['dtype'], r['seed']) for r in rows}
    if len(seen) != len(rows):
        raise ValueError('Duplicate baseline operator/precision/seed identity')
    summary = {}
    for dtype in ('fp32', 'bf16'):
        subset = [r for r in rows if r['dtype'] == dtype]
        passing = [r for r in subset if r['status'] == 'PASS']
        compiled = [r for r in passing if r['compile_ms'] > 0]
        summary[dtype] = dict(records=len(subset), passed=len(passing),
            operators=len({r['op_name'] for r in subset}),
            eager_geomean=gm([r['speedup_vs_eager'] for r in passing]),
            compile_geomean=gm([r['speedup_vs_compile'] for r in compiled]),
            compile_operators=len({r['op_name'] for r in compiled}))
    operators = []
    for name in sorted({r['op_name'] for r in rows}, key=lambda name: (OP_REGISTRY[name].level, name)):
        spec = OP_REGISTRY[name]
        subset = [r for r in rows if r['op_name'] == name]
        precision = {}
        for dtype in ('fp32', 'bf16'):
            current = [r for r in subset if r['dtype'] == dtype]
            precision[dtype] = dict(passed=sum(r['status'] == 'PASS' for r in current), total=len(current),
                eager_speedup=gm([r['speedup_vs_eager'] for r in current if r['status'] == 'PASS']),
                compile_speedup=gm([r['speedup_vs_compile'] for r in current if r['status'] == 'PASS' and r['compile_ms'] > 0]))
        operators.append(dict(name=name, level=spec.level, description=spec.description,
            input_shapes=[list(a.shape) for a in spec.make_inputs(np.random.default_rng(0))],
            analytic_flops=int(spec.flops), analytic_bytes=int(spec.bytes_moved),
            rtol=spec.rtol, atol=spec.atol, precision=precision,
            source=subset[0]['source'], source_sha256=subset[0]['source_sha256']))
    paired = json.loads(paired_path.read_text(encoding='utf-8'))
    historical = []
    train_paths = {'laptop_fullparam': 'experiments/laptop_fullparam/train.jsonl',
                   'qlora7b_fresh': 'experiments/colab_t4/qlora7b_fresh/train.jsonl'}
    for run in paired['runs']:
        env = run['environment']
        train_path = repo/train_paths[env['run_name']]
        training = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        historical.append(dict(name=env['run_name'], model=env['model_id'], mode=env['training_mode_label'],
            gpu=env['gpu'], dtype=env['dtype'], source_commit=env['source_commit'],
            checkpoint_sha256=env['checkpoint_weights_sha'], training_samples=len(training),
            optimizer_updates=max(r.get('optimizer_step_count', 0) for r in training),
            training_failures=dict(Counter(r['status'] for r in training)),
            methods=[dict(name=name, passed=method['pass_count'], total=method['total_ops'],
                          budget=method['search_budget_k']) for name, method in run['methods'].items()]))
    env = json.loads(env_path.read_text(encoding='utf-8'))
    payload = dict(schema_version='kernelbench_website_results_v1',
        author='Eljo Mustafaj', evidence_date='2026-11-26',
        baseline=dict(passed=sum(r['status'] == 'PASS' for r in rows), total=len(rows),
            operator_count=len(operators), hardware=env['gpu'], seeds=env['seeds'],
            timing_mode=env['timing_mode'], warmup=env['warmup'], iterations=env['iterations'],
            graph_calls_per_replay=16, summary=summary, operators=operators),
        historical=historical,
        retry=dict(status='pending', published_results=False, runs=[],
            note='The revised study has no published quality results in this website draft. Historical results remain separate.'),
        provenance=[dict(path=str(path.relative_to(repo)).replace('\\','/'), sha256=sha(path))
                    for path in (baseline_path, paired_path, env_path)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), operators=len(operators), records=len(rows), summary=summary)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'dist/results.json')
    args=parser.parse_args()
    build(args.repo.resolve(), args.out.resolve())
