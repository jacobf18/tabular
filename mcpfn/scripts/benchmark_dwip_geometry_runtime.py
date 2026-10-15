#!/usr/bin/env python3
"""Benchmark public impute() latency on matched square matrices, saving CSV/PNG/PDF.

Example (from any directory):
    python benchmark_dwip_geometry_runtime.py --output-dir /path/to/results

Each model/size runs in a separate process to release CUDA allocations. Loading
and warmup are excluded; timed calls include normalization, transfers, inference,
and conversion to an imputed NumPy matrix. No row/column windowing is applied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / 'src/tabimpute/workdir/tabimpute-spectral_geometry_20260914-geometry_lr1e5/best_validation_checkpoint.pth'
DEFAULT_MANIFEST = REPO / 'benchmark/movielens/configs/tabimpute_models.json'
LABELS = {'dwip': 'DWIP v2-2', 'geometry': 'Row/column + SVT + cell query'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_query(fields, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader'], text=True).strip()


def model_spec(args, name):
    if name == 'dwip':
        entries = json.loads(args.dwip_manifest.read_text())['models']
        entry = next(m for m in entries if 'dwip-v2-2' in [m['name'], *m.get('aliases', [])])
        return dict(checkpoint=entry['checkpoint_path'], config=entry['model_config'], family=entry.get('model_family', 'new_stable'))
    cfg = json.loads((args.checkpoint.parent / 'config.json').read_text())
    return dict(checkpoint=str(args.checkpoint), config=cfg['model'], family=cfg['model_family'])


def worker(args):
    sys.path.insert(0, str(REPO / 'src'))
    import numpy as np
    import torch
    from tabimpute.interface import ImputePFN

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    spec = model_spec(args, args.worker_model)
    n = args.worker_size
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, n]))
    rank = min(10, n)
    clean = (rng.standard_normal((n, rank)) @ rng.standard_normal((rank, n))) / rank ** .5
    mask = np.zeros(n * n, dtype=bool)
    mask[rng.choice(n * n, int(round(n * n * args.missing_fraction)), replace=False)] = True
    mask = mask.reshape(n, n)
    x = clean.copy()
    x[mask] = np.nan
    row = dict(model=args.worker_model, size=n, rank=rank, missing_cells=int(mask.sum()),
               input_sha256=hashlib.sha256(x.tobytes()).hexdigest(), precision=args.precision,
               warmup=args.warmup, repeats=args.repeats, seconds=[], status='running')
    phase = 'load'
    try:
        imputer = ImputePFN(device='cuda', checkpoint_path=spec['checkpoint'],
            entry_wise_features=False, json_config={'config': spec['config']},
            model_family=spec['family'], precision_mode=args.precision,
            prediction_stat='mean', max_num_rows=None, max_num_cols=None)
        for i in range(args.warmup + args.repeats):
            phase = 'warmup' if i < args.warmup else 'timing'
            inp = x.copy()  # Copy/setup excluded; public impute() itself is included.
            if i == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = imputer.impute(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if prediction.shape != x.shape or not np.isfinite(prediction).all():
                raise ValueError('Imputation returned invalid shape or nonfinite values')
            if i >= args.warmup:
                row['seconds'].append(elapsed)
            del prediction
        row.update(status='ok', median_seconds=float(np.median(row['seconds'])),
                   min_seconds=min(row['seconds']), max_seconds=max(row['seconds']))
    except torch.cuda.OutOfMemoryError as error:
        row.update(status='out_of_memory', phase=phase, error=str(error))
    except Exception as error:
        row.update(status='error', phase=phase, error=f'{type(error).__name__}: {error}')
    row.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    args.worker_output.write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


def save_outputs(out, rows, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    (out / 'results.json').write_text(json.dumps(rows, indent=2))
    fields = ['model', 'size', 'status', 'missing_cells', 'median_seconds', 'min_seconds', 'max_seconds', 'peak_allocated_gib', 'peak_reserved_gib']
    with (out / 'timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
    colors = {'dwip': '#2563eb', 'geometry': '#d95f02'}
    failures = []
    for name in LABELS:
        records = sorted([r for r in rows if r['model'] == name], key=lambda r: r['size'])
        xs = [r['size'] for r in records]
        ys = [r.get('median_seconds', float('nan')) if r['status'] == 'ok' else float('nan') for r in records]
        ax.plot(xs, ys, 'o-', color=colors[name], label=LABELS[name], linewidth=2)
        good = [r for r in records if r['status'] == 'ok']
        if good:
            median = np.array([r['median_seconds'] for r in good])
            ax.errorbar([r['size'] for r in good], median,
                yerr=[median - [r['min_seconds'] for r in good], [r['max_seconds'] for r in good] - median],
                fmt='none', ecolor=colors[name], capsize=3)
        failures.extend(f"{LABELS[name]}, N={r['size']}: {r['status']}" for r in records if r['status'] != 'ok')
    ax.set(xscale='log', yscale='log', xlabel='Matrix size N (N × N)', ylabel='Imputation time (seconds)',
           title=f'Full-matrix imputation runtime · {args.missing_fraction:.0%} missing')
    ax.set_xticks(args.sizes, [str(n) for n in args.sizes])
    ax.grid(True, which='major', alpha=.22)
    ax.legend(loc='upper left')
    note = f'Median of {args.repeats} calls after {args.warmup} warmup; whiskers: min–max.\n{args.precision}; GPU-synchronized wall time; loading excluded.'
    if failures:
        note += '\n' + '\n'.join(failures)
    fig.supxlabel(note, fontsize=9)
    fig.savefig(out / 'runtime_comparison.png', dpi=180)
    fig.savefig(out / 'runtime_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 25, 50, 100, 500, 1000])
    parser.add_argument('--output-dir', type=Path, default=REPO / 'runs/benchmark_evaluations/dwip_geometry_runtime')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--dwip-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--missing-fraction', type=float, default=.8)
    parser.add_argument('--precision', choices=['amp_bfloat16', 'full_bfloat16', 'float32'], default='amp_bfloat16')
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    parser.add_argument('--timeout', type=int, default=900, help='Seconds per model/size, including loading and warmup')
    parser.add_argument('--allow-shared-gpu', action='store_true', help='Allow other GPU jobs; timings will reflect contention')
    parser.add_argument('--worker-model', choices=list(LABELS), help=argparse.SUPPRESS)
    parser.add_argument('--worker-size', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1 or args.warmup < 1 or not 0 < args.missing_fraction < 1 or not 0 < args.memory_fraction <= 1:
        parser.error('Require N>=2, repeats/warmup>=1, 0<missing-fraction<1, and 0<memory-fraction<=1')
    if args.worker_model:
        worker(args)
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        parser.error(f'Results already exist in {out}; choose a fresh --output-dir')
    specs = {name: model_spec(args, name) for name in LABELS}
    for spec in specs.values():
        spec['checkpoint_sha256'] = sha256(spec['checkpoint'])
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    models=specs, gpu=gpu_query('name,driver_version,memory.total'),
                    script_sha256=sha256(__file__), scope='Complete public impute() call on all missing cells; no matrix windowing; CPU input/output; model loading excluded',
                    gpu_process_checks=[])
    rows = []
    for index, n in enumerate(args.sizes):
        # Alternate order to reduce systematic first/second-model timing bias.
        for name in (list(LABELS) if index % 2 == 0 else list(reversed(LABELS))):
            active = gpu_query('pid,used_memory', 'compute-apps')
            manifest['gpu_process_checks'].append(dict(size=n, model=name, processes=active, utc=datetime.now(timezone.utc).isoformat()))
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            if active and not args.allow_shared_gpu:
                raise RuntimeError(f'GPU is in use ({active}); wait for idle or explicitly use --allow-shared-gpu')
            target = out / f'{name}_{n}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker-model', name, '--worker-size', str(n), '--worker-output', str(target),
                       '--checkpoint', str(args.checkpoint.resolve()), '--dwip-manifest', str(args.dwip_manifest.resolve()),
                       '--repeats', str(args.repeats), '--warmup', str(args.warmup), '--precision', args.precision,
                       '--missing-fraction', str(args.missing_fraction), '--seed', str(args.seed), '--threads', str(args.threads), '--memory-fraction', str(args.memory_fraction)]
            print(f'Running {LABELS[name]} at {n} x {n}', flush=True)
            with (out / f'{name}_{n}.log').open('w') as log:
                try:
                    p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads)}, timeout=args.timeout)
                    row = json.loads(target.read_text()) if target.exists() else dict(model=name, size=n, status='worker_failed', returncode=p.returncode)
                except subprocess.TimeoutExpired:
                    row = dict(model=name, size=n, status='timeout', timeout_seconds=args.timeout)
            rows.append(row)
            print(json.dumps(row), flush=True)
            save_outputs(out, rows, args)
    for n in args.sizes:
        hashes = {r['input_sha256'] for r in rows if r['size'] == n and 'input_sha256' in r}
        assert len(hashes) <= 1, f'Input mismatch at {n}'
    print(f'Saved results and plots to {out}', flush=True)


if __name__ == '__main__':
    main()
