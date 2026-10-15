"""Run the frozen LLM score-completion protocol for a new checkpoint."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from run_dwip_transfer_evaluation import BENCH, REPO, METHODS, evaluation_environment, sha256


DATASETS = ('mteb', 'mmlu', 'merged', 'benchpress')
PARENT = REPO / 'runs/objective_queues/dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918'
CHECKPOINT = REPO / 'runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth'
OUTPUT = REPO / 'runs/benchmark_evaluations/matrixformer_large_trained_llm_20261001'
LABEL = 'matrixformer_large_trained'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_panel(path):
    with np.load(path, allow_pickle=True) as stored:
        panel = dict(stored)
    for method in METHODS:
        for metric in ('r2', 'rmse'):
            values = panel[f'{method}_{metric}']
            assert values.shape == (10, 15) and np.isfinite(values).all(), (path, method, metric)
    return panel


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        assert sha256(filename) == expected, f'Input/source changed: {filename}'


def export_comparison(output, references):
    summary, folds, completed = [], [], []
    for dataset in DATASETS:
        path = output / 'logs' / f'tabimpute_{LABEL}_{dataset}.npz'
        if not path.exists():
            continue
        parent, current = load_panel(Path(references[dataset])), load_panel(path)
        assert np.array_equal(parent['benchmark_names'], current['benchmark_names'])
        completed.append(dataset)
        for method in METHODS:
            for budget in range(1, 16):
                row = dict(dataset=dataset, selection=method, budget=budget, folds=10)
                for metric in ('r2', 'rmse'):
                    before, after = parent[f'{method}_{metric}'][:, budget - 1], current[f'{method}_{metric}'][:, budget - 1]
                    row.update({f'parent_{metric}': float(before.mean()), f'new_{metric}': float(after.mean()),
                                f'delta_{metric}': float((after - before).mean())})
                    for fold, (old_value, new_value) in enumerate(zip(before, after)):
                        folds.append(dict(dataset=dataset, selection=method, budget=budget, fold=fold,
                                          metric=metric, parent=float(old_value), new=float(new_value),
                                          delta=float(new_value - old_value)))
                summary.append(row)
    if summary:
        for name, records in (('comparison.csv', summary), ('comparison_per_fold.csv', folds),
                              ('summary_5_10_15.csv', [row for row in summary if row['budget'] in (5, 10, 15)])):
            with (output / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(records)
    lines = ['# Larger-matrix checkpoint: LLM score completion', '',
             f'Completed datasets: {len(completed)}/4. Higher R2 is better; lower standardized RMSE is better.', '',
             'Same frozen input matrices, 10 folds, selection seed 42, entropy/MI/random selection, and budgets 1--15.',
             'Inference retains the original 4000-row cap and 200-row test batches; no 2000-row chunking change.',
             'This is benchmark-score imputation, not new language-model answer generation.',
             'The parent checkpoint and external baselines are reused, not retuned. Public benchmarks were previously inspected.', '',
             '| Dataset | Selection | Budget | Parent R2 | New R2 | Change |',
             '|---|---|---:|---:|---:|---:|']
    for row in summary:
        if row['budget'] in (5, 10, 15):
            lines.append(f"| {row['dataset']} | {row['selection']} | {row['budget']} | {row['parent_r2']:.4f} | {row['new_r2']:.4f} | {row['delta_r2']:+.4f} |")
    lines += ['', 'Fold-level R2/RMSE arrays, evaluator logs, source snapshots and hashes are saved beside this summary.',
              'The unchanged historical evaluator saves fold metrics, not per-entry predictions; no independent prediction-rescoring claim is made.',
              'Paper tables are not automatically updated by this evaluation.']
    (output / 'LLM_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_manifest = read_json(PARENT / 'manifest.json')
        parent_hashes = read_json(PARENT / 'teacher_full_comparison.json')['input_sha256']
        references = parent_manifest['selected_panels']
        data_paths = [BENCH / 'data' / f'{dataset}.matrix.npz' for dataset in DATASETS]
        for path in [BENCH / 'code/eval_tabimpute.py', *data_paths, *map(Path, references.values())]:
            assert sha256(path) == parent_hashes[str(path)], f'Frozen benchmark/reference differs: {path}'
        config = read_json(checkpoint.parent / 'config.json')
        assert sha256(config['parent_checkpoint']) == '1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d'
        sources = [Path(__file__).resolve(), REPO / 'scripts/run_dwip_transfer_evaluation.py',
                   REPO / 'scripts/run_rowcol_controlled_experiments.py',
                   *sorted((BENCH / 'code').glob('*.py')), *sorted((REPO / 'src/tabimpute').rglob('*.py'))]
        inputs = [checkpoint, checkpoint.parent / 'config.json', *data_paths, *map(Path, references.values())]
        environment = evaluation_environment(output, checkpoint, LABEL, ','.join(DATASETS))
        environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(output / 'matplotlib'))
        manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        parent_checkpoint=config['parent_checkpoint'], parent_panels=references,
                        input_sha256={str(path): sha256(path) for path in inputs},
                        source_sha256={str(path): sha256(path) for path in sources},
                        environment={key: value for key, value in environment.items() if key.startswith('BENCHSELECT_')},
                        datasets=list(DATASETS), folds=10, budgets=list(range(1, 16)),
                        inference='Unchanged historical evaluator; FP32 weights, BF16 autocast, predictive mean',
                        context='4000 rows including at most 200 test rows; at most 3800 training context rows, subsampling seed 42',
                        selection='Fixed synthetic-selected checkpoint; report-only real-world evaluation')
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            assert read_json(manifest_path) == manifest, 'Resume configuration or source hash mismatch'
        else:
            write_json(manifest_path, manifest)
            for path in sources:
                destination = output / 'source_snapshot' / path.relative_to(REPO.parent)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        try:
            completed = export_comparison(output, references)
            for dataset in DATASETS:
                if dataset in completed:
                    continue
                write_json(output / 'status.json', dict(state='running', dataset=dataset, completed=completed, pid=os.getpid(), updated_unix=time.time()))
                environment['BENCHSELECT_DATASETS'] = dataset
                with (output / f'{dataset}.log').open('a') as stream:
                    subprocess.run([sys.executable, str(BENCH / 'code/eval_tabimpute.py')], cwd=BENCH,
                                   env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                verify_hashes(manifest['input_sha256'])
                verify_hashes(manifest['source_sha256'])
                completed = export_comparison(output, references)
                print(f'Completed {dataset}: {len(completed)}/4 datasets', flush=True)
            write_json(output / 'audit.json', dict(status='passed', datasets=completed, settings=180, fold_settings=1800,
                       input_and_source_hashes_verified=True, saved_fold_metrics_finite=True,
                       panel_sha256={dataset: sha256(output / 'logs' / f'tabimpute_{LABEL}_{dataset}.npz') for dataset in DATASETS},
                       checkpoint_sha256=manifest['checkpoint_sha256'], per_entry_predictions_saved=False))
            write_json(output / 'status.json', dict(state='complete', completed=completed, updated_unix=time.time()))
        except Exception as error:
            write_json(output / 'status.json', dict(state='failed', error=repr(error), updated_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
