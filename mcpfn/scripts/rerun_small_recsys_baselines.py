"""Rerun frozen small rating benchmarks and export test-user bootstrap intervals."""

import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import cornac
from cornac.data import Dataset
from cornac.eval_methods.base_method import rating_eval
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919'
DEFAULT_OUTPUT = ROOT / 'runs/benchmark_evaluations/small_recsys_baseline_ci'
REPLICATES = 1000
BOOTSTRAP_SEED = 0
CASES = {
    '100K': dict(label='MovieLens 100K', source='movielens_100k',
                 sha256='b771f5c99a854f13a2b95f19b75fa930e9f5f663f6885dc8e6fa88ac8374e592'),
    'netflix': dict(label='Netflix small', source='netflix_small',
                    sha256='41c3757f909c95b763137c2efb9bf653bef5ce5ac99fee2918dcb2d9c11c63d0'),
}
CONFIGS = {
    'SVD': dict(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123),
    'PMF': dict(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    'UserKNN': dict(k=20, similarity='pearson', mean_centered=True, seed=123),
    'ItemKNN': dict(k=20, similarity='pearson', mean_centered=True, seed=123),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def user_bootstrap_interval(values, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Require at least two finite user scores in a one-dimensional array')
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError('Require a positive integer bootstrap replicate count')
    generator = np.random.default_rng(seed)
    means = [values[generator.integers(0, len(values), size=len(values))].mean()
             for replicate in range(replicates)]
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def export_intervals(output, records):
    labels = {case['label']: dataset for dataset, case in CASES.items()}
    rows = []
    for record in records:
        path = output / labels[record['dataset']] / record['method'] / 'per_user.npz'
        if digest(path) != record['artifact_sha256']['per_user.npz']:
            raise ValueError(f'Per-user artifact hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['users'], np.unique(saved['users']))
            for metric in ('RMSE', 'MAE'):
                values = saved[metric]
                if values.shape != saved['users'].shape or len(values) != record['users']:
                    raise ValueError(f'Per-user coverage mismatch: {path}/{metric}')
                np.testing.assert_allclose(values.mean(), record[metric], rtol=0, atol=1e-12)
                lower, upper = user_bootstrap_interval(values)
                rows.append(dict(dataset=record['dataset'], method=record['method'], metric=metric,
                                 value=record[metric], ci95_low=lower, ci95_high=upper,
                                 unit='test_user', units=len(values), bootstrap_replicates=REPLICATES,
                                 bootstrap_seed=BOOTSTRAP_SEED, ci_status='available'))
    path = output / 'summary.csv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return dict(method='95% percentile bootstrap of mean per-user scores', unit='test_user',
                replicates=REPLICATES, seed=BOOTSTRAP_SEED, intervals=len(rows),
                conditional_on='Fitted model, frozen split and observed items; not training-seed uncertainty',
                artifact_sha256={'summary.csv': digest(path)})


def frozen_datasets(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    num_users, num_items = map(int, arrays['shape'])
    user_map = OrderedDict((index, index) for index in range(num_users))
    item_map = OrderedDict((index, index) for index in range(num_items))
    datasets = []
    coordinates = []
    for subset in ('train', 'test'):
        triplets = tuple(arrays[f'{subset}_{field}'] for field in ('users', 'items', 'ratings'))
        users, items, ratings = triplets
        if not (len(users) == len(items) == len(ratings)) or not np.isfinite(ratings).all():
            raise ValueError(f'Invalid {subset} observations')
        if not ((users >= 0) & (users < num_users) & (items >= 0) & (items < num_items)).all():
            raise ValueError(f'Invalid {subset} indices')
        indices = users * num_items + items
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f'Duplicate {subset} coordinates')
        coordinates.append(indices)
        dataset = Dataset(num_users, num_items, user_map, item_map, triplets, seed=123)
        for actual, expected in zip(dataset.uir_tuple, triplets):
            np.testing.assert_array_equal(actual, expected)
        datasets.append(dataset)
    np.testing.assert_array_equal(np.unique(arrays['train_users']), np.arange(num_users))
    np.testing.assert_array_equal(np.unique(arrays['train_items']), np.arange(num_items))
    if np.intersect1d(*coordinates).size:
        raise ValueError('Train/test leakage')
    return datasets


def fit_case(dataset, method, directory):
    split_path = directory.parent / 'split.npz'
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIGS[method], verbose=False)
    signature = dict(dataset=CASES[dataset]['label'], method=method, config=config,
                     split_sha256=digest(split_path), script_sha256=digest(Path(__file__)),
                     cornac=cornac.__version__, numpy=np.__version__, python=platform.python_version(),
                     thread_environment={name: os.environ.get(name) for name in
                                         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')})
    metrics_path = directory / 'metrics.json'
    if metrics_path.exists():
        record = json.loads(metrics_path.read_text())
        if record['signature'] != signature:
            raise ValueError(f'Resume configuration mismatch: {directory}')
        for filename, expected in record['artifact_sha256'].items():
            if digest(directory / filename) != expected:
                raise ValueError(f'Resume artifact mismatch: {directory / filename}')
        return record
    started = time.perf_counter()
    train, test = frozen_datasets(split_path)
    model = getattr(cornac.models, method)(**config).fit(train)
    fit_seconds = time.perf_counter() - started
    users, items, truth = test.uir_tuple
    prediction = np.fromiter((model.rate(int(user), int(item)) for user, item in zip(users, items)),
                             dtype=np.float64, count=len(truth))
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    if not ((prediction >= train.min_rating) & (prediction <= train.max_rating)).all():
        raise ValueError('Cornac rate() did not clip to the training rating range')
    identities, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    errors = prediction - truth
    rmse = np.sqrt(np.bincount(inverse, weights=errors ** 2) / counts)
    mae = np.bincount(inverse, weights=np.abs(errors)) / counts
    cornac_means, cornac_users = rating_eval(model, [cornac.metrics.RMSE(), cornac.metrics.MAE()],
                                           test, user_based=True)
    for values, expected, overall in zip((rmse, mae), cornac_users, cornac_means):
        np.testing.assert_allclose(values, [expected[user] for user in identities], rtol=0, atol=1e-12)
        np.testing.assert_allclose(values.mean(), overall, rtol=0, atol=1e-12)
    np.savez_compressed(directory / 'predictions.npz', prediction=prediction)
    np.savez_compressed(directory / 'per_user.npz', users=identities, RMSE=rmse, MAE=mae)
    record = dict(signature=signature, dataset=CASES[dataset]['label'], method=method,
                  RMSE=float(rmse.mean()), MAE=float(mae.mean()), users=len(identities),
                  train_ratings=train.num_ratings, test_ratings=test.num_ratings,
                  rating_range=[float(train.min_rating), float(train.max_rating)],
                  fit_seconds=fit_seconds, total_seconds=time.perf_counter() - started,
                  effective_num_threads=getattr(model, 'num_threads', None),
                  pmf_variant=getattr(model, 'variant', None),
                  cornac_user_macro_audit='passed',
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  artifact_sha256={filename: digest(directory / filename) for filename in
                                   ('predictions.npz', 'per_user.npz')})
    write_json(metrics_path, record)
    print(json.dumps(record), flush=True)
    return record


def publish(output, results, records):
    scores_path = results / 'recsys/ratings.csv'
    archive = output / 'ratings_before_rerun.csv'
    if not archive.exists():
        shutil.copyfile(scores_path, archive)
    with archive.open() as stream:
        historical = {(row['dataset'], row['method']): row for row in csv.DictReader(stream)}
    with scores_path.open() as stream:
        ratings = list(csv.DictReader(stream))
    comparisons = []
    for record in records:
        selected = [row for row in ratings if row['dataset'] == record['dataset'] and row['method'] == record['method']]
        if len(selected) != 1:
            raise ValueError('Expected exactly one baseline row to replace')
        for metric in ('RMSE', 'MAE'):
            old = float(historical[record['dataset'], record['method']][metric])
            new = record[metric]
            comparisons.append(dict(dataset=record['dataset'], method=record['method'], metric=metric,
                                    historical=old, rerun=new, difference=new - old,
                                    agrees_at_four_decimals=f'{old:.4f}' == f'{new:.4f}'))
            selected[0][metric] = str(new)
        selected[0]['source_kind'] = 'frozen_split_baseline_rerun_full_precision'
    for path, rows in ((scores_path, ratings), (output / 'historical_comparison.csv', comparisons)):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    provenance_path = results / 'recsys/provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['small_baselines'] = ('SVD, PMF, UserKNN, ItemKNN rerun with original settings on frozen splits; '
                                     'per-entry predictions independently rescored; older MF/BPR rows preserved')
    provenance['small_baseline_rerun'] = dict(manifest=str(output.relative_to(results) / 'manifest.json'),
                                             manifest_sha256=digest(output / 'manifest.json'),
                                             historical_ratings_sha256=digest(archive))
    write_json(provenance_path, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--output', type=Path, help=f'Standalone output directory (default: {DEFAULT_OUTPUT})')
    location.add_argument('--paper-root', '--root', dest='paper_root', type=Path,
                          help='Optional paper repository; write into results/recsys/small_baseline_rerun')
    parser.add_argument('--split-source', type=Path, default=SPLITS)
    parser.add_argument('--publish', action='store_true', help='Replace only the eight displayed small-baseline rows')
    arguments = parser.parse_args()
    if arguments.publish and arguments.paper_root is None:
        parser.error('--publish requires --paper-root')
    results = arguments.paper_root / 'results' if arguments.paper_root is not None else None
    output = results / 'recsys/small_baseline_rerun' if results is not None else (arguments.output or DEFAULT_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, case in CASES.items():
        directory = output / dataset
        directory.mkdir(exist_ok=True)
        split_path = directory / 'split.npz'
        if not split_path.exists():
            source = arguments.split_source / case['source'] / 'split.npz'
            if digest(source) != case['sha256']:
                raise ValueError(f'Frozen split source hash mismatch: {source}')
            shutil.copyfile(source, split_path)
        if digest(split_path) != case['sha256']:
            raise ValueError(f'Frozen split hash mismatch: {split_path}')
        for method in CONFIGS:
            write_json(output / 'status.json', dict(state='running', dataset=dataset, method=method))
            records.append(fit_case(dataset, method, directory / method))
    uncertainty = export_intervals(output, records)
    write_json(output / 'manifest.json', dict(cases=records, uncertainty=uncertainty,
                                            split_sha256={name: case['sha256'] for name, case in CASES.items()},
                                            protocol='Original Cornac settings, seed 123; unchanged frozen splits and ordering; '
                                                     'macro-average per-user RMSE/MAE; rate() clips to training range; '
                                                     'no neural fits, ranking baselines, or hyperparameter selection'))
    if arguments.publish:
        publish(output, results, records)
    write_json(output / 'status.json', dict(state='complete', fits=len(records), published=arguments.publish))


if __name__ == '__main__':
    main()
