"""Verify saved SVD++ predictions and refresh Figure 1 and the rating appendix."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distributed_imputation import digest
from export_final_manuscript_results import module_from
from plot_final_four_quadrant_preview import draw, draw_panel
from plot_repeated_imputation_preview import bootstrap_bounds


DATASETS = {'100K': 'MovieLens 100K', 'netflix': 'Netflix small'}
METRICS = ('RMSE', 'MAE')
KEYS = ['domain', 'dataset', 'metric', 'method']
ARTIFACTS = ['manifest.json', 'summary.csv'] + [
    f'{dataset}/{filename}' for dataset in DATASETS
    for filename in ('selection.json', 'SVDpp/metrics.json',
                     'SVDpp/per_user.npz', 'SVDpp/predictions.npz')]


def read_json(path):
    return json.loads(path.read_text())


def read_scores(path):
    return pd.read_csv(path, float_precision='round_trip')


def verified_scores(source, paper):
    manifest = read_json(source / 'manifest.json')
    summary = read_scores(source / 'summary.csv')
    expected = {(label, 'SVD++', metric) for label in DATASETS.values() for metric in METRICS}
    actual = set(summary[['dataset', 'method', 'metric']].itertuples(index=False, name=None))
    if len(summary) != len(expected) or actual != expected:
        raise ValueError('Expected exactly four SVD++ summary rows for the two rating datasets')
    rows = []
    for dataset, label in DATASETS.items():
        directory = source / dataset / 'SVDpp'
        record = read_json(directory / 'metrics.json')
        selection = read_json(source / dataset / 'selection.json')
        if record['manifest_sha256'] != digest(source / 'manifest.json'):
            raise ValueError('SVD++ source manifest changed')
        if record['config'] != selection['config']:
            raise ValueError('Final SVD++ fit does not use the validation-selected configuration')
        for filename, expected_hash in record['artifact_sha256'].items():
            if digest(directory / filename) != expected_hash:
                raise ValueError(f'SVD++ artifact changed: {dataset}/{filename}')
        split_path = paper / 'results/recsys/small_baseline_rerun' / dataset / 'split.npz'
        if digest(split_path) != manifest['datasets'][dataset]['sha256']:
            raise ValueError('SVD++ and the manuscript baselines use different rating splits')
        baseline_path = paper / f'results/accuracy_uncertainty/recsys_{dataset}_per_user.npz'
        with np.load(split_path, allow_pickle=False) as split, \
                np.load(directory / 'predictions.npz', allow_pickle=False) as predictions, \
                np.load(directory / 'per_user.npz', allow_pickle=False) as scores, \
                np.load(baseline_path, allow_pickle=False) as baseline:
            users, inverse, counts = np.unique(split['test_users'], return_inverse=True, return_counts=True)
            np.testing.assert_array_equal(users, scores['users'])
            np.testing.assert_array_equal(users, baseline['users'])
            prediction = predictions['prediction']
            if prediction.shape != split['test_ratings'].shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid SVD++ predictions')
            if len(users) != record['users'] or len(prediction) != record['test_ratings']:
                raise ValueError('SVD++ evaluation support differs from the recorded fit')
            errors = prediction - split['test_ratings']
            for metric in METRICS:
                squared = metric == 'RMSE'
                recomputed = np.bincount(inverse, weights=errors**2 if squared else abs(errors)) / counts
                if squared:
                    recomputed = np.sqrt(recomputed)
                np.testing.assert_allclose(recomputed, scores[metric], atol=1e-12, rtol=0)
                lower, upper = bootstrap_bounds(recomputed)
                value = recomputed.mean()
                saved = summary[(summary.dataset == label) & (summary.metric == metric)].iloc[0]
                if (saved.units, saved.bootstrap_replicates, saved.bootstrap_seed, saved.ci_status) != (
                        len(users), 1000, 0, 'available'):
                    raise ValueError('Unexpected SVD++ confidence-interval protocol')
                np.testing.assert_allclose(value, record[metric], atol=1e-12, rtol=0)
                np.testing.assert_allclose([value, lower, upper],
                                           [saved.value, saved.ci95_low, saved.ci95_high], atol=1e-12, rtol=0)
                rows.append(dict(domain='recsys', dataset=label, metric=metric, method='SVD++',
                                 label='SVD++', value=value, ci_lower=lower, ci_upper=upper,
                                 units=len(users), available=True))
    return pd.DataFrame(rows)


def merge_scores(existing, addition):
    if existing.duplicated(KEYS).any() or addition.duplicated(KEYS).any():
        raise ValueError('Duplicate result keys')
    if not addition.method.eq('SVD++').all() or not addition.domain.eq('recsys').all():
        raise ValueError('Only SVD++ rating results may be added')
    retained = existing[~((existing.domain == 'recsys') & (existing.method == 'SVD++'))]
    return pd.concat([retained, addition.reindex(columns=existing.columns)], ignore_index=True)


def plot_appendix(values, paper, style):
    style.apply()
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 2.9))
    panels = [(dataset, metric) for dataset in DATASETS.values() for metric in METRICS]
    for axis, (dataset, metric) in zip(axes, panels):
        selected = values[(values.dataset == dataset) & (values.metric == metric)]
        draw_panel(axis, selected, style, limits=(.6, 1.09, [.6, .8, 1.], 3))
        axis.set_title(f'{dataset}\n{metric}', fontsize=11)
        axis.set_xlabel('Error (lower is better)', fontsize=10)
    figure.tight_layout(pad=.6, w_pad=1.)
    style.save(figure, paper / 'figures/previews/table1_recommendation_bars_four_columns')
    plt.close(figure)


def refresh(paper, source=None):
    portable = paper / 'results/recsys/svdpp'
    if source is not None:
        verified_scores(source, paper)
        for filename in ARTIFACTS:
            destination = portable / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (source / filename).resolve() != destination.resolve():
                shutil.copy2(source / filename, destination)
    addition = verified_scores(portable, paper)
    data = paper / 'results/final_model'
    ratings = merge_scores(read_scores(data / 'recsys_summary.csv'), addition)
    figure_values = merge_scores(read_scores(data / 'figure_values.csv'), addition[addition.metric == 'RMSE'])
    ratings.to_csv(data / 'recsys_summary.csv', index=False)
    figure_values.to_csv(data / 'figure_values.csv', index=False)
    style = module_from(paper / 'scripts/figure_style.py')
    draw(figure_values, paper / 'figures', style, '10–50% missing (rate-averaged), all 11 patterns')
    plot_appendix(ratings, paper, style)
    subprocess.run([
        sys.executable, str(paper / 'scripts/plot_unification_vertical.py'),
        '--source', str(paper / 'figures/unification.pdf'),
        '--rating-results', str(data / 'recsys_summary.csv'),
        '--output', str(paper / 'figures/unification_final'),
    ], check=True)
    provenance = dict(
        checkpoint_sha256=read_json(data / 'validation.json')['checkpoint_sha256'],
        generator='mcpfn/scripts/add_svdpp_manuscript_results.py',
        source_artifact_sha256={filename: digest(portable / filename) for filename in ARTIFACTS},
        figure_input_sha256={filename: digest(data / filename)
                             for filename in ('figure_values.csv', 'recsys_summary.csv')},
        verified=['identical frozen splits', 'identical test users', 'per-entry rescoring',
                  'macro-user RMSE and MAE', '1000 test-user bootstrap resamples, seed 0'],
        method_order='per panel, ascending error',
        svdpp_tuning='Four train-only validation candidates; refit the selected candidate on all training ratings',
    )
    (data / 'svdpp_validation.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (paper / 'figures/four_quadrant_summary_sorted.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(addition.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--import-source', type=Path)
    args = parser.parse_args()
    refresh(args.paper, args.import_source)


if __name__ == '__main__':
    main()
