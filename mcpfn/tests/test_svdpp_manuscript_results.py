import json

import numpy as np
import pandas as pd
import pytest

from add_svdpp_manuscript_results import DATASETS, merge_scores, read_scores, verified_scores
from distributed_imputation import digest
from plot_repeated_imputation_preview import bootstrap_bounds


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def saved_results(tmp_path):
    source, paper = tmp_path / 'source', tmp_path / 'paper'
    manifest = {'datasets': {}}
    summaries = []
    config = dict(k=20, lambda_reg=.02)
    for dataset, label in DATASETS.items():
        directory = source / dataset / 'SVDpp'
        directory.mkdir(parents=True)
        split_path = paper / 'results/recsys/small_baseline_rerun' / dataset / 'split.npz'
        split_path.parent.mkdir(parents=True)
        np.savez(split_path, test_users=[0, 0, 1, 2], test_ratings=[2., 3., 4., 2.])
        manifest['datasets'][dataset] = dict(sha256=digest(split_path))
        np.savez(directory / 'predictions.npz', prediction=[2.25, 3.25, 4.5, 2.75])
        np.savez(directory / 'per_user.npz', users=[0, 1, 2], RMSE=[.25, .5, .75], MAE=[.25, .5, .75])
        baseline = paper / f'results/accuracy_uncertainty/recsys_{dataset}_per_user.npz'
        baseline.parent.mkdir(parents=True, exist_ok=True)
        np.savez(baseline, users=[0, 1, 2])
        write_json(source / dataset / 'selection.json', dict(config=config))
        lower, upper = bootstrap_bounds([.25, .5, .75])
        for metric in ['RMSE', 'MAE']:
            summaries.append(dict(dataset=label, method='SVD++', metric=metric, value=.5,
                                  ci95_low=lower, ci95_high=upper, units=3, bootstrap_replicates=1000,
                                  bootstrap_seed=0, ci_status='available'))
    write_json(source / 'manifest.json', manifest)
    for dataset in DATASETS:
        directory = source / dataset / 'SVDpp'
        write_json(directory / 'metrics.json', dict(
            manifest_sha256=digest(source / 'manifest.json'), config=config, users=3,
            test_ratings=4, RMSE=.5, MAE=.5,
            artifact_sha256={name: digest(directory / name) for name in ['predictions.npz', 'per_user.npz']}))
    pd.DataFrame(summaries).to_csv(source / 'summary.csv', index=False)
    return source, paper


def test_saved_predictions_and_intervals_are_independently_verified(saved_results):
    values = verified_scores(*saved_results)
    assert len(values) == 4
    assert set(values.method) == {'SVD++'}
    assert set(values.dataset) == set(DATASETS.values())
    assert values.value.eq(.5).all()
    assert values.available.all()


def test_merge_preserves_existing_results_and_is_idempotent(saved_results):
    addition = verified_scores(*saved_results)
    existing = addition.assign(method='matrixformer', label='MatrixFormer', value=.4)
    combined = merge_scores(existing, addition)
    pd.testing.assert_frame_equal(combined.iloc[:len(existing)], existing)
    pd.testing.assert_frame_equal(merge_scores(combined, addition), combined)
    with pytest.raises(ValueError, match='Duplicate'):
        merge_scores(pd.concat([existing, existing]), addition)


def test_score_csv_round_trip_preserves_full_precision(tmp_path):
    path = tmp_path / 'scores.csv'
    text = 'value,ci_lower,ci_upper\n0.9382067113063457,0.9749930514306961,18.832420568142396\n'
    path.write_text(text)
    read_scores(path).to_csv(path, index=False)
    assert path.read_text() == text


@pytest.mark.parametrize('failure', ['split', 'predictions', 'interval', 'missing'])
def test_inconsistent_saved_results_are_rejected(saved_results, failure):
    source, paper = saved_results
    if failure == 'split':
        path = paper / 'results/recsys/small_baseline_rerun/100K/split.npz'
        path.write_bytes(b'different split')
    elif failure == 'predictions':
        np.savez(source / '100K/SVDpp/predictions.npz', prediction=[1., 2., 3., 4.])
    else:
        summary = pd.read_csv(source / 'summary.csv')
        if failure == 'interval':
            summary.loc[0, 'ci95_low'] += .1
        else:
            summary = summary.iloc[1:]
        summary.to_csv(source / 'summary.csv', index=False)
    with pytest.raises((ValueError, AssertionError)):
        verified_scores(source, paper)
