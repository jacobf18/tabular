import csv
import json
import sys

import numpy as np
import pytest

import rerun_small_recsys_baselines as runner


@pytest.fixture
def frozen_split(tmp_path):
    directory = tmp_path / '100K'
    directory.mkdir()
    path = directory / 'split.npz'
    np.savez(path, train_users=[2, 0, 1, 2, 0, 1], train_items=[2, 0, 1, 0, 2, 2],
             train_ratings=[5.0, 2.0, 3.0, 4.0, 1.0, 2.0], test_users=[1, 0, 2],
             test_items=[0, 1, 1], test_ratings=[4.0, 3.0, 2.0], shape=[3, 3])
    return path


def test_frozen_dataset_preserves_indices_and_training_order(frozen_split):
    train, test = runner.frozen_datasets(frozen_split)
    with np.load(frozen_split) as saved:
        for subset, dataset in [('train', train), ('test', test)]:
            for field, actual in zip(('users', 'items', 'ratings'), dataset.uir_tuple):
                np.testing.assert_array_equal(actual, saved[f'{subset}_{field}'])
    assert train.uid_map == {0: 0, 1: 1, 2: 2}
    assert train.iid_map == {0: 0, 1: 1, 2: 2}


@pytest.mark.parametrize('problem', ['overlap', 'duplicate'])
def test_frozen_dataset_rejects_leakage_and_duplicates(frozen_split, problem):
    with np.load(frozen_split) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if problem == 'overlap':
        arrays['test_items'][0] = 1
    else:
        arrays['test_users'][1] = arrays['test_users'][0]
        arrays['test_items'][1] = arrays['test_items'][0]
    np.savez(frozen_split, **arrays)
    with pytest.raises(ValueError, match='leakage|Duplicate'):
        runner.frozen_datasets(frozen_split)


def test_bootstrap_matches_paper_seaborn_protocol():
    algorithms = pytest.importorskip('seaborn.algorithms')
    values = np.array([0.1, 0.3, 0.5, 0.9, 2.0])
    reference = algorithms.bootstrap(values, func=np.mean, n_boot=1000, seed=0)
    np.testing.assert_array_equal(runner.user_bootstrap_interval(values), np.percentile(reference, [2.5, 97.5]))


@pytest.mark.parametrize('values', [[], [1.0], [1.0, np.nan], [np.inf, 0.0], [[1.0, 2.0]]])
def test_bootstrap_rejects_invalid_user_scores(values):
    with pytest.raises(ValueError, match='finite user scores'):
        runner.user_bootstrap_interval(values)


@pytest.mark.parametrize('replicates', [0, -1, 1.5])
def test_bootstrap_rejects_invalid_replicates(replicates):
    with pytest.raises(ValueError, match='replicate count'):
        runner.user_bootstrap_interval([1.0, 2.0], replicates=replicates)


def test_constant_user_scores_have_zero_width_interval():
    assert runner.user_bootstrap_interval([2.0, 2.0]) == (2.0, 2.0)


def test_fit_resumes_and_exports_verified_intervals(frozen_split):
    directory = frozen_split.parent / 'SVD'
    record = runner.fit_case('100K', 'SVD', directory)
    assert record == runner.fit_case('100K', 'SVD', directory)
    assert record['cornac_user_macro_audit'] == 'passed'
    output = frozen_split.parent.parent
    uncertainty = runner.export_intervals(output, [record])
    assert uncertainty['intervals'] == 2
    with (output / 'summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    with np.load(directory / 'per_user.npz') as saved:
        for row in summary:
            values = saved[row['metric']]
            assert row['unit'] == 'test_user' and int(row['units']) == 3
            np.testing.assert_allclose(float(row['value']), values.mean(), rtol=0, atol=1e-12)
            np.testing.assert_array_equal([float(row['ci95_low']), float(row['ci95_high'])],
                                          runner.user_bootstrap_interval(values))
    with (directory / 'predictions.npz').open('ab') as stream:
        stream.write(b'changed artifact')
    with pytest.raises(ValueError, match='artifact mismatch'):
        runner.fit_case('100K', 'SVD', directory)


def test_macro_metrics_give_users_equal_weight(tmp_path, monkeypatch):
    class ConstantModel:
        def __init__(self, **configuration):
            self.configuration = configuration

        def fit(self, train):
            return self

        def rate(self, user, item):
            return np.float64(1.0)

    directory = tmp_path / '100K'
    directory.mkdir()
    np.savez(directory / 'split.npz', train_users=[0, 0, 1], train_items=[0, 2, 1],
             train_ratings=[1.0, 3.0, 5.0], test_users=[0, 1, 1], test_items=[1, 0, 2],
             test_ratings=[5.0, 1.0, 1.0], shape=[2, 3])
    monkeypatch.setattr(runner.cornac.models, 'SVD', ConstantModel)
    record = runner.fit_case('100K', 'SVD', directory / 'SVD')
    assert record['RMSE'] == record['MAE'] == 2.0
    assert record['RMSE'] != np.sqrt(16.0 / 3.0)


def test_publish_preserves_unrelated_score_strings_and_archive(tmp_path):
    results = tmp_path / 'results'
    output = results / 'recsys/small_baseline_rerun'
    output.mkdir(parents=True)
    scores = results / 'recsys/ratings.csv'
    original = ('dataset,method,normalization,RMSE,MAE,source_kind\n'
                'MovieLens 100K,SVD,method_specific,0.8998,0.743,old\n'
                'MovieLens 100K,MatrixFormer+,expert_specific,0.12345678901234567,0.7278777597350297,old\n')
    scores.write_text(original)
    (output / 'manifest.json').write_text('{}\n')
    (results / 'recsys/provenance.json').write_text('{}\n')
    records = [dict(dataset='MovieLens 100K', method='SVD', RMSE=0.899831175719489, MAE=0.7430130014630236)]
    runner.publish(output, results, records)
    runner.publish(output, results, records)
    assert (output / 'ratings_before_rerun.csv').read_text() == original
    assert scores.read_text().splitlines()[-1] == original.splitlines()[-1]
    provenance = json.loads((results / 'recsys/provenance.json').read_text())
    assert provenance['small_baseline_rerun']['manifest_sha256'] == runner.digest(output / 'manifest.json')


def test_publishing_requires_explicit_paper_root(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['rerun_small_recsys_baselines.py', '--publish'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_only_rating_baselines_are_enabled():
    assert set(runner.CONFIGS) == {'SVD', 'PMF', 'UserKNN', 'ItemKNN'}
    assert all(config['seed'] == 123 for config in runner.CONFIGS.values())
