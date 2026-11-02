import numpy as np
import pandas as pd
import pytest

import summarize_single_matrixformer_orientation as summary
from recsys_evaluation_common import sha256, write_json
from summarize_single_matrixformer_orientation import bootstrap, tabular_summaries, user_errors


def test_bootstrap_is_paired_and_uses_units_not_cells():
    values = np.array([[1., 2., 1.5], [3., 4., 3.5], [5., 6., 5.5]])
    result = bootstrap(values)
    assert result[0]['mean'] == 3
    for key in ('delta_standard', 'delta_ci_low', 'delta_ci_high'):
        assert result[1][key] == pytest.approx(1)
        assert result[2][key] == pytest.approx(0.5)
    assert result[0]['units'] == 3
    with pytest.raises(ValueError):
        bootstrap(np.ones((2, 2)))


def test_tabular_weights_datasets_equally_and_intervals_use_dataset_count():
    rows = []
    for dataset, count, value in [('small', 1, 1.), ('large', 9, 3.)]:
        for index in range(count):
            for variant in ['standard', 'transpose', 'average']:
                rows.append({'case_id': f'{dataset}/{index}', 'suite': 'uci', 'dataset': dataset,
                             'pattern': 'MCAR', 'variant': variant, 'nrmse': value})
    summary, _ = tabular_summaries(pd.DataFrame(rows))
    assert (summary['mean'] == 2).all()
    assert (summary.units == 2).all()


def test_rating_rmse_is_user_macro_not_pooled():
    users = np.array([0, 0, 0, 1])
    truth = np.zeros(4)
    prediction = np.array([1., 1., 1., 3.])
    identities, metrics = user_errors(users, truth, prediction)
    np.testing.assert_array_equal(identities, [0, 1])
    assert metrics['RMSE'].mean() == 2
    assert metrics['MAE'].mean() == 2


@pytest.fixture
def scoped_campaign(tmp_path):
    manifest = {'protocols': {'recsys': {'cases': [
        {'dataset': dataset} for dataset in ['100K', 'netflix', '1M', '10M']]}}}
    write_json(tmp_path / 'manifest.json', manifest)
    scope = {'version': 1, 'manifest_sha256': sha256(tmp_path / 'manifest.json'),
             'excluded_datasets': {'recsys': ['10M']}, 'user_request': 'Exclude MovieLens 10M',
             'reason': 'Not used in the paper'}
    for stage, count in [('tabular', 16), ('llm', 12), ('recsys', 1)]:
        (tmp_path / stage).mkdir()
        write_json(tmp_path / stage / 'status.json', {'state': 'complete'})
        write_json(tmp_path / stage / 'parity.json', {'checks': [
            {'case_id': str(index), 'bit_identical': True, 'max_abs_difference': 0}
            for index in range(count)]})
    return tmp_path, manifest, scope


def test_reporting_scope_keeps_all_original_cases_without_override(scoped_campaign):
    output, manifest, _ = scoped_campaign
    cases, scope = summary.reporting_scope(output, manifest)
    assert len(cases) == 4
    assert scope is None


def test_reporting_scope_only_excludes_requested_dataset(scoped_campaign):
    output, manifest, scope = scoped_campaign
    write_json(output / 'scope_override.json', scope)
    cases, actual = summary.reporting_scope(output, manifest)
    assert [case['dataset'] for case in cases] == ['100K', 'netflix', '1M']
    assert actual == scope
    assert len(manifest['protocols']['recsys']['cases']) == 4


@pytest.mark.parametrize('change', [
    {'manifest_sha256': 'incorrect'},
    {'excluded_datasets': {'recsys': ['10M', 'netflix']}},
    {'excluded_datasets': {'tabular': ['uci']}},
    {'user_request': ''},
])
def test_reporting_scope_rejects_unapproved_or_unbound_changes(scoped_campaign, change):
    output, manifest, scope = scoped_campaign
    write_json(output / 'scope_override.json', scope | change)
    with pytest.raises(ValueError):
        summary.reporting_scope(output, manifest)


def test_reporting_scope_rejects_missing_required_dataset(scoped_campaign):
    output, manifest, scope = scoped_campaign
    manifest['protocols']['recsys']['cases'].pop(1)
    write_json(output / 'manifest.json', manifest)
    scope['manifest_sha256'] = sha256(output / 'manifest.json')
    write_json(output / 'scope_override.json', scope)
    with pytest.raises(ValueError, match='all four original'):
        summary.reporting_scope(output, manifest)


def test_completion_accepts_only_the_explicitly_cancelled_dataset(scoped_campaign):
    output, _, scope = scoped_campaign
    status = {'state': 'evaluating', 'dataset': '10M', 'completed': 3, 'expected': 4}
    write_json(output / 'recsys/status.json', status)
    statuses, hashes = summary.verify_completion(output, scope)
    assert statuses['recsys'] == status
    assert set(hashes) == {'tabular', 'llm', 'recsys'}
    with pytest.raises(ValueError, match='not complete'):
        summary.verify_completion(output, None)
    write_json(output / 'recsys/status.json', status | {'dataset': '1M', 'completed': 2})
    with pytest.raises(ValueError, match='not complete'):
        summary.verify_completion(output, scope)


def test_missing_required_predictions_still_fail_audit(scoped_campaign, monkeypatch):
    output, manifest, _ = scoped_campaign
    monkeypatch.setattr(summary.campaign, 'cache', lambda *args: (None, None))
    with pytest.raises(ValueError, match='Missing result'):
        summary.audit_case(output, manifest, 'recsys', {'dataset': '1M'})


def test_parity_failure_prevents_summary(scoped_campaign):
    output, _, scope = scoped_campaign
    write_json(output / 'recsys/parity.json', {'checks': [
        {'case_id': '100K', 'bit_identical': False, 'max_abs_difference': 0.01}]})
    with pytest.raises(ValueError, match='parity checks failed'):
        summary.verify_completion(output, scope)
