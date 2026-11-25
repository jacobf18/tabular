import numpy as np
import pytest

from distributed_imputation import digest, read, write
from plot_repeated_imputation_preview import bootstrap_bounds, format_value_labels, overlay_mice_retry, recompute_score, select_rate
from repeated_imputation_statistics import dataset_scores


def test_score_uses_held_out_not_observed_standard_deviation():
    truth = np.array([[0., 2.], [2., 4.], [100., 100.], [200., 200.]])
    mask = np.array([[True, True], [True, True], [False, False], [False, False]])
    normalized, raw = recompute_score(truth, mask, truth[mask] + 1, 'uci')
    assert normalized == 1 and raw == 1


def test_constant_column_rule_matches_historical_scorer():
    truth = np.array([[1., 0.], [1., 2.], [1., 100.]])
    mask = np.array([[True, True], [True, True], [False, False]])
    normalized, unused = recompute_score(truth, mask, np.array([2., 1., 2., 3.]), 'missbench')
    assert normalized == 1
    normalized, unused = recompute_score(truth, mask, np.array([1., 1., 1., 3.]), 'missbench')
    assert normalized == .5


def test_invalid_mask_is_not_a_zero_error():
    with pytest.raises(ValueError):
        recompute_score(np.ones((2, 2)), np.zeros((2, 2), dtype=bool), np.array([]), 'uci')


def test_patterns_are_equal_weighted_not_replicate_weighted():
    cases, results = [], []
    for pattern, repeats, value in [('stochastic', 5, 1.), ('deterministic', 1, 7.)]:
        for repeat in range(repeats):
            case_id = f'{pattern}/{repeat}'
            cases.append(dict(case_id=case_id, suite='uci', dataset='dataset', pattern=pattern,
                              family='MCAR', missingness=.4))
            results.append(dict(case_id=case_id, method='miwae', status='ok', nrmse=value))
    scores, unused = dataset_scores(cases, results, ['miwae'], shared_methods=True)
    assert next(row['nrmse'] for row in scores if row['panel'] == 'Overall') == 4
    results[-1]['status'] = 'error'
    scores, coverage = dataset_scores(cases, results, ['miwae'], shared_methods=True)
    assert not any(row['panel'] == 'Overall' for row in scores)
    assert coverage[0]['failed'] == 1


def test_bootstrap_is_deterministic_and_dataset_level():
    assert bootstrap_bounds([2., 2., 2.]) == (2., 2.)
    assert bootstrap_bounds([1., 2., 9.]) == bootstrap_bounds([1., 2., 9.])


def test_value_label_formatting_preserves_bar_values_and_flags():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    bar = axis.barh(0, .8975383145)[0]
    label = axis.text(.92, 0, '0.8975†', fontsize=6.5)
    missing = axis.text(.01, 1, '—')
    format_value_labels(figure)
    assert label.get_text() == '0.898†'
    assert label.get_fontsize() == 7.3
    assert missing.get_text() == '—'
    assert bar.get_width() == .8975383145
    plt.close(figure)


def test_value_labels_round_raw_bar_values_not_existing_text():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    bar = axis.barh(0, 1.9946)[0]
    label = axis.text(2.1, 0, '1.995‡')
    format_value_labels(figure)
    assert label.get_text() == '1.995‡'
    assert bar.get_width() == 1.9946
    plt.close(figure)


def test_rate_selection_happens_before_dataset_averaging():
    cases, records = [], []
    for suite, rates in [('missbench', [.4]), ('uci', [.1, .4, .5])]:
        for rate in rates:
            identifier = f'{suite}/{rate}'
            cases.append(dict(case_id=identifier, suite=suite, dataset='example', pattern='MCAR',
                              family='MCAR', missingness=rate))
            records.append(dict(case_id=identifier, method='matrixformer', status='ok',
                                nrmse=1. if rate == .4 else 100.))
    selected, results = select_rate(cases, records, .4)
    scores, coverage = dataset_scores(selected, results, ['matrixformer'], shared_methods=True)
    assert len(selected) == len(results) == 2
    assert all(row['nrmse'] == 1 for row in scores)
    assert all(row['complete'] for row in coverage)
    assert len(cases) == len(records) == 4
    with pytest.raises(ValueError, match='every suite'):
        select_rate(cases, records, .1)


@pytest.fixture
def mice_retry_inputs(tmp_path, request):
    base, retry = tmp_path / 'base', tmp_path / 'retry'
    base.mkdir()
    retry.mkdir()
    truth_path = tmp_path / 'truth.npy'
    truth = np.array([[0.], [2.], [4.], [6.]])
    np.save(truth_path, truth)
    rate = getattr(request, 'param', .4)
    case = dict(case_id=f'uci/example/MCAR/rate_{rate}/seed_0', suite='uci', missingness=rate,
                truth_path=str(truth_path), truth_sha256=digest(truth_path))
    original = dict(cases=[case], methods=['mice'], timeout_seconds=1800, scoring='unchanged')
    write(base / 'manifest.json', original)
    mask = np.array([[True], [True], [False], [False]])
    mask_path = base / 'masks' / (case['case_id'] + '.npz')
    mask_path.parent.mkdir(parents=True)
    np.savez(mask_path, packed=np.packbits(mask), shape=truth.shape)
    (retry / 'masks').symlink_to(base / 'masks', target_is_directory=True)
    masks = {case['case_id']: digest(mask_path)}
    prior = dict(case, method='mice', status='timeout', manifest_sha256=digest(base / 'manifest.json'),
                 mask_sha256=digest(mask_path))
    prior_path = base / 'results/mice/cases' / (case['case_id'] + '.json')
    write(prior_path, prior)
    lineage = dict(parent=str(base), parent_manifest_sha256=digest(base / 'manifest.json'),
                   case_ids=[case['case_id']], parent_result_sha256={str(prior_path): digest(prior_path)})
    if rate != .4:
        lineage['selected_missingness'] = None
    manifest = dict(original, timeout_seconds=7200, timeout_retry=lineage)
    write(retry / 'manifest.json', manifest)
    prediction_path = retry / 'results/mice/predictions' / (case['case_id'] + '.npz')
    prediction_path.parent.mkdir(parents=True)
    np.savez(prediction_path, missing_values=truth[mask] + 1., shape=truth.shape)
    result = dict(case, method='mice', status='ok', nrmse=1., rmse=1.,
                  manifest_sha256=digest(retry / 'manifest.json'), mask_sha256=digest(mask_path),
                  prediction_sha256=digest(prediction_path))
    write(retry / 'results/mice/cases' / (case['case_id'] + '.json'), result)
    records = [prior, dict(case, method='knn', status='ok', nrmse=2.)]
    return base, retry, original, masks, records


def test_retry_overlay_replaces_only_timeout_and_preserves_archives(mice_retry_inputs):
    base, retry, original, masks, records = mice_retry_inputs
    before = {path: digest(path) for path in base.rglob('*') if path.is_file()}
    updated, checks, receipt = overlay_mice_retry(*mice_retry_inputs)
    assert records[0]['status'] == 'timeout'
    assert updated[0]['status'] == 'ok' and updated[0]['nrmse'] == 1.
    assert updated[1] == records[1]
    assert len(updated) == len(records) and len(checks) == 1
    assert receipt['original_records_unchanged'] and receipt['predictions_verified'] == 1
    assert before == {path: digest(path) for path in before}


@pytest.mark.parametrize('mice_retry_inputs', [.1, .2, .3, .5], indirect=True)
def test_all_rate_retry_overlay_recovers_other_registered_rates(mice_retry_inputs):
    updated, checks, receipt = overlay_mice_retry(*mice_retry_inputs)
    assert updated[0]['status'] == 'ok'
    assert len(checks) == receipt['predictions_verified'] == 1


@pytest.mark.parametrize('changed', ['parent', 'protocol', 'case', 'prediction', 'successful_original'])
def test_retry_overlay_rejects_changed_inputs(mice_retry_inputs, changed):
    base, retry, original, masks, records = mice_retry_inputs
    manifest = read(retry / 'manifest.json')
    if changed == 'parent':
        manifest['timeout_retry']['parent_manifest_sha256'] = 'wrong'
    elif changed == 'protocol':
        manifest['scoring'] = 'different'
    elif changed == 'case':
        manifest['cases'][0]['missingness'] = .5
    elif changed == 'prediction':
        prediction = next((retry / 'results/mice/predictions').rglob('*.npz'))
        prediction.write_bytes(b'changed')
    else:
        records[0]['status'] = 'ok'
    write(retry / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        overlay_mice_retry(*mice_retry_inputs)
