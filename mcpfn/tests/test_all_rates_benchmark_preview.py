import numpy as np
import pytest

from plot_all_rates_benchmark_preview import exclusion_reason, exclude_unscorable, incomplete_rate_groups
from plot_final_four_quadrant_preview import tabular_description
from repeated_imputation_statistics import dataset_scores


def test_eligibility_depends_on_truth_and_mask_not_model_error():
    truth = np.array([[1., 0.], [1., 2.], [1., 4.]])
    mask = np.array([[True, False], [True, False], [False, False]])
    assert exclusion_reason(truth, mask, 'missbench').startswith('No variable')
    mask[:2, 1] = True
    assert exclusion_reason(truth, mask, 'missbench') is None
    assert exclusion_reason(truth, np.zeros_like(mask), 'missbench').startswith('No held-out')


def test_shared_exclusion_keeps_method_specific_timeout(tmp_path):
    truth_path = tmp_path / 'truth.npy'
    np.save(truth_path, np.array([[1., 0.], [1., 2.], [1., 4.]]))
    (tmp_path / 'masks').mkdir()
    cases, records = [], []
    for identifier, mask in [('undefined', [[True, False], [True, False], [False, False]]),
                             ('valid', [[True, True], [True, True], [False, False]])]:
        np.savez(tmp_path / 'masks' / (identifier + '.npz'), packed=np.packbits(mask), shape=(3, 2))
        cases.append(dict(case_id=identifier, truth_path=str(truth_path), suite='missbench'))
        for method in ['matrixformer', 'mice']:
            status = 'error' if identifier == 'undefined' else ('timeout' if method == 'mice' else 'ok')
            records.append(dict(case_id=identifier, method=method, status=status))
    selected, results, excluded = exclude_unscorable(tmp_path, cases, records, ['matrixformer', 'mice'])
    assert [case['case_id'] for case in selected] == ['valid']
    assert len(results) == 2 and {row['status'] for row in results} == {'ok', 'timeout'}
    assert len(excluded) == 1 and excluded[0]['applies_to_methods'] == ['matrixformer', 'mice']
    assert len(cases) == 2 and len(records) == 4
    records[0]['status'] = 'ok'
    with pytest.raises(ValueError, match='inconsistent'):
        exclude_unscorable(tmp_path, cases, records, ['matrixformer', 'mice'])


def test_rate_means_equal_weighted_despite_different_seed_counts():
    cases, records = [], []
    for rate, repeats, score in [(.1, 1, 1.), (.2, 5, 3.)]:
        for repeat in range(repeats):
            identifier = f'{rate}/{repeat}'
            cases.append(dict(case_id=identifier, suite='missbench', dataset='example', pattern='MCAR',
                              family='MCAR', missingness=rate))
            records.append(dict(case_id=identifier, method='matrixformer', status='ok', nrmse=score))
    scores, coverage = dataset_scores(cases, records, ['matrixformer'], shared_methods=True)
    assert next(row['nrmse'] for row in scores if row['panel'] == 'Overall') == 2.
    assert coverage[0]['complete']
    exceptions = incomplete_rate_groups(cases, cases[1:])
    assert exceptions == [dict(suite='missbench', dataset='example', pattern='MCAR',
                               registered_rates=[.1, .2], eligible_rates=[.2])]


def test_plot_caption_tracks_rate_selection():
    assert tabular_description(dict(selected_missingness=.4)).startswith('40%')
    assert 'rate-averaged' in tabular_description(dict(selected_missingness=None,
                                                     averaged_missingness_rates=[.1, .2, .3, .4, .5]))
    with pytest.raises(ValueError):
        tabular_description(dict(selected_missingness=None))
