import pandas as pd
import pytest

from export_final_manuscript_results import METHODS, PATTERNS, hierarchical_scores


def test_hierarchy_weights_rates_and_patterns_not_number_of_masks():
    rows = []
    for pattern, rate, repeats, score in [('MCAR', .1, 1, 1.), ('MCAR', .2, 5, 3.),
                                          ('MNARPolarizationPattern', .2, 1, 10.)]:
        for repeat in range(repeats):
            rows.append(dict(suite='missbench', dataset='example', method='matrixformer',
                             pattern=pattern, missingness=rate, repeat=repeat, nrmse=score))
    by_pattern, overall, by_rate = hierarchical_scores(pd.DataFrame(rows))
    assert by_pattern.set_index('pattern').loc['MCAR', 'nrmse'] == 2.
    assert overall.nrmse.item() == 6.
    assert by_rate.set_index('missingness').loc[.2, 'nrmse'] == 6.5
    assert by_rate.set_index('missingness').loc[.1, 'nrmse'] == 1.


def test_hierarchy_preserves_independent_datasets_and_methods():
    rows = [dict(suite=suite, dataset=dataset, method=method, pattern='MCAR',
                 missingness=.4, nrmse=score)
            for suite in ['missbench', 'uci'] for dataset in ['first', 'second']
            for method, score in [('matrixformer', 1.), ('mice', 3.)]]
    by_pattern, overall, by_rate = hierarchical_scores(pd.DataFrame(rows))
    for frame in [by_pattern, overall, by_rate]:
        assert len(frame) == 8
        assert frame.groupby('method').nrmse.mean().to_dict() == pytest.approx(
            dict(matrixformer=1., mice=3.))


def test_final_method_and_pattern_labels():
    assert len(dict(METHODS)) == 10
    assert dict(METHODS)['notmiwae'] == 'not-MIWAE'
    assert 'tabimpute' not in dict(METHODS)
    assert len(dict(PATTERNS)) == 11
    assert dict(PATTERNS)['MNARLatentFactorPattern'] == 'Latent-MCAR'
    assert dict(PATTERNS)['MNARClusterLevelPattern'] == 'Clust-MCAR'
    assert dict(PATTERNS)['MNARTwoPhaseSubsetPattern'] == 'Phase-MAR'
    assert dict(PATTERNS)['MAR_Neural'] == 'NN-MNAR'
