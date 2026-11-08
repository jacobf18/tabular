import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from evaluate_scientific_uci import DATASETS, PATTERNS, aggregate, ci_half_width, select_uci_cases


def full_protocol():
    cases = [dict(case_id=str(index), suite='uci', dataset=dataset, pattern=pattern, missingness=rate, repeat=repeat)
             for index, (dataset, pattern, rate, repeat) in enumerate(
                 itertools.product(DATASETS, PATTERNS, (.1, .2, .3, .4, .5), range(10)))]
    return dict(stages=dict(tabular=cases + [dict(suite='missbench')]))


def test_full_inventory():
    assert len(select_uci_cases(full_protocol())) == 1800


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_rate', 'same_id'])
def test_reject_incomplete_or_duplicated_inventory(change):
    protocol = full_protocol()
    cases = protocol['stages']['tabular']
    if change == 'missing':
        cases.pop(0)
    elif change == 'duplicate':
        cases[0] = cases[1].copy()
    elif change == 'wrong_rate':
        cases[0]['missingness'] = .9
    else:
        cases[0]['case_id'] = cases[1]['case_id']
    with pytest.raises(ValueError):
        select_uci_cases(protocol)


def test_equal_dataset_weights_and_dataset_level_intervals():
    rows = [dict(case_id='a', dataset='first', original_nrmse=2., candidate_nrmse=1.)]
    rows += [dict(case_id=f'b{repeat}', dataset='second', original_nrmse=4., candidate_nrmse=3.) for repeat in range(10)]
    result = aggregate(rows)[0]
    assert result['original_nrmse'] == 3.
    assert result['candidate_nrmse'] == 2.
    assert result['original_ci95'] == pytest.approx(ci_half_width([2., 4.]))
    assert result['delta_ci95'] == 0.
    assert result['improved_datasets'] == result['datasets'] == 2
    assert result['cases'] == 11
    by_dataset = aggregate(rows, ('dataset',))
    assert all(row['candidate_ci95'] is None for row in by_dataset)


@pytest.mark.parametrize('invalid', ['duplicate', 'nonfinite', 'empty'])
def test_reject_invalid_scores(invalid):
    rows = [dict(case_id='a', dataset='first', original_nrmse=2., candidate_nrmse=1.)]
    if invalid == 'duplicate':
        rows *= 2
    elif invalid == 'nonfinite':
        rows[0]['candidate_nrmse'] = np.nan
    else:
        rows = []
    with pytest.raises(ValueError):
        aggregate(rows)
