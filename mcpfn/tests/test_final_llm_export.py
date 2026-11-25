import numpy as np
import pandas as pd
import pytest

from export_final_llm_results import DATASETS, matched_mi_subset, results_text, summary, validate_table


def complete_table():
    return pd.DataFrame([
        dict(dataset=dataset, selection=selection, budget=budget, fold=fold, method=method,
             rmse=score + .01 * fold, relative_rmse=score / 2 + .005 * fold, n_eval=fold+10)
        for dataset, _ in DATASETS for selection in ('mi', 'entropy', 'random')
        for budget in range(1, 16) for fold in range(10)
        for method, score in [('matrixformer', .5), ('kernel_ridge', 1.)]])


def test_requires_complete_shared_queries_and_unchanged_baseline():
    table = complete_table()
    validate_table(table, table)
    with pytest.raises(ValueError, match='Missing'):
        validate_table(table.iloc[1:], table)
    changed = table.copy()
    changed.loc[0, 'n_eval'] += 1
    with pytest.raises(ValueError, match='Unequal'):
        validate_table(changed, table)
    changed = table.copy()
    changed.loc[1, 'rmse'] += .01
    with pytest.raises(AssertionError):
        validate_table(changed, table)


def test_summary_is_fold_mean_with_fold_bootstrap():
    table = complete_table()
    panels = summary(table)
    assert len(panels) == 360
    selected = panels[panels.method == 'matrixformer']
    np.testing.assert_allclose(selected.rmse, .545)
    assert (selected.ci_lower < selected.rmse).all()
    assert (selected.ci_upper > selected.rmse).all()
    with pytest.raises(ValueError, match='ten folds'):
        summary(table.iloc[1:])


def test_results_prose_reports_every_dataset_without_local_references():
    panels = summary(complete_table())
    logit = pd.DataFrame([dict(dataset=dataset, budget=15, method='logit_ridge', rmse=1.)
                          for dataset in ('merged', 'benchpress') for fold in range(10)])
    text = results_text(panels, logit)
    assert 'in 180:' in text
    for _, label in DATASETS:
        assert f'45 of 45 on {label}' in text
    assert '/root/' not in text and '/mnt/' not in text
    assert 'MatrixFormer-final' not in text


def test_mi_archive_without_selection_or_query_count_columns():
    table = complete_table()
    subset = table.query("dataset in ['merged', 'benchpress'] and selection == 'mi' and budget in [5, 10, 15]")
    logit = subset[subset.method == 'kernel_ridge'].assign(method='logit_ridge')
    archive = pd.concat([subset, logit]).drop(columns=['selection', 'n_eval'])
    updated = matched_mi_subset(archive, table)
    assert len(updated) == 180 and set(updated.selection) == {'mi'}
    assert 'n_eval' not in updated
    with pytest.raises(ValueError, match='coverage'):
        matched_mi_subset(archive.iloc[1:], table)
