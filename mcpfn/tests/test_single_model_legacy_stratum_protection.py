from copy import deepcopy

import numpy as np
import pytest

import single_model_legacy_stratum_protection as protection


def metrics():
    return {f"{cohort}_draw_{draw}/{kind}": 1.
            for draw in range(2) for cohort in ("legacy", "mar_block_neural", "recsys")
            for kind in ("rmse", "mae" if cohort == "recsys" else "common_nrmse")}


def test_legacy_regression_cannot_hide_behind_anchor_improvement():
    before, after = metrics(), metrics()
    after["legacy_draw_0/common_nrmse"] = 1.1
    after["legacy_draw_1/common_nrmse"] = .8
    assert protection.template.metric_decision(before, after)["accepted"]
    assert not protection.metric_decision(before, after)["accepted"]


def test_other_cohorts_keep_original_paired_rule_and_individual_diagnostics():
    before, after = metrics(), metrics()
    after["mar_block_neural_draw_0/common_nrmse"] = 1.1
    after["mar_block_neural_draw_1/common_nrmse"] = .8
    result = protection.metric_decision(before, after)
    assert result["accepted"]
    assert result["individual_draw_diagnostics"] == protection.original.metric_decision(before, after)


def test_raw_gradients_unchanged_and_only_legacy_pair_is_split():
    before = metrics()
    original_labels, original_names, original_matrix = protection.template.gradient_layout(before, 6)
    labels, names, matrix = protection.gradient_layout(before, 6)
    assert labels == original_labels and matrix.shape == (12, 18)
    for index, name in enumerate(names):
        if name.startswith("legacy_draw_"):
            assert np.count_nonzero(matrix[index]) == 1 and matrix[index].sum() == 1
        else:
            np.testing.assert_array_equal(matrix[index], original_matrix[original_names.index(name)])


def test_truth_defined_null_remains_null_without_fabricating_metric_or_gradient():
    values = metrics()
    values["legacy_draw_0/common_nrmse"] = None
    assert protection.mean_metrics(values)["legacy_draw_0/common_nrmse"] is None
    labels, names, matrix = protection.gradient_layout(values, 6)
    assert ("legacy_draw_0", "common_nrmse") not in labels
    assert "legacy_draw_0/common_nrmse" not in names and matrix.shape == (11, 17)
    changed = deepcopy(values)
    changed["legacy_draw_1/rmse"] = .9
    assert protection.metric_decision(values, changed)["accepted"]


def test_projection_certificate_binds_new_scope_and_all_raw_coordinates():
    before = metrics()
    gram = np.eye(18)
    products = np.ones(18)
    result = protection.paired_solve(gram, products, 18., before, 6, .1)
    assert result["scope"] == "legacy_strata_other_paired_means"
    protection.paired_verify(gram, products, 18., result, .1, before, 6)
    result["scope"] = "paired_means"
    with pytest.raises(ValueError):
        protection.paired_verify(gram, products, 18., result, .1, before, 6)


def test_update_and_auditor_use_same_new_protection_not_original_pairing():
    assert protection.protected_update.__globals__["metric_decision"] is protection.metric_decision
    assert protection.check_record.__globals__["metric_decision"] is protection.metric_decision
    assert protection.check_endpoint.__globals__["paired_displacement"] is protection.paired_displacement
