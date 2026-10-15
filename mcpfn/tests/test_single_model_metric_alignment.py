from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audit_dwip_missbench_uci import load_original
import diagnose_single_model_metric_alignment as diagnostic


def episode(shape, prior, transposed=False):
    source = torch.zeros((1, *shape))
    mask = torch.zeros_like(source, dtype=torch.bool)
    mask[0, 0, 0] = True
    source[mask] = float("nan")
    return SimpleNamespace(source=source, target=torch.zeros_like(source), mask=mask,
                           metadata={"gradient_data": "synthetic", "domain": "imputation",
                                     "rows": shape[0], "columns": shape[1],
                                     "prior": prior, "transposed": transposed})


@pytest.mark.parametrize("shape,prior,transposed", [
    ((240, 80), "low_rank_mcar", False), ((80, 240), "slow_tail_mcar", True),
    ((24, 7), "anchor_latent_mcar", False), ((49, 14), "anchor_latent_mcar", True),
    ((35, 35), "anchor_latent_profile_mixed", False),
])
def test_full_spectral_and_intrinsic_anchor_shapes_are_valid(shape, prior, transposed):
    diagnostic.validate_episode(episode(shape, prior, transposed))


@pytest.mark.parametrize("failure", ["small_spectral", "large_anchor", "metadata", "observed_query", "empty"])
def test_context_guard_rejects_changed_shapes_or_invalid_queries(failure):
    case = episode((240, 80), "low_rank_mcar")
    if failure == "small_spectral":
        case = episode((24, 7), "low_rank_mcar")
    elif failure == "large_anchor":
        case = episode((2000, 1000), "anchor_latent_mcar")
    elif failure == "metadata":
        case.metadata["rows"] = 239
    elif failure == "observed_query":
        case.source[case.mask] = 0.
    else:
        case.mask[:] = False
    with pytest.raises(ValueError):
        diagnostic.validate_episode(case)


def scorers():
    return {suite: load_original(path, "compute_normalized_rmse_columnwise")
            for suite, path in diagnostic.SCORERS.items()}


def test_column_normalization_can_reverse_pooled_rmse_order():
    truth = np.array([[-100., -1.], [100., 1.], [0., 0.]])
    mask = np.array([[True, True], [True, True], [False, False]])
    first = diagnostic.score_queries(truth, truth + [1., 10.], mask, scorers())
    second = diagnostic.score_queries(truth, truth + [20., .1], mask, scorers())
    assert first["pooled_rmse"] < second["pooled_rmse"]
    for metric in ("missbench_nrmse", "uci_nrmse"):
        assert first[metric] == pytest.approx(5.005)
        assert second[metric] == pytest.approx(.15)
    assert first["queries"] == 4 and first["common_nonconstant_columns"] == 2


def test_scoring_uses_only_supervision_queries():
    truth = np.array([[-2., -1.], [2., 1.], [0., 0.]])
    mask = np.array([[True, True], [True, True], [False, False]])
    prediction = truth + .5
    first = diagnostic.score_queries(truth, prediction, mask, scorers())
    prediction[-1] = [1e9, -1e9]
    second = diagnostic.score_queries(truth, prediction, mask, scorers())
    assert first == second


def test_original_constant_column_rules_and_shared_coverage_are_both_visible():
    truth = np.array([[1., 2., -1.], [1., 2., 1.], [1., 2., 0.]])
    mask = np.array([[True, True, True], [True, True, True], [False, False, False]])
    prediction = truth + [0., 1., 1.]
    scores = diagnostic.score_queries(truth, prediction, mask, scorers())
    for suite in ("missbench", "uci"):
        assert scores[f"{suite}_nrmse"] == pytest.approx(.5)
        assert scores[f"{suite}_scored_columns"] == 2
        assert scores[f"{suite}_skipped_constant_columns"] == 1
    assert scores["common_nonconstant_nrmse"] == pytest.approx(1.)
    assert scores["common_nonconstant_columns"] == 1 and scores["queried_columns"] == 3


def test_invalid_or_empty_queries_are_rejected():
    truth = np.ones((3, 2))
    for mask in (np.zeros_like(truth, dtype=bool), np.ones_like(truth, dtype=int), np.ones((2, 2), dtype=bool)):
        with pytest.raises(ValueError, match="query mask"):
            diagnostic.score_queries(truth, truth, mask, scorers())


def cases():
    first = {"seed": 1, "gradient_data": "synthetic", "prior": "first", "scores": {
        role: {metric: 1. for metric in diagnostic.METRICS} for role in diagnostic.ROLES}}
    second = deepcopy(first)
    second.update(seed=2, prior="second")
    first["scores"]["observed_blend"].update(pooled_rmse=.9, missbench_nrmse=1.2, uci_nrmse=1.2)
    second["scores"]["observed_blend"].update(pooled_rmse=1.1, missbench_nrmse=.8, uci_nrmse=.8)
    return [first, second]


def test_summary_preserves_case_disagreements_hidden_by_aggregate_ties():
    rows = diagnostic.summarize(cases())
    aggregate = next(row for row in rows if row["family"] == "ALL" and row["role"] == "observed_blend")
    assert aggregate["pooled_rmse"] == 1. and aggregate["missbench_nrmse"] == 1.
    assert aggregate["missbench_opposite_direction_cases"] == aggregate["uci_opposite_direction_cases"] == 2
    assert aggregate["missbench_nrmse_wins_vs_general"] == aggregate["missbench_nrmse_losses_vs_general"] == 1
    assert len(rows) == 3 * 4


@pytest.mark.parametrize("failure", ["real", "nonfinite", "duplicate"])
def test_summary_rejects_non_synthetic_nonfinite_or_duplicate_cases(failure):
    data = cases()
    if failure == "real":
        data[0]["gradient_data"] = "real"
    elif failure == "nonfinite":
        data[0]["scores"]["general"]["pooled_rmse"] = float("nan")
    else:
        data[1]["seed"] = data[0]["seed"]
    with pytest.raises(ValueError):
        diagnostic.summarize(data)
