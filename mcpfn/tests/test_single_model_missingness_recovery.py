from copy import deepcopy

import pytest
import torch

import recover_single_model_missingness_probe as recovery


def case():
    target = torch.tensor([[1., 2.], [1., 4.], [1., 6.], [1., 8.]])
    query = torch.tensor([[True, True], [False, True], [False, False], [False, False]])
    source = target.masked_fill(query, torch.nan)
    predictions = {role: target + .1 for role in recovery.original.ROLES}
    predictions["general"][0, 0] = 1.
    return source, target, query, predictions


def test_canonical_constant_column_semantics_are_retained_not_made_truth_only():
    values = case()
    with pytest.raises(ValueError, match="coverage"):
        recovery.original.score_predictions(*values)
    scores = recovery.score_predictions(*values)
    assert scores["general"]["column_coverage"]["missbench_scored_columns"] == 2
    assert scores["v8"]["column_coverage"]["missbench_scored_columns"] == 1
    assert all(result["column_coverage"]["common_nonconstant_columns"] == 1 for result in scores.values())
    episode = recovery.original.Episode(values[0].unsqueeze(0), values[1].unsqueeze(0), values[2].unsqueeze(0),
                                        {"domain": "imputation", "gradient_data": "synthetic"})
    for role, prediction in values[3].items():
        original = recovery.original.episode_metrics(prediction.unsqueeze(0), episode)
        assert all(scores[role][name] == value for name, value in original.items())


def test_fully_defined_previous_scores_do_not_change():
    source, target, query, predictions = case()
    target[0, 0] = 2.
    target[1, 0] = 5.
    query[1, 0] = True
    source = target.masked_fill(query, torch.nan)
    assert recovery.score_predictions(source, target, query, predictions) == recovery.original.score_predictions(source, target, query, predictions)


def test_nonfinite_observed_prediction_still_rejected():
    source, target, query, predictions = case()
    predictions["v8"][-1, -1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        recovery.score_predictions(source, target, query, predictions)


def test_truth_only_guard_cannot_be_waived(monkeypatch):
    previous = recovery.original.episode_metrics
    count = 0

    def changed(prediction, episode):
        nonlocal count
        result = deepcopy(previous(prediction, episode))
        result["column_coverage"]["common_nonconstant_columns"] += count
        count += 1
        return result

    monkeypatch.setattr(recovery.original, "episode_metrics", changed)
    with pytest.raises(ValueError, match="Truth-defined"):
        recovery.score_predictions(*case())
