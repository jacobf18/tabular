import json

import pytest
import torch

import evaluate_single_model_missingness_probe as diagnostic
from single_model_distillation import blend_predictions


def case():
    target = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    query = torch.zeros_like(target, dtype=torch.bool)
    query[:2] = True
    source = target.masked_fill(query, torch.nan)
    predictions = {role: target + index for index, role in enumerate(diagnostic.ROLES)}
    return source, target, query, predictions


def test_all_models_use_same_query_coverage_and_blend_gap():
    scores = diagnostic.score_predictions(*case())
    assert scores["general"]["rmse"] == 0. and scores["blend"]["query_blend_rmse"] == 0.
    assert len({json.dumps(result["column_coverage"], sort_keys=True) for result in scores.values()}) == 1


@pytest.mark.parametrize("problem", ["missing_role", "nonfinite_observed", "no_queries", "no_observations"])
def test_never_omit_invalid_model_cases(problem):
    source, target, query, predictions = case()
    if problem == "missing_role":
        del predictions["specialist"]
    elif problem == "nonfinite_observed":
        predictions["v8"][-1, -1] = float("nan")
    elif problem == "no_queries":
        query[:] = False
    else:
        source[:] = float("nan")
    with pytest.raises(ValueError):
        diagnostic.score_predictions(source, target, query, predictions)


def test_router_does_not_receive_hidden_truth():
    source, target, query, predictions = case()
    first, decision = blend_predictions(source.unsqueeze(0), predictions["general"].unsqueeze(0),
                                         predictions["specialist"].unsqueeze(0))
    target[query] = 10000.
    second, repeated = blend_predictions(source.unsqueeze(0), predictions["general"].unsqueeze(0),
                                          predictions["specialist"].unsqueeze(0))
    assert torch.equal(first, second) and decision == repeated


def test_summary_requires_every_family_context_and_role():
    records = [{"context": context, "family": family, "role": role,
                **dict.fromkeys(diagnostic.METRICS, 1.)}
               for context in range(16) for family in diagnostic.probe.FAMILIES for role in diagnostic.ROLES]
    result = diagnostic.summarize(records)
    assert len(result) == 65 and result[0]["cases"] == 192
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.summarize(records[:-1])
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.summarize(records + [records[0]])
    records[0]["missbench_nrmse"] = None
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.summarize(records)


@pytest.mark.parametrize("state,ready", [("evaluating", False), ("complete", True)])
def test_waits_for_entire_real_campaign(tmp_path, state, ready):
    (tmp_path / "status.json").write_text(json.dumps({"state": state}))
    (tmp_path / "campaign_audit.json").write_text(json.dumps({"all_evaluations_complete": True}))
    assert diagnostic.campaign_ready(tmp_path) == ready


def test_failed_campaign_is_not_silently_bypassed(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(ValueError, match="bypass"):
        diagnostic.campaign_ready(tmp_path)
