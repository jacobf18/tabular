from copy import deepcopy
import pytest

import run_single_model_paired_decoder_pipeline as pipeline
import run_single_model_replicated_decoder_pipeline as previous
import train_single_matrixformer_paired_decoder as training


def test_full_confirmation_and_real_campaign_coverage_remain_unchanged():
    assert pipeline.training is training and pipeline.training_audit.training is training
    assert pipeline.PARENT_CACHE == previous.PARENT_CACHE
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("paired_decoder_v19")
        assert getattr(pipeline, name) != getattr(previous, name)
    for expression in ('audit["training_episodes"] != 1792', 'audit["prediction_grids"] != 1216',
                       'expected_count = 28 if payload["record"]["point_after"] is None else 56',
                       'len(bank) != 284', 'count != 1136', '"--stage", "audit-confirmation"',
                       'scripts/run_single_matrixformer_cached_benchmarks.py', 'scripts/review_single_matrixformer_campaign.py'):
        assert expression in pipeline.ADAPTED_SOURCE


def test_endpoint_replay_and_independent_score_audit_use_the_declared_training_rule():
    assert "training.paired.paired_displacement(" in pipeline.ADAPTED_SOURCE
    assert pipeline.ADAPTED_SOURCE.count("training.paired.metric_decision(") == 2
    assert "training.original.guarded.metric_decision(" not in pipeline.ADAPTED_SOURCE
    for name in ("confirm", "audit_confirmation"):
        current = getattr(pipeline, name).__code__
        original = getattr(previous, name).__code__
        assert current.co_code == original.co_code
        assert current.co_consts == original.co_consts


def test_confirmation_cannot_be_reused_reselected_or_substituted():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    wrong = deepcopy(reservation)
    wrong["confirmation_cases"] = previous.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(wrong, "pipeline", "run", True)
    assert pipeline.no_confirmation_reason({"step": 0}, {"selected_validation_passed": True}) is not None
    assert pipeline.no_confirmation_reason({"step": 50}, {"selected_validation_passed": False}) is not None
    assert pipeline.no_confirmation_reason({"step": 50}, {"selected_validation_passed": True}) is None
