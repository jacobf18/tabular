from copy import deepcopy

import pytest

import run_single_model_replicated_decoder_pipeline as pipeline
import run_single_model_v8_decoder_pipeline as previous
import train_single_matrixformer_replicated_decoder as training


def test_larger_training_audit_does_not_narrow_confirmation_or_real_campaigns():
    assert pipeline.training is training
    assert pipeline.training_audit.training is training
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("replicated_decoder_v18")
        assert getattr(pipeline, name) != getattr(previous, name)
    assert pipeline.PARENT_CACHE == previous.PARENT_CACHE
    source = pipeline.ADAPTED_SOURCE
    for expression in ('audit["training_episodes"] != 1792', 'audit["prediction_grids"] != 1216',
                       'expected_count = 28 if payload["record"]["point_after"] is None else 56',
                       'len(bank) != 284', 'count != 1136', '"--stage", "audit-confirmation"',
                       'scripts/run_single_matrixformer_cached_benchmarks.py', 'scripts/review_single_matrixformer_campaign.py'):
        assert expression in source


def test_confirmation_reservation_is_new_and_cannot_be_replaced():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    wrong = deepcopy(reservation)
    wrong["confirmation_cases"] = previous.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(wrong, "pipeline", "run", True)


def test_no_fallback_can_consume_confirmation():
    assert pipeline.no_confirmation_reason({"step": 0}, {"selected_validation_passed": True}) is not None
    assert pipeline.no_confirmation_reason({"step": 50}, {"selected_validation_passed": False}) is not None
    assert pipeline.no_confirmation_reason({"step": 50}, {"selected_validation_passed": True}) is None
