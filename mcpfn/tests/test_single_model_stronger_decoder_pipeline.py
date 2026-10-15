from copy import deepcopy

import pytest

import run_single_model_stronger_decoder_pipeline as pipeline
import run_single_model_targeted_decoder_pipeline as original
import train_single_matrixformer_stronger_decoder as training


def test_all_reserved_outputs_are_new_without_changing_original_pipeline():
    assert pipeline.training is training
    assert pipeline.training_audit.training is training
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name) != getattr(original, name)
        assert getattr(pipeline, name).name.startswith("stronger_decoder_v16")
    assert pipeline.PARENT_CACHE == original.PARENT_CACHE
    assert pipeline.POLICY == original.POLICY.replace("one hundred", "sixty-four")


def test_confirmation_reservation_cannot_change_or_reuse_v15_cases():
    current = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
               "confirmation_cases": training.confirmation_cases()}
    pipeline.verify_reservation(current, "pipeline", "run", True)
    altered = deepcopy(current)
    altered["confirmation_cases"] = original.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(altered, "pipeline", "run", True)


def test_new_horizon_keeps_every_raw_grid_and_complete_frozen_campaign():
    source = pipeline.ADAPTED_SOURCE
    assert 'audit["through_step"] != 64' in source
    assert 'audit["training_episodes"] != 896' in source
    assert 'audit["prediction_grids"] != 1368' in source
    assert '"--stage", "audit-confirmation"' in source
    assert 'scripts/run_single_matrixformer_cached_benchmarks.py' in source
    assert 'scripts/review_single_matrixformer_campaign.py' in source
    assert 'training.original.alignment.fixed.PARENT' in source


def test_missing_qualified_checkpoint_never_consumes_confirmation():
    reason = pipeline.no_confirmation_reason({"step": 0}, {"selected_validation_passed": False})
    assert reason == original.no_confirmation_reason({"step": 0}, {"selected_validation_passed": False})
    assert reason is not None
