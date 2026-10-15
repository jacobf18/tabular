from copy import deepcopy

import pytest

import run_single_model_stronger_decoder_pipeline as previous
import run_single_model_v8_decoder_pipeline as pipeline
import train_single_matrixformer_v8_decoder as training


def test_new_paths_preserve_all_qualification_stages_and_cached_parent_controls():
    assert pipeline.training is training
    assert pipeline.training_audit.training is training
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("v8_decoder_v17")
        assert getattr(pipeline, name) != getattr(previous, name)
    assert pipeline.PARENT_CACHE == previous.PARENT_CACHE
    source = pipeline.ADAPTED_SOURCE
    assert 'audit["training_episodes"] != 896' in source
    assert 'audit["prediction_grids"] != 1216' in source
    assert 'len(bank) != 284' in source
    assert '"--stage", "audit-confirmation"' in source
    assert 'scripts/run_single_matrixformer_cached_benchmarks.py' in source
    assert 'scripts/review_single_matrixformer_campaign.py' in source


def test_reserved_confirmation_is_new_and_cannot_be_substituted():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    wrong = deepcopy(reservation)
    wrong["confirmation_cases"] = previous.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(wrong, "pipeline", "run", True)


def test_no_eligible_selection_or_failed_replay_can_consume_confirmation():
    assert pipeline.no_confirmation_reason({"step": 0}, {"selected_validation_passed": True}) is not None
    assert pipeline.no_confirmation_reason({"step": 10}, {"selected_validation_passed": False}) is not None
    assert pipeline.no_confirmation_reason({"step": 10}, {"selected_validation_passed": True}) is None
