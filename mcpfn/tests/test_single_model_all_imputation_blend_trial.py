from copy import deepcopy
import inspect
import types

import pytest

import test_single_model_uniform_v8_retention_trial as trial_test_blueprint
import test_single_model_cohort_anchor_retention_trial as complete_trial_blueprint
import train_single_matrixformer_uniform_v8_retention as previous_training
import train_single_matrixformer_all_imputation_blend as training


function = complete_trial_blueprint.fixed_preflight
fixed_preflight = types.FunctionType(function.__code__, globals(), "fixed_preflight", function.__defaults__)
ADAPTED_TEST_SOURCE = complete_trial_blueprint.ADAPTED_TEST_SOURCE.replace("cohort_anchor_retention", "all_imputation_blend")
for old, new in {"1061100901": "1151100901", "1062100901": "1152100901",
        "1071100901": "1161100901", "1072100901": "1162100901"}.items():
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, "exec"), globals())


def test_only_structured_teacher_assignment_changes_with_identical_priors_and_retention():
    assert training.OUTPUT.name == "all_imputation_blend_v45"
    assert training.PARENT == previous_training.PARENT and training.PARENT_SHA256 == previous_training.PARENT_SHA256
    assert training.retention is previous_training.retention and training.retention_training is previous_training.retention_training
    assert training.teacher_policy.POLICY == {"legacy": "observed_blend", "structured": "observed_blend", "recsys": "specialist"}
    assert previous_training.teacher_policy.POLICY["structured"] == "general"
    assert training.paired is previous_training.paired and training.RATE == previous_training.RATE
    assert training.TRIAL_QUALIFICATION == previous_training.TRIAL_QUALIFICATION
    assert training.diagnostics is previous_training.diagnostics and training.capacity.probe is training.preflight
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([
        previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, prior in zip(training.batches(), previous_training.batches()):
        training.retention.validate_task_mixture(batch["tasks"])
        for task, original in zip(batch["tasks"], prior["tasks"]):
            restored = deepcopy(task)
            restored["gradient_seed"] = original["gradient_seed"]
            specification = restored["specification"]
            specification.get("structured", specification)["seed"] = original["specification"].get(
                "structured", original["specification"])["seed"]
            assert restored == original
    before, after = deepcopy(previous_training.STORAGE_ACCOUNTING), deepcopy(training.STORAGE_ACCOUNTING)
    assert before["scratch"].pop("legacy_blend_component_allowance") == 2121943040
    assert after["scratch"].pop("all_imputation_blend_component_allowance") == 5005854720
    assert before == after
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) == 41224310912 < 40 * 1024**3
    assert training.blend_replay.teacher_policy is training.teacher_policy
    assert training.blend_replay.storage_accounting(training.batches())["raw_component_grids"] == 3328


def test_registration_binds_all_imputation_components_without_changing_qualification():
    assert "config[\"all_imputation_blend_replay\"]" in training.ADAPTED_SOURCE
    assert "legacy_blend_replay" not in training.REGISTER_SOURCE
    for name in ("trial", "audit", "pipeline", "teacher", "full_replay"):
        assert f"tests/test_single_model_all_imputation_blend_{name}.py" in training.REGISTER_SOURCE
    for name in ("train", "verify_registration", "register_trial", "setup"):
        assert getattr(training, name).__globals__["retention_training"] is training.retention_training
    assert "require_completed_v44" in training.RETENTION_EVIDENCE_SOURCE
    assert "2496316664" in training.RETENTION_EVIDENCE_SOURCE


def test_no_registration_before_external_capacity_receipt_is_bound(monkeypatch):
    monkeypatch.setattr(training, "CAPACITY_RECEIPT_SHA256", None)
    with pytest.raises(ValueError, match="storage-preservation receipt"):
        training.retention_evidence()


CAPACITY_TEST_SOURCE = inspect.getsource(trial_test_blueprint.test_capacity_binding_checks_actual_receipt_and_every_preserved_input)
CAPACITY_TEST_SOURCE = CAPACITY_TEST_SOURCE.replace("1652558524", "2496316664")
CAPACITY_TEST_SOURCE = CAPACITY_TEST_SOURCE.replace("'canonical_artifacts_checked': 3", "'canonical_artifacts_checked': 6")
CAPACITY_TEST_SOURCE = CAPACITY_TEST_SOURCE.replace("require_completed_v43", "require_completed_v44")
CAPACITY_TEST_SOURCE = CAPACITY_TEST_SOURCE.replace("    events = []", "    monkeypatch.setattr(training, 'CAPACITY_RECEIPT_SHA256', 'fixed-receipt')\n    events = []")
exec(compile(CAPACITY_TEST_SOURCE, __file__, "exec"), globals())
