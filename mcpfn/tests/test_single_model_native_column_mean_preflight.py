from copy import deepcopy
import inspect

import pytest

import preflight_single_model_native_column_mean as probe
import preflight_single_model_general_native_retention as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_fresh_batch_retains_all_original_tasks_masses_and_model():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1231100901, 1232100901)
    assert probe.OUTPUT.name == "native_column_mean_preflight_v1"
    assert probe.SCRATCH.name == "native_column_mean_preflight_v1"
    assert probe.PARENT == predecessor.PARENT
    assert probe.PARENT.parent.name == "legacy_strata_v29"
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    before, after = predecessor.batch(), probe.batch()
    assert len(after["tasks"]) == 28
    assert {key: value for key, value in before.items() if key != "tasks"} == {key: value for key, value in after.items() if key != "tasks"}
    for original, actual in zip(before["tasks"], after["tasks"]):
        assert (actual["name"], actual["weight"], actual["cohort"], actual["index"]) == (original["name"], original["weight"], original["cohort"], original["index"])
        assert actual["gradient_seed"] - original["gradient_seed"] == 30000000
    assert probe.blend_replay.storage_accounting(after) == predecessor.blend_replay.storage_accounting(before)
    assert probe.ARITHMETIC == predecessor.ARITHMETIC
    assert probe.RESERVES == predecessor.RESERVES and probe.WRITE_BUDGETS == predecessor.WRITE_BUDGETS
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger


def test_native_mean_formula_is_bound_at_registration_and_runtime():
    assert probe.retention is probe.cohort_retention
    assert probe.retention.__name__ == "single_model_native_column_mean"
    assert probe.reference_identities() == predecessor.reference_identities()
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'config["native_teacher_mean"] != retention.MEAN_SPECIFICATION' in probe.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in probe.ADAPTED_SOURCE
    assert 'require_predecessor' in probe.register_trial.__code__.co_names
    assert 'require_completed_v47' in probe.require_predecessor.__code__.co_names
    assert probe.require_predecessor.__globals__ is vars(probe)
    assert 'require_completed_v46()' not in probe.NATIVE_PROBE_SOURCE
    assert 'test_single_model_native_column_mean.py' in probe.REGISTER_SOURCE
    assert 'test_single_model_native_column_mean_preflight.py' in probe.REGISTER_SOURCE
    assert probe.replay_cohort_targets.__code__ is predecessor.replay_cohort_targets.__code__
    assert probe.replay_cohort_targets.__globals__ is vars(probe)


def test_all_adapted_functions_have_resolved_correct_globals():
    for value in vars(probe).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(probe):
            assert unresolved_globals(value) == []
    assert predecessor.retention.__name__ == "single_model_general_native_retention"
    assert predecessor.DATA_SEED == 1201100901
    assert predecessor.stronger.original.balanced_loss is predecessor.stronger.BASE_LOSS


def diagnostic_fixture():
    summary = {"complete": True, "cases": 760, "imputation_columns": 135775, "empty_query_columns": 21835,
        "stored_metric_checks": 8160, "boundaries": [0, 10, 25, 50, 64], "artifact_sha256": {},
        "config_sha256": probe.DEVELOPMENT_HASHES["config.json"]}
    summary.update({key: False for key in ("cuda_initialized", "inference_performed", "training_performed",
        "confirmation_evaluated", "checkpoint_selection_performed", "gate_changed", "real_data_used")})
    audit = {"complete": True, "case_metric_checks": 2880, "imputation_case_boundaries": 680, "column_boundaries": 135775,
        "summary_sha256": probe.DEVELOPMENT_HASHES["summary.json"], "csv_regrouping_recomputed": True,
        "input_and_source_hashes_rechecked": True, "cuda_initialized": False, "raw_grid_replay_performed": False}
    return {"summary.json": summary, "audit.json": audit, "status.json": {"state": "complete"},
        "config.json": {"input_sha256": {}, "source_sha256": {}}}


@pytest.mark.parametrize("mutation", [None, "partial_cases", "partial_columns", "omitted_empty", "omitted_boundary", "selected",
    "confirmation", "inference", "partial_recheck", "wrong_summary", "raw_replay", "incomplete", "previous_closure"])
def test_only_complete_closed_v47_diagnosis_allows_admission(monkeypatch, mutation):
    records = diagnostic_fixture()
    changes = {"partial_cases": ("summary.json", "cases", 759), "partial_columns": ("summary.json", "imputation_columns", 135774),
        "omitted_empty": ("summary.json", "empty_query_columns", 0), "omitted_boundary": ("summary.json", "boundaries", [0, 10, 25, 64]),
        "selected": ("summary.json", "checkpoint_selection_performed", True), "confirmation": ("summary.json", "confirmation_evaluated", True),
        "inference": ("summary.json", "inference_performed", True), "partial_recheck": ("audit.json", "case_metric_checks", 2879),
        "wrong_summary": ("audit.json", "summary_sha256", "changed"), "raw_replay": ("audit.json", "raw_grid_replay_performed", True),
        "incomplete": ("status.json", "state", "analyzing")}
    if mutation in changes:
        filename, field, value = changes[mutation]
        records[filename][field] = value

    def closed():
        if mutation == "previous_closure":
            raise ValueError("Original V47 closure must be verified")
        return {}, [], [], [], {"original_closed_evidence": "bound"}

    monkeypatch.setattr(probe.outcome, "require_closed", closed)
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "digest")
    monkeypatch.setattr(probe, "read", lambda path: deepcopy(records[path.name]))
    if mutation is None:
        evidence = probe.require_completed_v47()
        assert evidence["original_closed_evidence"] == "bound"
        for filename, digest in probe.DEVELOPMENT_HASHES.items():
            assert evidence[str(probe.development.OUTPUT / filename)] == digest
    else:
        with pytest.raises(ValueError):
            probe.require_completed_v47()
