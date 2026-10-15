from copy import deepcopy
import inspect

import pytest

import preflight_single_model_parent_structured_retention as probe
import preflight_single_model_native_column_mean as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_fresh_batch_preserves_complete_mixture_and_full_network():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1261100901, 1262100901)
    assert probe.OUTPUT.name == probe.SCRATCH.name == "parent_structured_retention_preflight_v1"
    assert probe.PARENT == predecessor.PARENT
    assert probe.PARENT.parent.name == "legacy_strata_v29"
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    before, after = predecessor.batch(), probe.batch()
    assert len(after["tasks"]) == 28
    expected = deepcopy(before)
    for task in expected["tasks"]:
        task["gradient_seed"] += 30000000
        specification = task["specification"]
        specification.get("structured", specification)["seed"] += 30000000
    assert after == expected
    assert probe.blend_replay.storage_accounting(after) == predecessor.blend_replay.storage_accounting(before)
    assert probe.ARITHMETIC == predecessor.ARITHMETIC
    assert probe.RESERVES == predecessor.RESERVES and probe.WRITE_BUDGETS == predecessor.WRITE_BUDGETS
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger
    assert probe.RATE == predecessor.RATE == 5e-6


def test_only_structured_training_reference_changes_not_original_evaluation_baseline():
    identities = probe.reference_identities()
    assert set(identities) == {"v29", "general"}
    assert identities["general"] == predecessor.reference_identities()["general"]
    assert identities["v29"] == {"checkpoint": str(probe.PARENT), "sha256": probe.PARENT_SHA256}
    assert probe.retention is probe.cohort_retention
    assert probe.retention.__name__ == "single_model_parent_structured_retention"
    assert probe.retention.BACKWARD_SOURCE == predecessor.retention.BACKWARD_SOURCE
    assert '{"general": 2, "v29": 24}' in probe.PARENT_PROBE_SOURCE
    assert 'Path(reference_identities()["v29"]["checkpoint"])' in probe.REGISTER_SOURCE
    assert '"retention_baseline_for_future_training": str(training.alignment.fixed.PARENT)' in probe.REGISTER_SOURCE
    assert probe.training.alignment.fixed.PARENT == predecessor.training.alignment.fixed.PARENT
    assert probe.training.alignment.fixed.PARENT.parent.name == "query_budget_v8"
    assert probe.replay_cohort_targets.__code__ is predecessor.replay_cohort_targets.__code__
    assert probe.replay_cohort_targets.__globals__ is vars(probe)


def test_native_mean_and_new_closure_are_bound_everywhere():
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'config["native_teacher_mean"] != retention.MEAN_SPECIFICATION' in probe.ADAPTED_SOURCE
    assert '"mean_distillation": stronger.SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'config["mean_distillation"] != stronger.SPECIFICATION' in probe.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in probe.ADAPTED_SOURCE
    assert 'require_predecessor' in probe.register_trial.__code__.co_names
    assert 'require_completed_v48' in probe.require_predecessor.__code__.co_names
    assert 'require_completed_v47()' not in probe.PARENT_PROBE_SOURCE
    assert probe.require_predecessor.__globals__ is vars(probe)
    for name in ("parent_structured_retention", "parent_structured_retention_preflight", "native_column_mean"):
        assert f'test_single_model_{name}.py' in probe.REGISTER_SOURCE


def test_adapted_globals_are_resolved_and_predecessor_is_unchanged():
    for value in vars(probe).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(probe):
            assert unresolved_globals(value) == []
    assert predecessor.retention.__name__ == "single_model_native_column_mean"
    assert predecessor.DATA_SEED == 1231100901
    assert set(predecessor.reference_identities()) == {"v8", "general"}
    assert predecessor.stronger.original.balanced_loss is predecessor.stronger.BASE_LOSS
    assert probe.main.__globals__ is vars(probe)


def complete_diagnosis():
    rows = [{"reference": "general", "group": group, "metric": "rmse",
             "initializer_failed": True, "candidate_failed": True} for group in ("imputation", "structured_overall")]
    rows.append({"reference": "V8", "group": "imputation", "metric": "rmse",
                 "initializer_failed": True, "candidate_failed": False})
    rows.extend({"reference": "V8", "group": group, "metric": "common_nrmse",
                 "initializer_failed": False, "candidate_failed": True}
                for group in ("mnar_two_phase_subset", "mnar_skip_logic"))
    rows.extend({"reference": "V8", "group": f"passing_{index}", "metric": "metric",
                 "initializer_failed": False, "candidate_failed": False} for index in range(27))
    report = {"complete": True, "prediction_grids": 284, "comparisons": rows,
              "post_hoc_consumed_bank": True, "goal_success_not_assessed": True, "artifact_sha256": {}}
    report.update({key: False for key in ("cuda_initialized", "training_performed", "confirmation_repeated",
                                        "checkpoint_selection_performed", "real_benchmarks_evaluated")})
    return report


@pytest.mark.parametrize("mutation", [None, "partial_grids", "partial_comparisons", "inherited", "introduced", "repaired",
    "live", "incomplete", "fresh_label", "trained", "repeated", "selected", "real", "cuda", "goal_claim", "closure"])
def test_requires_complete_negative_closure_and_unchanged_all32_attribution(monkeypatch, mutation):
    report = complete_diagnosis()
    changes = {"partial_grids": ("prediction_grids", 283), "fresh_label": ("post_hoc_consumed_bank", False),
        "trained": ("training_performed", True), "repeated": ("confirmation_repeated", True),
        "selected": ("checkpoint_selection_performed", True), "real": ("real_benchmarks_evaluated", True),
        "cuda": ("cuda_initialized", True), "goal_claim": ("goal_success_not_assessed", False)}
    if mutation in changes:
        field, value = changes[mutation]
        report[field] = value
    if mutation == "partial_comparisons":
        report["comparisons"].pop()
    if mutation == "inherited":
        report["comparisons"][0]["initializer_failed"] = False
    if mutation == "introduced":
        report["comparisons"][3]["candidate_failed"] = False
    if mutation == "repaired":
        report["comparisons"][2]["candidate_failed"] = True

    def verify_closure():
        if mutation == "closure":
            raise ValueError("Verify the original fully audited failed V48 gate")

    monkeypatch.setattr(probe.diagnosis, "verify", verify_closure)
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: {} if mutation == "live" else None)
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "digest")
    monkeypatch.setattr(probe, "read", lambda path: deepcopy(report) if path.name == "audit.json" else {
        "state": "running" if mutation == "incomplete" else "complete", "pid": 123})
    if mutation is None:
        evidence = probe.require_completed_v48()
        for filename, digest in probe.DIAGNOSTIC_HASHES.items():
            assert evidence[str(probe.diagnosis.OUTPUT / filename)] == digest
    else:
        with pytest.raises(ValueError):
            probe.require_completed_v48()
