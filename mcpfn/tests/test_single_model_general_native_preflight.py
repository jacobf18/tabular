from copy import deepcopy
import inspect

import pytest

import preflight_single_model_general_native_retention as probe
import preflight_single_model_stronger_mean_distillation as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_fresh_batch_preserves_original_full_mixture_and_ordinary_model():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1201100901, 1202100901)
    assert probe.OUTPUT.name == "general_native_retention_preflight_v1"
    assert probe.PARENT == predecessor.PARENT
    assert probe.PARENT.parent.name == "legacy_strata_v29"
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    batch = probe.batch()
    assert len(batch["tasks"]) == 28
    assert probe.blend_replay.storage_accounting(batch)["blend_tasks"] == 26
    assert probe.teacher_policy.POLICY == {"legacy": "observed_blend", "structured": "observed_blend", "recsys": "specialist"}
    assert probe.stronger is predecessor.stronger
    assert probe.ARITHMETIC == predecessor.ARITHMETIC
    assert probe.RESERVES == predecessor.RESERVES and probe.WRITE_BUDGETS == predecessor.WRITE_BUDGETS


def test_reference_identity_and_every_loss_are_bound_at_registration_and_runtime():
    identities = probe.reference_identities()
    assert set(identities) == {"v8", "general"}
    assert identities["v8"] == predecessor.reference_identities()["v8"]
    assert identities["general"]["checkpoint"] == str(probe.GENERAL)
    assert identities["general"]["sha256"] == probe.cohort_retention.REFERENCE_SHA256S["general"]
    assert 'Path(reference_identities()["general"]["checkpoint"])' in probe.REGISTER_SOURCE
    assert '"mean_distillation": stronger.SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'config["mean_distillation"] != stronger.SPECIFICATION' in probe.ADAPTED_SOURCE
    assert 'with stronger.training_objective(), teacher_policy.runtime(' in probe.ADAPTED_SOURCE
    assert 'Counter(record["reference_role"]' in probe.PROBE_SOURCE
    assert '{"general": 2, "v8": 24}' in probe.PROBE_SOURCE
    assert probe.replay_cohort_targets.__code__ is predecessor.replay_cohort_targets.__code__
    assert probe.replay_cohort_targets.__globals__ is vars(probe)


def test_all_actual_adapted_functions_have_resolved_globals():
    for value in vars(probe).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(probe):
            assert unresolved_globals(value) == []


def test_original_probe_source_and_callbacks_are_unmodified():
    assert predecessor.retention.__name__ == "single_model_uniform_v8_retention"
    assert predecessor.reference_identities().keys() == {"v8", "v29"}
    assert predecessor.DATA_SEED == 1171100901
    assert probe.retention is probe.cohort_retention
    assert probe.stronger.original.balanced_loss is probe.stronger.BASE_LOSS


def complete_diagnosis():
    failures = [("V8", "mar", "common_nrmse")]
    failures.extend(("general", "imputation", metric) for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse"))
    rows = [{"reference": reference, "group": group, "metric": metric,
             "initializer_failed": True, "candidate_failed": True} for reference, group, metric in failures]
    rows.extend({"reference": "V8", "group": f"passing_{index}", "metric": "metric",
                 "initializer_failed": False, "candidate_failed": False} for index in range(28))
    return {"complete": True, "prediction_grids": 284, "comparisons": rows, "post_hoc_consumed_bank": True,
            "cuda_initialized": False, "training_performed": False, "confirmation_repeated": False,
            "checkpoint_selection_performed": False, "real_benchmarks_evaluated": False, "artifact_sha256": {}}


@pytest.mark.parametrize("mutation", [None, "partial_grids", "partial_metrics", "changed_failure", "live", "repeated", "fresh_label"])
def test_complete_negative_attribution_is_required_before_new_seed_admission(monkeypatch, mutation):
    report = complete_diagnosis()
    if mutation == "partial_grids":
        report["prediction_grids"] = 283
    elif mutation == "partial_metrics":
        report["comparisons"].pop()
    elif mutation == "changed_failure":
        report["comparisons"][0]["initializer_failed"] = False
    elif mutation == "repeated":
        report["confirmation_repeated"] = True
    elif mutation == "fresh_label":
        report["post_hoc_consumed_bank"] = False
    monkeypatch.setattr(probe.diagnosis, "verify", lambda: None)
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "digest")
    monkeypatch.setattr(probe, "read", lambda path: {"state": "complete", "pid": 101}
                        if path.name == "status.json" else deepcopy(report))
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: {"pid": pid} if mutation == "live" else None)
    if mutation is None:
        evidence = probe.require_completed_v46()
        assert evidence[str(probe.diagnosis.OUTPUT / "audit.json")] == probe.DIAGNOSTIC_SHA256
    else:
        with pytest.raises(ValueError):
            probe.require_completed_v46()
