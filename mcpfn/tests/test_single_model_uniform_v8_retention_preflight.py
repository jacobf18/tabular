from copy import deepcopy
import inspect
import math
import os
import types
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_uniform_v8_retention as probe
import preflight_single_model_family_balanced_retention as previous
import test_single_model_family_balanced_retention_preflight as previous_tests
from single_model_distillation import Episode


def test_probe_preserves_every_original_task_and_supervised_weight_with_fresh_seeds():
    current, original = probe.batch(), previous.batch()
    assert len(current["tasks"]) == len(original["tasks"]) == 28
    for actual, expected in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(actual)
        assert restored["gradient_seed"] != expected["gradient_seed"]
        restored["gradient_seed"] = expected["gradient_seed"]
        specification = restored["specification"]
        specification.get("structured", specification)["seed"] = expected["specification"].get(
            "structured", expected["specification"])["seed"]
        assert restored == expected
    assert probe.reference_identities() == previous.reference_identities()
    assert probe.ARITHMETIC == {"blas_threads": 24, "torch_threads": 4}
    assert probe.TEACHER_REPLAY_SOURCE == previous.TEACHER_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_AUDIT_SOURCE == previous.TEACHER_REPLAY_AUDIT_SOURCE
    for name in ("run", "verify", "audit", "register_trial", "replay_cohort_targets"):
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe) and function.__closure__ is None
    assert "require_completed_v43" in probe.require_predecessor.__code__.co_names
    assert probe.diagnosis.__name__ == "diagnose_single_model_family_balanced_initializer"


FIXTURE_SOURCE = previous_tests.FIXTURE_SOURCE
exec(compile(FIXTURE_SOURCE, __file__, "exec"), globals())

function = previous_tests.test_all_reference_and_weighting_decisions_are_bound
test_all_reference_and_weighting_decisions_are_bound = types.FunctionType(
    function.__code__, globals(), function.__name__, function.__defaults__)
test_all_reference_and_weighting_decisions_are_bound.__dict__.update(function.__dict__)


def test_substituting_v29_on_any_original_imputation_task_is_rejected(tmp_path):
    retention_fixture(tmp_path)
    path = tmp_path / "retention_targets/manifest.json"
    manifest = probe.read(path)
    manifest["records"][1]["reference_role"] = "v29"
    probe.write_json(path, manifest)
    with pytest.raises(ValueError):
        probe.retention_records(tmp_path)


def test_independent_cpu_audit_preserves_all_26_v8_references_and_both_rating_exclusions(tmp_path, monkeypatch):
    retention_fixture(tmp_path)
    monkeypatch.setattr(probe, "audit_original", lambda output: None)
    monkeypatch.setattr(probe, "verify", lambda output: probe.read(output / "config.json"))
    monkeypatch.setattr(probe, "audit_teacher_replay", lambda output: None)
    probe.audit(tmp_path)
    report = probe.read(tmp_path / "retention_audit.json")
    assert report["complete"] and report["reference_grids"] == 26
    assert report["reference_role_counts"] == {"v8": 26}
    tasks = probe.read(tmp_path / "training_batch/manifest.json")["tasks"]
    assert len(tasks) == 28
    for row in report["endpoint_penalties"]:
        assert row["reference_role"] == "v8"
        assert row["before"] == row["before_mixture_weighted"] == 0.
        assert row["candidate"] > 0.
        assert row["candidate_mixture_weighted"] == pytest.approx(row["candidate"] * row["weighting"]["task_weight"])


def completed_reports():
    training = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 56, "training_episodes": 1792, "prediction_grids": 1368,
        "full_network_training": True, "fixed_half_training": True, "displacement_fraction": .5,
        "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
        "independently_regenerated_structured_draws": 1536, "full_optimizer_trajectory_replayed": False,
        "autograd_replayed": False, "best": {"step": 25}, "raw_endpoint": {"adam": {"gradient_vectors": 84}}}
    replay = {"complete": True, "endpoint_grids": 56, "selected_grids": 152}
    gate = {"passed": False, "selected_step": 25, "cases": 284, "confirmation_evaluated": True}
    audit = {"complete": True, "prediction_grids": 1136, "cuda_initialized": False}
    closure = {"complete": True, "gate": deepcopy(gate), "campaign": None, "review": None}
    diagnostic = {"complete": True, "prediction_grids": 284, "comparisons": [{}] * 22,
        "post_hoc_consumed_bank": True, "cuda_initialized": False, "training_performed": False,
        "confirmation_repeated": False, "checkpoint_selection_performed": False, "real_benchmarks_evaluated": False}
    return training, replay, gate, audit, closure, diagnostic


@pytest.mark.parametrize("entry,key,value", [(0,"accepted_updates",55), (0,"prediction_grids",1216),
    (0,"autograd_replayed",True), (1,"selected_grids",0), (2,"passed",True), (2,"cases",283),
    (3,"prediction_grids",1135), (3,"cuda_initialized",True), (4,"campaign",{}),
    (5,"comparisons",[]), (5,"post_hoc_consumed_bank",False), (5,"training_performed",True)])
def test_incomplete_or_reinterpreted_predecessor_evidence_is_rejected(entry, key, value):
    reports = completed_reports()
    probe.validate_completed(*reports)
    reports[entry][key] = value
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)
