from contextlib import nullcontext
from copy import deepcopy
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import preflight_single_model_cohort_anchor_retention as probe
import test_single_model_all_imputation_retention_preflight as previous_tests
import test_single_model_anchor_retention_preflight as original_tests
from single_model_distillation import Episode
from single_model_registered_adaptation import replace_once


def completed_reports():
    reports = original_tests.latest()
    training = reports[1]
    training.update(accepted_updates=56, fixed_half_training=True, displacement_fraction=.5)
    training["best"]["step"] = 50
    replay = {"complete": True, "endpoint_grids": 56, "selected_grids": 152}
    gate = {"selected_step": 50, "endpoint_reproduced": True, "selected_validation_passed": True}
    failures = [{"baseline": "V8_retention_baseline", "group": "imputation", "metric": metric}
        for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse", "rmse")]
    failures.append({"baseline": "V8_retention_baseline", "group": "mnar_two_phase_subset", "metric": "common_nrmse"})
    closure = {"complete": True, "passed_original_confirmation": False, "prediction_grids": 1136,
        "cases": 284, "selected_step": 50, "comparisons": [{} for index in range(32)], "failures": failures}
    diagnostic = {"complete": True, "prediction_grids": 284, "comparisons": [{} for index in range(22)],
        "cuda_initialized": False, "post_hoc_consumed_bank": True, "checkpoint_selection_performed": False,
        "confirmation_repeated": False, "training_performed": False, "real_benchmarks_evaluated": False}
    return training, replay, gate, closure, diagnostic


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("complete_training_audit",), False), (0, ("through_step",), 63),
    (0, ("accepted_updates",), 57), (0, ("prediction_grids",), 1216), (0, ("full_network_training",), False),
    (0, ("fixed_half_training",), False), (0, ("displacement_fraction",), 1.), (0, ("best", "step"), 64),
    (0, ("independently_regenerated_training_tasks",), 1764), (0, ("independently_regenerated_rating_contexts",), 126),
    (0, ("independently_regenerated_structured_draws",), 1535), (0, ("confirmation_evaluated",), True),
    (0, ("admission_attempts", "data_rejected_steps"), [40]), (0, ("admission_attempts", "optimizer_attempts"), 63),
    (0, ("raw_endpoint", "adam", "gradient_vectors"), 28), (0, ("raw_endpoint", "optimizer_attempt_step"), 63),
    (1, ("complete",), False), (1, ("endpoint_grids",), 55), (1, ("selected_grids",), 0),
    (2, ("selected_step",), 64), (2, ("endpoint_reproduced",), False), (2, ("selected_validation_passed",), False),
    (3, ("complete",), False), (3, ("passed_original_confirmation",), True), (3, ("prediction_grids",), 1135),
    (3, ("cases",), 283), (3, ("selected_step",), 64), (3, ("comparisons",), []), (3, ("failures",), []),
    (4, ("complete",), False), (4, ("prediction_grids",), 283), (4, ("comparisons",), []),
    (4, ("cuda_initialized",), True), (4, ("post_hoc_consumed_bank",), False),
    (4, ("checkpoint_selection_performed",), True), (4, ("confirmation_repeated",), True),
    (4, ("training_performed",), True), (4, ("real_benchmarks_evaluated",), True),
])
def test_requires_all_closed_negative_predecessor_evidence(entry, path, value):
    reports = completed_reports()
    probe.validate_completed(*reports)
    target = reports[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)


def test_predecessor_cannot_replace_one_failed_cohort():
    reports = completed_reports()
    reports[3]["failures"][-1]["group"] = "mar"
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)


@pytest.mark.parametrize("running", [False, True])
def test_original_auditor_must_be_stopped_even_without_a_status_file(tmp_path, running):
    directory = tmp_path / "123"
    directory.mkdir()
    script = b"/repo/audit_single_model_legacy_blend_teacher.py" if running else b"unrelated.py"
    (directory / "cmdline").write_bytes(b"python\0" + script + b"\0")
    if running:
        with pytest.raises(ValueError):
            probe.require_auditor_stopped(tmp_path)
    else:
        probe.require_auditor_stopped(tmp_path)


def test_only_reference_assignment_changes_the_scientific_recipe():
    previous = probe.previous_probe
    assert probe.PARENT == previous.PARENT and probe.PARENT_SHA256 == previous.PARENT_SHA256
    assert probe.retention is probe.cohort_retention and probe.retention is not previous.retention
    assert probe.teacher_policy is previous.teacher_policy and probe.paired is previous.paired
    assert probe.retention.BACKWARD_SOURCE_SHA256 == previous.retention.BACKWARD_SOURCE_SHA256
    assert probe.RATE == previous.RATE == 5e-6
    assert probe.RESERVES == previous.RESERVES and probe.WRITE_BUDGETS == previous.WRITE_BUDGETS
    assert probe.STORAGE_ACCOUNTING == previous.STORAGE_ACCOUNTING
    assert probe.DATA_SEED == 1051100901 and probe.GRADIENT_SEED == 1052100901
    assert probe.OUTPUT.name == "cohort_anchor_retention_preflight_v1"
    assert probe.ARITHMETIC == {"blas_threads": 24, "torch_threads": 4}
    assert probe.TRAINABLE_TENSORS == 238 and probe.TRAINABLE_PARAMETERS == 77998496 and probe.FROZEN_TENSORS == 16
    current, original = probe.batch(), previous.batch()
    seeds = probe.training.alignment.fixed.seeds_in(current)
    assert len(seeds) == 56 and not seeds.intersection(probe.training.alignment.fixed.seeds_in(original))
    for current_task, previous_task in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = previous_task["gradient_seed"]
        restored["specification"].get("structured", restored["specification"])["seed"] = (
            previous_task["specification"].get("structured", previous_task["specification"])["seed"])
        assert restored == previous_task
    for name in ("run", "verify", "audit", "audit_original", "register_trial", "replay_admitted", "audit_teacher_replay"):
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe) and function.__closure__ is None
    assert "require_completed_v41" in probe.require_predecessor.__code__.co_names
    assert 'references = retention.make_references(model, identities)' in probe.ADAPTED_SOURCE
    assert 'output / "retention_targets", references, identities,' in probe.ADAPTED_SOURCE
    assert 'expected = payload["snapshot"]["contracted"]' in probe.ADAPTED_SOURCE
    for name in ("retention_references", "arithmetic", "teacher_replay_source_sha256", "teacher_replay_audit_source_sha256"):
        assert name in probe.REGISTER_SOURCE and name in probe.ADAPTED_SOURCE
    assert probe.TEACHER_REPLAY_SOURCE == previous.TEACHER_REPLAY_SOURCE
    assert probe.TEACHER_REPLAY_AUDIT_SOURCE == previous.TEACHER_REPLAY_AUDIT_SOURCE
    assert "original_arithmetic" in probe.main.__code__.co_names
    assert not torch.cuda.is_initialized()


FIXTURE_SOURCE = previous_tests.FIXTURE_SOURCE
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE,
    '    identity = {"checkpoint": str(probe.PARENT), "sha256": probe.PARENT_SHA256}',
    '    identities = probe.reference_identities()')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '        truth = torch.tensor(',
    '        if index % 14 not in (0, 13):\n'
    '            metadata.update(structured=True, component="low_rank_mcar", family=probe.retention.FAMILIES[index % 14 - 1])\n'
    '        reference_role = probe.retention.reference_role(metadata)\n'
    '        truth = torch.tensor(')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '"eligible": probe.retention.eligible(metadata),',
    '"eligible": probe.retention.eligible(metadata), "reference_role": reference_role,')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '"identity": identity,',
    '"identity": identities[reference_role], "reference_role": reference_role,')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, '"artifact": str(replay_path), "sha256": probe.sha256(replay_path)',
    '"artifact": str(replay_path), "sha256": probe.sha256(replay_path), "reference_role": reference_role, "identity": identities[reference_role]')
FIXTURE_SOURCE = replace_once(FIXTURE_SOURCE, 'reference = {**identity, "state_tensors": 254, "bitwise_equal": True}',
    'reference = {role: {**identity, "state_tensors": 254, "bitwise_equal": True} for role, identity in identities.items()}')
exec(compile(FIXTURE_SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("mutation", [None, "omit", "hide_spectral", "duplicate", "source", "mask", "identity", "nonfinite", "shape", "task"])
def test_every_original_task_source_and_full_reference_remains_bound(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "probe", probe)
    monkeypatch.setattr(previous_tests, "retention_fixture", retention_fixture)
    previous_tests.test_all26_imputation_references_are_mandatory_and_hash_bound(tmp_path, mutation)


@pytest.mark.parametrize("mutation", [None, "same_process", "different_prediction", "missing_spectral_reference"])
def test_independent_audit_retains_all_queries_and_unchanged_penalty(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(previous_tests, "probe", probe)
    monkeypatch.setattr(previous_tests, "retention_fixture", retention_fixture)
    previous_tests.test_independent_audit_covers_every_spectral_grid_and_original_penalty(tmp_path, monkeypatch, mutation)
    if mutation is None:
        report = probe.read(tmp_path / "retention_audit.json")
        assert report["reference_role_counts"] == {"v8": 4, "v29": 22}
        assert set(report["reference"]) == {"v8", "v29"}


@pytest.mark.parametrize("mutation", ["record_role", "payload_role", "swap_identity", "missing_identity", "rating_reference"])
def test_cannot_swap_or_omit_cohort_references(tmp_path, mutation):
    retention_fixture(tmp_path)
    path = tmp_path / "retention_targets/manifest.json"
    manifest = probe.read(path)
    record = manifest["records"][0]
    if mutation == "record_role":
        record["reference_role"] = "v29"
    elif mutation in {"payload_role", "swap_identity"}:
        payload = torch.load(record["artifact"], weights_only=False)
        if mutation == "payload_role":
            payload["reference_role"] = "v29"
        else:
            payload["identity"] = probe.reference_identities()["v29"]
        torch.save(payload, record["artifact"])
        record["sha256"] = probe.sha256(record["artifact"])
    elif mutation == "missing_identity":
        manifest["reference"].pop("v29")
    else:
        manifest["records"][13]["reference_role"] = "v8"
    probe.write_json(path, manifest)
    with pytest.raises(ValueError):
        probe.retention_records(tmp_path)


@pytest.mark.parametrize("mutation", ["role", "identity"])
def test_independent_replay_must_bind_which_reference_produced_each_grid(tmp_path, monkeypatch, mutation):
    retention_fixture(tmp_path)
    monkeypatch.setattr(probe, "audit_original", lambda output: None)
    monkeypatch.setattr(probe, "verify", lambda output: probe.read(output / "config.json"))
    path = tmp_path / "retention_replay/manifest.json"
    manifest = probe.read(path)
    if mutation == "role":
        manifest["records"][0]["reference_role"] = "v29"
    else:
        manifest["records"][0]["identity"] = probe.reference_identities()["v29"]
    probe.write_json(path, manifest)
    with pytest.raises(ValueError):
        probe.audit(tmp_path)
    assert not (tmp_path / "retention_audit.json").exists()


def test_rejected_admission_does_not_allocate_or_replay_references(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_rejected_admission_does_no_retention_or_model_work(tmp_path, monkeypatch)


@pytest.mark.parametrize("failure", [None, "prediction", "reference_mutation"])
def test_independent_replay_runs_both_references_and_verifies_them_again(tmp_path, monkeypatch, failure):
    calls = []
    identities = probe.reference_identities()
    references = {role: torch.nn.Linear(2, 1).eval().requires_grad_(False) for role in identities}
    verified = {role: {**identity, "state_tensors": 254, "bitwise_equal": True} for role, identity in identities.items()}
    records = []
    for index, role in enumerate(("v8", "v29")):
        metadata = {"seed": 1000 + index, "domain": "imputation", "gradient_data": "synthetic",
            "component": 0, "prior": "low_rank_mcar"}
        if role == "v29":
            metadata.update(structured=True, component="low_rank_mcar", family="mar")
        target = torch.full((1, 3, 2), float(index + 1))
        mask = torch.tensor([[[True, False], [True, False], [False, True]]])
        episode = Episode(target.masked_fill(mask, float("nan")), target, mask, metadata)
        records.append(({}, {"seed": metadata["seed"], "reference_role": role}, episode, {"prediction": target}))

    def verify_model(model, identity):
        role = next(role for role, reference in references.items() if model is reference)
        assert identity == identities[role]
        calls.append(("verify", role))
        if failure == "reference_mutation" and calls.count(("verify", role)) == 2:
            raise ValueError("Changed frozen reference")
        return verified[role]

    def predict(model, source, criterion, **options):
        role = next(role for role, reference in references.items() if model is reference)
        calls.append(("predict", role))
        assert criterion == "criterion" and options == probe.STREAM_OPTIONS
        return torch.full_like(source, (1. if role == "v8" else 2.) + float(failure == "prediction"))

    monkeypatch.setattr(probe, "verify", lambda output: {"base": {}})
    monkeypatch.setattr(probe, "replay_original_teachers", lambda output: calls.append(("original_teachers", None)))
    monkeypatch.setattr(probe, "retention_records", lambda output: ({"reference": verified}, records))
    monkeypatch.setattr(probe.training.alignment.mixture, "original_setup", lambda config: (object(), None, "criterion", None, None))
    monkeypatch.setattr(probe.retention, "make_references", lambda student, declared: references if declared == identities else {})
    monkeypatch.setattr(probe.retention, "verify_model", verify_model)
    monkeypatch.setattr(probe, "teacher_predictions", predict)
    monkeypatch.setattr(probe, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    (tmp_path / "retention_targets").mkdir()
    probe.write_json(tmp_path / "retention_targets/manifest.json", {"complete": True})
    probe.write_json(tmp_path / "result.json", {})
    if failure:
        with pytest.raises(ValueError):
            probe.replay_targets(tmp_path)
        assert not (tmp_path / "retention_replay/manifest.json").exists()
    else:
        probe.replay_targets(tmp_path)
        report = probe.read(tmp_path / "retention_replay/manifest.json")
        assert report["complete"] and report["reference"] == verified
        assert [record["reference_role"] for record in report["records"]] == ["v8", "v29"]
        assert calls == [("original_teachers", None), ("verify", "v8"), ("verify", "v29"),
            ("predict", "v8"), ("predict", "v29"), ("verify", "v8"), ("verify", "v29")]
    assert not torch.cuda.is_initialized()
