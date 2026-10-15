import ast
from copy import deepcopy
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import preflight_single_model_all_imputation_retention as probe
import test_single_model_anchor_retention_preflight as original_tests
from single_model_distillation import Episode


def latest():
    records = original_tests.latest()
    records[1].update(accepted_updates=59, fixed_half_training=True, displacement_fraction=.5)
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("campaign",), "campaign"), (0, ("review",), "review"),
    (0, ("no_inference_ensemble",), False), (1, ("accepted_updates",), 60), (1, ("attempted_batches",), 63),
    (1, ("through_step",), 63), (1, ("full_network_training",), False), (1, ("complete_training_audit",), False),
    (1, ("fixed_half_training",), False), (1, ("displacement_fraction",), 1.),
    (1, ("prediction_grids",), 1216), (1, ("best", "step"), 10),
    (1, ("independently_regenerated_training_tasks",), 1791), (1, ("independently_regenerated_rating_contexts",), 127),
    (1, ("independently_regenerated_structured_draws",), 1535), (1, ("confirmation_evaluated",), True),
    (1, ("admission_attempts", "data_rejected_steps"), [51]), (1, ("admission_attempts", "optimizer_attempts"), 63),
    (1, ("raw_endpoint", "adam", "gradient_vectors"), 28), (1, ("raw_endpoint", "optimizer_attempt_step"), 63),
    (2, ("complete",), False), (2, ("endpoint_grids",), 55), (2, ("selected_grids",), 152),
    (3, ("endpoint_reproduced",), False), (3, ("selected_validation_passed",), True),
    (4, ("passed",), True), (4, ("confirmation_evaluated",), True), (4, ("selected_step",), 10),
    (4, ("real_benchmark_success",), True)])
def test_requires_the_complete_negative_fixed_half_predecessor(entry, path, value):
    records = latest()
    probe.validate_latest(*records)
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_latest(*records)


def test_scope_only_probe_keeps_all_teachers_losses_optimizer_and_shapes():
    previous = probe.previous_probe
    assert probe.latest is probe.completed and probe.latest.training.OUTPUT.name == "fixed_half_shrunk_retention_v39"
    assert probe.PARENT == previous.PARENT == probe.latest.training.PARENT
    assert probe.PARENT_SHA256 == probe.retention.REFERENCE_SHA256
    assert probe.RESERVES == previous.RESERVES
    assert probe.WRITE_BUDGETS == {**previous.WRITE_BUDGETS, "checkpoint": 512 * 1024**2}
    assert probe.STORAGE_ACCOUNTING == {**previous.STORAGE_ACCOUNTING,
        "checkpoint_artifact_allowance_bytes": previous.STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] + 128 * 1024**2}
    assert probe.STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] < probe.WRITE_BUDGETS["checkpoint"]
    assert sum(value for key, value in probe.STORAGE_ACCOUNTING.items()
        if key != "checkpoint_artifact_allowance_bytes") < probe.WRITE_BUDGETS["scratch"]
    assert probe.RATE == previous.RATE == 5e-6 and probe.paired is previous.paired
    assert probe.teacher_policy is previous.teacher_policy
    assert probe.DATA_SEED == 991100901 and probe.GRADIENT_SEED == 992100901
    assert probe.OUTPUT.name == "all_imputation_retention_preflight_v1"
    current, original = probe.batch(), previous.batch()
    seeds = probe.training.alignment.fixed.seeds_in(current)
    assert len(seeds) == 56 and not seeds.intersection(probe.training.alignment.fixed.seeds_in(original))
    assert sum(task["specification"]["domain"] == "imputation" for task in current["tasks"]) == 26
    for current_task, original_task in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = original_task["gradient_seed"]
        specification = restored["specification"].get("structured", restored["specification"])
        original_specification = original_task["specification"].get("structured", original_task["specification"])
        specification["seed"] = original_specification["seed"]
        assert restored == original_task


def test_generated_helpers_resolve_all_imputation_loss_and_fixed_half_auditor(monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_cloned_helpers_resolve_new_trial_globals_not_old_owners()
    original_tests.test_generated_run_reaches_real_retention_context_and_does_not_modify_inference()
    for name in ("run", "audit", "audit_original", "audit_admitted", "register_trial"):
        assert inspect.unwrap(getattr(probe, name)).__globals__["paired"] is probe.previous_probe.paired
    assert "require_exact_predecessor" in probe.require_predecessor.__code__.co_names
    assert "require_displacement_panel" not in probe.require_predecessor.__code__.co_names
    assert 'expected = payload["snapshot"]["contracted"]' in probe.ADAPTED_SOURCE
    assert '"displacement_fraction": paired.FRACTION' in probe.REGISTER_SOURCE
    assert 'config["displacement_fraction"] != paired.FRACTION' in probe.ADAPTED_SOURCE
    assert 'tests/test_single_model_all_imputation_retention.py' in probe.REGISTER_SOURCE
    assert 'tests/test_single_model_all_imputation_retention_preflight.py' in probe.REGISTER_SOURCE
    nodes = {node.name: node for node in ast.parse(probe.SOURCE).body if isinstance(node, ast.FunctionDef)}
    assert "retention.independent_penalty" in ast.get_source_segment(probe.SOURCE, nodes["audit"])


def test_rejected_admission_does_no_retention_or_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_rejected_admission_does_no_retention_or_model_work(tmp_path, monkeypatch)


def test_raw_endpoint_preserves_every_full_network_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_raw_endpoint_retains_full_coordinate_alias_and_optimizer_evidence(tmp_path, monkeypatch)


def test_exact_predecessor_binds_all_fixed_half_and_retention_reports(monkeypatch):
    captured = []
    monkeypatch.setattr(probe.previous_probe, "verify_hashes", lambda expected: captured.append(expected))
    expected = probe.require_exact_predecessor()
    assert captured == [expected] and len(expected) == 5
    assert expected[str(probe.completed.OUTPUT / "result.json")] == (
        "2db49b6c9031d1b3987414433b3673d63ea694d003c1488a50c3b752c9fd4050")


FIXTURE_SOURCE = inspect.getsource(original_tests.retention_fixture)
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('"domain": "imputation", "gradient_data":',
    '"domain": "recsys" if index in (13, 27) else "imputation", "gradient_data":')
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('"domain": "imputation", "component": metadata["component"]',
    '"domain": metadata["domain"], "component": metadata["component"]')
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('"eligible": native', '"eligible": probe.retention.eligible(metadata)')
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('        if native:', '        if probe.retention.eligible(metadata):')
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('[4., 5.]]])', '[4., 3.5]]])')
FIXTURE_SOURCE = FIXTURE_SOURCE.replace('        episode = Episode(source, truth, mask, metadata)',
    '        scaling = (SimpleNamespace(means=torch.full((truth.shape[2],), 2.), stds=torch.full((truth.shape[2],), .5),\n'
    '            row_offsets=torch.zeros(truth.shape[1]), row_scales=torch.ones(truth.shape[1]))\n'
    '            if metadata["domain"] == "recsys" else None)\n'
    '        episode = Episode(source, truth, mask, metadata, scaling)')
exec(compile(FIXTURE_SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("mutation", [None, "omit", "hide_spectral", "duplicate", "source", "mask",
    "identity", "nonfinite", "shape", "task"])
def test_all26_imputation_references_are_mandatory_and_hash_bound(tmp_path, mutation):
    decisions = retention_fixture(tmp_path)
    path = tmp_path / "retention_targets/manifest.json"
    manifest = probe.read(path)
    record = manifest["records"][1]
    payload_path = probe.Path(record["artifact"])
    payload = torch.load(payload_path, weights_only=False)
    if mutation == "omit":
        manifest["records"].pop()
    elif mutation == "hide_spectral":
        record["eligible"] = False
    elif mutation == "duplicate":
        manifest["records"][2] = deepcopy(record)
    elif mutation == "source":
        payload["source"][~payload["mask"]] += .1
    elif mutation == "mask":
        payload["mask"][0, 0, 0] = False
    elif mutation == "identity":
        payload["identity"]["sha256"] = "wrong"
    elif mutation == "nonfinite":
        payload["prediction"][0, 0, 0] = float("nan")
    elif mutation == "shape":
        payload["prediction"] = payload["prediction"][:, :1]
    elif mutation == "task":
        task_path = tmp_path / "training_batch/task_01.pt"
        task = torch.load(task_path, weights_only=False)
        task["task"]["weight"] *= 2
        torch.save(task, task_path)
    if mutation in {"source", "mask", "identity", "nonfinite", "shape"}:
        torch.save(payload, payload_path)
        record["sha256"] = probe.sha256(payload_path)
    probe.write_json(path, manifest)
    if mutation:
        with pytest.raises((ValueError, RuntimeError)):
            probe.retention_records(tmp_path)
    else:
        _, records = probe.retention_records(tmp_path)
        assert len(records) == 26
        assert all(decision["eligible"] is (index not in (13, 27)) for index, decision in enumerate(decisions))


@pytest.mark.parametrize("mutation", [None, "same_process", "different_prediction", "missing_spectral_reference"])
def test_independent_audit_covers_every_spectral_grid_and_original_penalty(tmp_path, monkeypatch, mutation):
    decisions = retention_fixture(tmp_path)
    monkeypatch.setattr(probe, "audit_original", lambda output: None)
    monkeypatch.setattr(probe, "verify", lambda output: probe.read(output / "config.json"))
    path = tmp_path / "retention_replay/manifest.json"
    replay = probe.read(path)
    if mutation == "same_process":
        replay["pid"] = os.getpid()
    elif mutation == "missing_spectral_reference":
        replay["records"].pop(1)
    elif mutation == "different_prediction":
        prediction_path = probe.Path(replay["records"][1]["artifact"])
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = archive["prediction"].copy()
        prediction[0, 0, 0] += .1
        probe.atomic_npz(prediction_path, prediction=prediction)
        replay["records"][1]["sha256"] = probe.sha256(prediction_path)
    probe.write_json(path, replay)
    if mutation:
        with pytest.raises(ValueError):
            probe.audit(tmp_path)
        assert not (tmp_path / "retention_audit.json").exists()
        return
    probe.audit(tmp_path)
    report = probe.read(tmp_path / "retention_audit.json")
    assert report["complete"] and report["eligibility_decisions"] == 28 and report["reference_grids"] == 26
    assert report["no_inference_ensemble"] and report["full_autograd_replayed"] is False
    for row in report["endpoint_penalties"]:
        index = int(row["task"].split("_")[-1])
        assert decisions[index]["eligible"] and row["queries"] == 4 and row["before"] == 0.
        assert row["candidate"] == pytest.approx(10. * .25**2 * (1. + 1. / .375) / 2., rel=1e-6)
    probe.verify_hashes(report["artifact_sha256"])
