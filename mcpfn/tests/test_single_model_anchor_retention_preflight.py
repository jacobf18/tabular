import ast
from copy import deepcopy
import inspect
import json
import os

import numpy as np
import pytest
import torch

import preflight_single_model_anchor_retention as probe
import test_single_model_legacy_point_preflight as inherited
from single_model_distillation import Episode


def latest():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0, "real_benchmark_success": False}
    report = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
        "accepted_updates": 61, "best": {"step": 0}, "prediction_grids": 1368, "full_network_training": True,
        "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
        "independently_regenerated_structured_draws": 1536, "confirmation_evaluated": False,
        "admission_attempts": {"data_rejected_steps": [], "optimizer_attempts": 64},
        "raw_endpoint": {"adam": {"gradient_vectors": 84}, "optimizer_attempt_step": 64}}
    result = {"complete": True, "gate": gate, "campaign": None, "review": None, "no_inference_ensemble": True}
    return (result, report, {"complete": True, "endpoint_grids": 56, "selected_grids": 0},
            {"endpoint_reproduced": True, "selected_validation_passed": False, "selected_step": 0}, gate)


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("campaign",), "campaign"), (0, ("review",), "review"),
    (0, ("no_inference_ensemble",), False), (1, ("accepted_updates",), 64), (1, ("attempted_batches",), 63),
    (1, ("through_step",), 63), (1, ("full_network_training",), False), (1, ("complete_training_audit",), False),
    (1, ("prediction_grids",), 1216), (1, ("best", "step"), 10),
    (1, ("independently_regenerated_training_tasks",), 1791), (1, ("independently_regenerated_rating_contexts",), 127),
    (1, ("independently_regenerated_structured_draws",), 1535), (1, ("confirmation_evaluated",), True),
    (1, ("admission_attempts", "data_rejected_steps"), [51]), (1, ("admission_attempts", "optimizer_attempts"), 63),
    (1, ("raw_endpoint", "adam", "gradient_vectors"), 28), (1, ("raw_endpoint", "optimizer_attempt_step"), 63),
    (2, ("complete",), False), (2, ("endpoint_grids",), 55), (2, ("selected_grids",), 152),
    (3, ("endpoint_reproduced",), False), (3, ("selected_validation_passed",), True),
    (4, ("passed",), True), (4, ("confirmation_evaluated",), True), (4, ("selected_step",), 10),
    (4, ("real_benchmark_success",), True)])
def test_predecessor_requires_all_original_v36_negative_evidence(entry, path, value):
    records = latest()
    probe.validate_latest(*records)
    target = records[entry]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_latest(*records)


def test_fresh_preflight_preserves_scientific_base_except_registered_additions():
    assert probe.PARENT == probe.template.PARENT == probe.latest.training.PARENT
    assert probe.PARENT_SHA256 == probe.retention.REFERENCE_SHA256
    assert probe.RESERVES == probe.template.RESERVES and probe.WRITE_BUDGETS == probe.template.WRITE_BUDGETS
    assert probe.RATE == 5e-6 and probe.paired is probe.template.paired
    assert probe.teacher_policy is probe.template.teacher_policy
    assert probe.retention.SPECIFICATION["weight"] == 10.
    assert probe.DATA_SEED == 871100901 and probe.GRADIENT_SEED == 872100901
    current, previous = probe.batch(), probe.template.batch()
    current_seeds = probe.training.alignment.fixed.seeds_in(current)
    assert len(current_seeds) == 56 and not current_seeds.intersection(probe.training.alignment.fixed.seeds_in(previous))
    for current_task, old_task in zip(current["tasks"], previous["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = old_task["gradient_seed"]
        current_specification = restored["specification"].get("structured", restored["specification"])
        old_specification = old_task["specification"].get("structured", old_task["specification"])
        current_specification["seed"] = old_specification["seed"]
        assert restored == old_task


def test_generated_run_reaches_real_retention_context_and_does_not_modify_inference():
    nodes = {node.name: node for node in ast.parse(probe.ADAPTED_SOURCE).body if isinstance(node, ast.FunctionDef)}
    run = ast.get_source_segment(probe.ADAPTED_SOURCE, nodes["run"])
    assert "anchor = deepcopy(model).eval().requires_grad_(False)" in run
    assert "with retention.runtime(training.alignment.mixture.backend" in run
    assert "full_network.capture_coordinates(model, view):" in run
    assert "replicated.capture_batch(" in run
    assert run.index("model.load_state_dict") < run.index("anchor = deepcopy") < run.index("with retention.runtime")
    assert "retention" not in ast.get_source_segment(probe.ADAPTED_SOURCE, nodes["predictions"])
    assert "cap90_recipe.validate_base" in ast.get_source_segment(probe.ADAPTED_SOURCE, nodes["verify"])
    assert probe.STREAM_OPTIONS == probe.retention.STREAM_OPTIONS
    assert not torch.cuda.is_initialized()


def test_cloned_helpers_resolve_new_trial_globals_not_old_owners():
    names = ("run", "verify", "register_trial", "predictions", "audit_admitted", "replay_admitted",
        "replay_original_teachers", "audit_original", "admit_batch", "audit_admission", "check_routing",
        "audit_teacher_replay", "frozen_inventory", "require_volume_space", "save_raw", "audit_raw")
    for name in names:
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe), name
        assert function.__closure__ is None, name
    assert probe.template.run.__globals__ is vars(probe.template)
    assert probe.OUTPUT != probe.template.OUTPUT and probe.SCRATCH != probe.template.SCRATCH


def test_rejected_admission_does_no_retention_or_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", probe)
    inherited.test_rejected_admission_still_performs_no_model_or_optimizer_work(tmp_path, monkeypatch)


def test_raw_endpoint_retains_full_coordinate_alias_and_optimizer_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", probe)
    inherited.test_actual_proposal_endpoint_uses_full_alias_through_preflight_storage(tmp_path, monkeypatch)


def retention_fixture(output):
    (output / "training_batch").mkdir()
    (output / "retention_targets").mkdir()
    (output / "retention_replay").mkdir()
    identity = {"checkpoint": str(probe.PARENT), "sha256": probe.PARENT_SHA256}
    specifications, tasks, decisions, replays = [], [], [], []
    endpoints = {role: [] for role in ("before", "candidate")}
    for index in range(28):
        seed, native = 1000 + index, index in (0, 14)
        metadata = {"seed": seed, "domain": "imputation", "gradient_data": "synthetic",
            "component": 6 if native else 0, "prior": "anchor_latent_mcar" if native else "low_rank_mcar"}
        truth = torch.tensor([[[0., 1.], [2., 3.], [4., 5.]]])
        mask = torch.tensor([[[True, False], [True, True], [False, True]]])
        source = truth.masked_fill(mask, float("nan"))
        episode = Episode(source, truth, mask, metadata)
        teacher = truth + .1
        specification = {"index": index, "name": f"task_{index}", "gradient_seed": seed + 100,
            "specification": {"seed": seed, "domain": "imputation", "component": metadata["component"]}, "weight": 1 / 28}
        path = output / f"training_batch/task_{index:02d}.pt"
        payload = {"episode": probe.training.alignment.mixture.backend.cache_episode(episode),
                   "teacher": teacher, "task": specification}
        torch.save(payload, path)
        metrics = probe.training.alignment.metric_summary(teacher, episode, probe.training.point_metric_terms(teacher, episode))
        task = {**specification, "artifact": str(path), "sha256": probe.sha256(path),
                "metadata": dict(metadata), "composite_loss": 1., "metrics_before": metrics}
        record = {"seed": seed, "eligible": native, "shape": list(truth.shape),
                  "queries": int(mask.sum()), "metadata": dict(metadata)}
        if native:
            path = output / f"retention_targets/reference_{seed}.pt"
            torch.save({"prediction": teacher, "source": source, "mask": mask, "identity": identity,
                "specification": probe.retention.SPECIFICATION, "metadata": dict(metadata)}, path)
            record.update(artifact=str(path), sha256=probe.sha256(path))
            replay_path = output / f"retention_replay/reference_{seed}.npz"
            probe.atomic_npz(replay_path, prediction=teacher.numpy())
            replays.append({"seed": seed, "artifact": str(replay_path), "sha256": probe.sha256(replay_path)})
        for role, offset in (("before", 0.), ("candidate", .25)):
            path = output / f"{role}_task_{index:02d}.npz"
            probe.atomic_npz(path, prediction=(teacher + offset).numpy())
            endpoints[role].append({"task": task["name"], "prediction": str(path), "sha256": probe.sha256(path)})
        specifications.append(specification)
        tasks.append(task)
        decisions.append(record)
    probe.write_json(output / "config.json", {"batch": {"tasks": specifications}})
    probe.write_json(output / "training_batch/manifest.json", {"tasks": tasks})
    reference = {**identity, "state_tensors": 254, "bitwise_equal": True}
    probe.write_json(output / "retention_targets/manifest.json", {"complete": True, "policy": probe.retention.POLICY,
        "specification": probe.retention.SPECIFICATION, "reference": reference, "records": decisions,
        "backward_source_sha256": probe.retention.BACKWARD_SOURCE_SHA256, "inference_ensemble": False})
    probe.write_json(output / "retention_replay/manifest.json", {"complete": True, "pid": os.getpid() + 1,
        "reference": reference, "records": replays,
        "input_manifest_sha256": probe.sha256(output / "retention_targets/manifest.json")})
    for role, rows in endpoints.items():
        probe.write_json(output / f"{role}_predictions.json", {"rows": rows})
    probe.write_json(output / "result.json", {"training_pid": os.getpid() + 2, "data_admission_only": False})
    probe.write_json(output / "audit.json", {"complete": True})
    return decisions


@pytest.mark.parametrize("mutation", [None, "omit", "eligibility", "duplicate", "source", "mask", "identity", "nonfinite", "shape", "task"])
def test_complete_retention_manifest_binds_every_task_source_mask_and_prediction(tmp_path, mutation):
    decisions = retention_fixture(tmp_path)
    manifest_path = tmp_path / "retention_targets/manifest.json"
    manifest = probe.read(manifest_path)
    if mutation in ("omit", "eligibility", "duplicate"):
        if mutation == "omit":
            manifest["records"].pop()
        elif mutation == "eligibility":
            manifest["records"][0]["eligible"] = False
        else:
            manifest["records"][1]["seed"] = manifest["records"][0]["seed"]
        probe.write_json(manifest_path, manifest)
    elif mutation == "task":
        config = probe.read(tmp_path / "config.json")
        config["batch"]["tasks"][0]["weight"] *= 2
        probe.write_json(tmp_path / "config.json", config)
    elif mutation:
        path = decisions[0]["artifact"]
        payload = torch.load(path, weights_only=False)
        if mutation == "source":
            payload["source"][0, 0, 1] += 1
        elif mutation == "mask":
            payload["mask"][0, 0, 0] = False
        elif mutation == "identity":
            payload["identity"]["sha256"] = "wrong"
        elif mutation == "nonfinite":
            payload["prediction"][0, 0, 0] = float("nan")
        else:
            payload["prediction"] = payload["prediction"][:, :1]
        torch.save(payload, path)
        manifest["records"][0]["sha256"] = probe.sha256(probe.Path(path))
        probe.write_json(manifest_path, manifest)
    if mutation:
        with pytest.raises(ValueError):
            probe.retention_records(tmp_path)
    else:
        report, records = probe.retention_records(tmp_path)
        assert len(report["records"]) == 28 and len(records) == 2
        assert [item[1]["seed"] for item in records] == [1000, 1014]


@pytest.mark.parametrize("mutation", [None, "same_process", "different_prediction", "missing_reference"])
def test_independent_retention_audit_checks_all_grids_and_original_query_penalties(tmp_path, monkeypatch, mutation):
    retention_fixture(tmp_path)
    monkeypatch.setattr(probe, "audit_original", lambda output: None)
    monkeypatch.setattr(probe, "verify", lambda output: probe.read(output / "config.json"))
    path = tmp_path / "retention_replay/manifest.json"
    replay = probe.read(path)
    if mutation == "same_process":
        replay["pid"] = os.getpid()
    elif mutation == "missing_reference":
        replay["records"].pop()
    elif mutation == "different_prediction":
        prediction_path = probe.Path(replay["records"][0]["artifact"])
        with np.load(prediction_path, allow_pickle=False) as archive:
            changed = archive["prediction"].copy()
        changed[0, 0, 0] += .1
        probe.atomic_npz(prediction_path, prediction=changed)
        replay["records"][0]["sha256"] = probe.sha256(prediction_path)
    probe.write_json(path, replay)
    if mutation:
        with pytest.raises(ValueError):
            probe.audit(tmp_path)
        assert not (tmp_path / "retention_audit.json").exists()
    else:
        probe.audit(tmp_path)
        report = probe.read(tmp_path / "retention_audit.json")
        assert report["complete"] and report["eligibility_decisions"] == 28 and report["reference_grids"] == 2
        assert report["no_inference_ensemble"] and report["full_autograd_replayed"] is False
        for row in report["endpoint_penalties"]:
            assert row["queries"] == 4 and row["before"] == 0.
            assert row["candidate"] == pytest.approx(.625, abs=1e-6)
        probe.verify_hashes(report["artifact_sha256"])
