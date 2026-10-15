import ast
from copy import deepcopy
import inspect
import os

import numpy as np
import pytest
import torch

import preflight_single_model_shrunk_column_anchor_retention as probe
import test_single_model_anchor_retention_preflight as original_tests
import test_single_model_column_anchor_retention_preflight as column_tests


def latest():
    records = column_tests.latest()
    records[1]["accepted_updates"] = 59
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("campaign",), "campaign"), (0, ("review",), "review"),
    (0, ("no_inference_ensemble",), False), (1, ("accepted_updates",), 60), (1, ("attempted_batches",), 63),
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
def test_requires_the_actual_complete_negative_v38_evidence(entry, path, value):
    records = latest()
    probe.validate_latest(*records)
    target = records[entry]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        probe.validate_latest(*records)


def test_probe_has_fresh_seeds_and_preserves_original_full_network_recipe():
    assert probe.latest.OUTPUT.name == "column_anchor_retention_v38_pipeline"
    assert probe.OUTPUT.name == "shrunk_column_anchor_retention_preflight_v1"
    assert probe.PARENT == probe.predecessor.PARENT == probe.latest.training.PARENT
    assert probe.PARENT_SHA256 == probe.retention.REFERENCE_SHA256
    assert probe.RESERVES == probe.predecessor.RESERVES and probe.WRITE_BUDGETS == probe.predecessor.WRITE_BUDGETS
    assert probe.RATE == 5e-6 and probe.paired is probe.predecessor.paired
    assert probe.teacher_policy is probe.predecessor.teacher_policy
    assert probe.DATA_SEED == 931100901 and probe.GRADIENT_SEED == 932100901
    current, previous = probe.batch(), probe.predecessor.batch()
    seeds = probe.training.alignment.fixed.seeds_in(current)
    assert len(seeds) == 56 and not seeds.intersection(probe.training.alignment.fixed.seeds_in(previous))
    for current_task, previous_task in zip(current["tasks"], previous["tasks"]):
        restored = deepcopy(current_task)
        restored["gradient_seed"] = previous_task["gradient_seed"]
        current_specification = restored["specification"].get("structured", restored["specification"])
        previous_specification = previous_task["specification"].get("structured", previous_task["specification"])
        current_specification["seed"] = previous_specification["seed"]
        assert restored == previous_task
    assert "retention_training.require_report(training, audit)" in probe.latest.ADAPTED_SOURCE
    assert probe.retention.SPECIFICATION["prior_count"] == probe.retention.SPECIFICATION["prior_variance"] == 1.
    assert probe.retention.SPECIFICATION["variance_floor"] == 1e-6
    assert not torch.cuda.is_initialized()


def test_generated_helpers_use_the_new_globals_and_require_independent_penalty_audit(monkeypatch):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_cloned_helpers_resolve_new_trial_globals_not_old_owners()
    original_tests.test_generated_run_reaches_real_retention_context_and_does_not_modify_inference()
    nodes = {node.name: node for node in ast.parse(probe.SOURCE).body if isinstance(node, ast.FunctionDef)}
    audit_source = ast.get_source_segment(probe.SOURCE, nodes["audit"])
    assert "retention.independent_penalty" in audit_source
    assert "difference.square().mean()" not in audit_source
    assert inspect.unwrap(probe.retention.runtime).__globals__ is vars(probe.retention)


@pytest.mark.parametrize("mutation", [None, "omit", "eligibility", "duplicate", "source", "mask", "identity", "nonfinite", "shape", "task"])
def test_preserves_every_original_task_reference_source_and_query_mask(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(original_tests, "probe", probe)
    original_tests.test_complete_retention_manifest_binds_every_task_source_mask_and_prediction(tmp_path, mutation)


@pytest.mark.parametrize("mutation", [None, "same_process", "different_prediction", "missing_reference"])
def test_independent_audit_recomputes_count_shrunk_penalties(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(original_tests, "probe", probe)
    decisions = original_tests.retention_fixture(tmp_path)
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
            prediction = archive["prediction"].copy()
        prediction[0, 0, 0] += .1
        probe.atomic_npz(prediction_path, prediction=prediction)
        replay["records"][0]["sha256"] = probe.sha256(prediction_path)
    probe.write_json(path, replay)
    if mutation:
        with pytest.raises(ValueError):
            probe.audit(tmp_path)
        assert not (tmp_path / "retention_audit.json").exists()
        return
    probe.audit(tmp_path)
    report = probe.read(tmp_path / "retention_audit.json")
    assert report["complete"] and report["eligibility_decisions"] == 28 and report["reference_grids"] == 2
    assert report["no_inference_ensemble"] and report["full_autograd_replayed"] is False
    for row in report["endpoint_penalties"]:
        index = int(row["task"].split("_")[-1])
        payload = torch.load(decisions[index]["artifact"], weights_only=False)
        cached = torch.load(tmp_path / f"training_batch/task_{index:02d}.pt", weights_only=False)
        episode = probe.training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        reference = payload["prediction"].double()
        candidate = (payload["prediction"] + .25).double()
        contributions = []
        for column_index in range(episode.target.shape[2]):
            queried = episode.mask[0, :, column_index]
            count = int(queried.sum())
            if count:
                variance = episode.target[0, queried, column_index].double().var(unbiased=False)
                denominator = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6)
                contributions.append(((candidate - reference)[0, queried, column_index].square().mean() / denominator).item())
        assert row["queries"] == 4 and row["before"] == 0.
        assert row["candidate"] == pytest.approx(10. * sum(contributions) / len(contributions), rel=1e-6)
    probe.verify_hashes(report["artifact_sha256"])
