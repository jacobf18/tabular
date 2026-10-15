from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import preflight_single_model_dual_parent_retention as probe
import preflight_single_model_parent_structured_retention as predecessor
import test_single_model_anchor_retention as originals
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_fresh_batch_preserves_mixture_optimizer_and_original_guards():
    assert (probe.DATA_SEED, probe.GRADIENT_SEED) == (1291100901, 1292100901)
    assert probe.OUTPUT.name == "dual_parent_retention_preflight_v1"
    assert probe.PARENT == predecessor.PARENT
    assert (probe.TRAINABLE_PARAMETERS, probe.TRAINABLE_TENSORS, probe.FROZEN_TENSORS) == (77998496, 238, 16)
    expected = deepcopy(predecessor.batch())
    for task in expected["tasks"]:
        task["gradient_seed"] += 30000000
        specification = task["specification"]
        specification.get("structured", specification)["seed"] += 30000000
    assert probe.batch() == expected
    assert len(expected["tasks"]) == 28
    assert probe.RATE == predecessor.RATE == 5e-6
    assert probe.RESERVES == predecessor.RESERVES and probe.WRITE_BUDGETS == predecessor.WRITE_BUDGETS
    assert probe.ARITHMETIC == predecessor.ARITHMETIC
    assert probe.teacher_policy is predecessor.teacher_policy and probe.stronger is predecessor.stronger
    assert probe.training.alignment.fixed.PARENT == predecessor.training.alignment.fixed.PARENT


def test_complete_sources_identities_and_component_audits_are_bound():
    identities = probe.reference_identities()
    assert set(identities) == {"general", "v8", "v29"}
    assert identities["v29"]["checkpoint"] == str(probe.PARENT)
    assert identities["v8"]["checkpoint"] == str(probe.training.alignment.fixed.PARENT)
    assert probe.retention is probe.cohort_retention
    assert probe.retention.BACKWARD_SOURCE == predecessor.retention.BACKWARD_SOURCE
    for role in ("v8", "v29", "general"):
        assert f'Path(reference_identities()["{role}"]["checkpoint"])' in probe.REGISTER_SOURCE
    for name in ("dual_parent_retention", "dual_parent_retention_preflight"):
        assert f'test_single_model_{name}.py' in probe.REGISTER_SOURCE
    assert '"native_teacher_mean": retention.MEAN_SPECIFICATION' in probe.REGISTER_SOURCE
    assert '"mean_distillation": stronger.SPECIFICATION' in probe.REGISTER_SOURCE
    assert 'require_completed_v49' in probe.require_predecessor.__code__.co_names
    assert 'validate_saved_components' in probe.retention_records.__code__.co_names
    assert 'validate_replay_archive' in probe.audit.__code__.co_names
    for function in vars(probe).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(probe):
            assert unresolved_globals(function) == []


def complete_diagnosis():
    rows = [{"reference": "V8", "group": family, "metric": "common_nrmse",
        "initializer_failed": True, "candidate_failed": True} for family in ("mar", "mnar_skip_logic")]
    rows.extend({"reference": "V8", "group": f"passing_{index}", "metric": "metric",
        "initializer_failed": False, "candidate_failed": False} for index in range(30))
    report = {"complete": True, "prediction_grids": 284, "comparisons": rows,
        "post_hoc_consumed_bank": True, "goal_success_not_assessed": True, "artifact_sha256": {}}
    report.update({name: False for name in ("cuda_initialized", "training_performed", "confirmation_repeated",
        "checkpoint_selection_performed", "real_benchmarks_evaluated")})
    return report


@pytest.mark.parametrize("failure", [None, "live", "unfinished", "partial", "dropped", "repaired", "new_failure",
    "cuda_initialized", "training_performed", "confirmation_repeated", "checkpoint_selection_performed",
    "real_benchmarks_evaluated", "post_hoc_consumed_bank", "goal_success_not_assessed"])
def test_new_probe_requires_exact_stopped_negative_attribution(monkeypatch, failure):
    report = complete_diagnosis()
    state = {"state": "complete", "pid": 101}
    monkeypatch.setattr(probe.diagnosis, "verify", lambda: None)
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: {"pid": pid} if failure == "live" else None)
    monkeypatch.setattr(probe, "verify_hashes", lambda mapping: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "hash")
    monkeypatch.setattr(probe, "read", lambda path: state if path.name == "status.json" else report)
    if failure == "unfinished":
        state["state"] = "awaiting_independent_audit"
    elif failure == "partial":
        report["prediction_grids"] = 283
    elif failure == "dropped":
        report["comparisons"].pop()
    elif failure == "repaired":
        report["comparisons"][0]["candidate_failed"] = False
    elif failure == "new_failure":
        report["comparisons"][2]["initializer_failed"] = True
    elif failure in report:
        report[failure] = not report[failure]
    if failure is None:
        assert probe.require_completed_v49()
    else:
        with pytest.raises(ValueError):
            probe.require_completed_v49()


def payload_for(episode):
    role = probe.retention.reference_role(episode.metadata)
    components = {name: torch.full_like(episode.source, {"general": 1., "v8": 2., "v29": 4.}[name])
        for name in probe.retention.reference_weights(role)}
    return {"components": components, "component_weights": probe.retention.reference_weights(role),
        "prediction": probe.retention.combine_predictions(role, components)}


@pytest.mark.parametrize("failure", [None, "weights", "drop", "prediction", "nonfinite"])
def test_saved_target_requires_every_fixed_component(failure):
    episode = originals.episode()
    episode.metadata = originals.metadata("low_rank_mcar", structured=True, family="mar")
    payload = payload_for(episode)
    if failure == "weights":
        payload["component_weights"]["v8"] = .6
    elif failure == "drop":
        payload["components"].pop("v8")
    elif failure == "prediction":
        payload["prediction"] += .1
    elif failure == "nonfinite":
        payload["components"]["v8"][episode.mask] = float("nan")
    if failure is None:
        probe.validate_saved_components(episode, payload)
    else:
        with pytest.raises(ValueError):
            probe.validate_saved_components(episode, payload)


@pytest.mark.parametrize("failure", [None, "missing", "offsetting_components", "combined"])
def test_cpu_replay_rejects_component_tampering_even_with_unchanged_mean(tmp_path, failure):
    episode = originals.episode()
    episode.metadata = originals.metadata("low_rank_mcar", structured=True, family="mar")
    payload = payload_for(episode)
    arrays = {"prediction": payload["prediction"].numpy().copy(),
        **{f"component_{name}": value.numpy().copy() for name, value in payload["components"].items()}}
    if failure == "missing":
        arrays.pop("component_v8")
    elif failure == "offsetting_components":
        arrays["component_v8"] += 1
        arrays["component_v29"] -= 1
    elif failure == "combined":
        arrays["prediction"] += 1
    path = tmp_path / "reference.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if failure is None:
            probe.validate_replay_archive(archive, "structured_mean", payload)
        else:
            with pytest.raises(ValueError):
                probe.validate_replay_archive(archive, "structured_mean", payload)


def test_independent_replay_runs_all50_components_and_preserves_all26_grids(tmp_path, monkeypatch):
    references = {name: SimpleNamespace(name=name) for name in probe.reference_identities()}
    verified = {name: {"name": name} for name in references}
    records = []
    for index in range(26):
        episode = originals.episode()
        episode.metadata = originals.metadata() if index < 2 else originals.metadata(
            "low_rank_mcar", structured=True, family=probe.retention.FAMILIES[(index - 2) // 2])
        episode.metadata["seed"] = 1000 + index
        records.append(({}, {"seed": episode.metadata["seed"]}, episode, payload_for(episode)))
    calls = Counter()

    def predict(reference, source, criterion, **options):
        calls[reference.name] += 1
        return torch.full_like(source, {"general": 1., "v8": 2., "v29": 4.}[reference.name])

    monkeypatch.setattr(probe, "replay_original_teachers", lambda output: None)
    monkeypatch.setattr(probe, "verify", lambda output: {"base": {}})
    monkeypatch.setattr(probe, "read", lambda path: {})
    monkeypatch.setattr(probe, "status", lambda *arguments, **keywords: None)
    monkeypatch.setattr(probe, "sha256", lambda path: "hash")
    monkeypatch.setattr(probe, "retention_records", lambda output: ({"reference": verified}, records))
    monkeypatch.setattr(probe.training.alignment.mixture, "original_setup", lambda base: (None,) * 5)
    monkeypatch.setattr(probe.retention, "make_references", lambda model, identities: references)
    monkeypatch.setattr(probe.retention, "verify_model", lambda model, identity: verified[model.name])
    monkeypatch.setattr(probe, "teacher_predictions", predict)
    monkeypatch.setattr(probe, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    probe.replay_cohort_targets(tmp_path)
    report = json.loads((tmp_path / "retention_replay/manifest.json").read_text())
    assert dict(calls) == {"general": 2, "v8": 24, "v29": 24}
    assert report["component_grids"] == 50 and len(report["records"]) == 26
    for row, (task, record, episode, payload) in zip(report["records"], records):
        with np.load(row["artifact"], allow_pickle=False) as archive:
            probe.validate_replay_archive(archive, row["reference_role"], payload)
