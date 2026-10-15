from copy import deepcopy
import math

import pytest

import preflight_single_model_decoder_training as previous
import preflight_single_model_targeted_decoder as targeted


def test_targeted_batch_preserves_every_family_and_domain_mass():
    tasks = targeted.batch()["tasks"]
    assert len(tasks) == 14
    assert [task["name"] for task in tasks] == [task["name"] for task in previous.batch()["tasks"]]
    weights = {task["name"]: task["weight"] for task in tasks}
    assert weights["legacy"] == .25
    assert weights["recsys"] == .5
    assert weights["mar"] == weights["mnar_skip_logic"] == .1
    assert math.fsum(weights.values()) == 1.
    assert math.fsum(task["weight"] for task in tasks if "structured" in task["specification"]) == .25
    assert all(task["weight"] > 0 for task in tasks)


def test_only_prospectively_declared_prior_fields_and_seeds_change():
    old = previous.batch()["tasks"]
    current = targeted.batch()["tasks"]
    old_seeds, fresh_seeds = set(), set()
    for before, after in zip(old, current):
        expected = deepcopy(before)
        expected["gradient_seed"] = targeted.GRADIENT_SEED + before["index"] * 100
        expected["specification"].get("structured", expected["specification"])["seed"] = targeted.DATA_SEED + before["index"] * 100
        if "structured" in expected["specification"]:
            expected["weight"] = .1 if before["name"] in {"mar", "mnar_skip_logic"} else .005
        if before["name"] in {"mar", "mnar_skip_logic"}:
            expected["specification"]["structured"].update(
                component="low_rank_mcar" if before["name"] == "mar" else "steep_tail_mcar", shape=[240, 80])
        assert after == expected
        old_seeds.update((before["gradient_seed"], before["specification"].get("structured", before["specification"])["seed"]))
        fresh_seeds.update((after["gradient_seed"], after["specification"].get("structured", after["specification"])["seed"]))
    assert len(fresh_seeds) == 28
    assert fresh_seeds.isdisjoint(old_seeds)


def test_unqualified_input_does_not_replace_v8_teachers_or_baseline():
    assert targeted.PARENT == targeted.ROOT / "decoder_repair_v14/latest_checkpoint.pth"
    assert targeted.PARENT != targeted.training.alignment.fixed.PARENT
    assert targeted.training is previous.training
    assert targeted.training.alignment.fixed.PARENT == previous.training.alignment.fixed.PARENT
    assert targeted.RATE == previous.RATE == 2e-6


def test_preflight_registration_rejects_incomplete_prior_trial(monkeypatch, tmp_path):
    monkeypatch.setattr(targeted, "OUTPUT", tmp_path / "new")
    monkeypatch.setattr(targeted.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 30 * 1024**3})())
    reports = {
        targeted.ROOT / "decoder_repair_v14_pipeline/result.json": {"complete": False},
        targeted.ROOT / "decoder_repair_v14_audit/audit.json": {},
        targeted.ROOT / "decoder_repair_v14_endpoint_replay/audit.json": {},
        targeted.EVIDENCE / "result.json": {},
    }
    monkeypatch.setattr(targeted, "read", lambda path: reports[path])
    with pytest.raises(ValueError, match="fully audited"):
        targeted.register(targeted.OUTPUT)
    assert not targeted.OUTPUT.exists()
