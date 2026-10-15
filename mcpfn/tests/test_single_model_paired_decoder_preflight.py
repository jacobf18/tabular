from copy import deepcopy

import pytest

import preflight_single_model_paired_decoder as preflight


def test_only_fresh_seeds_and_paired_training_rule_change():
    actual = preflight.batch()
    previous = deepcopy(preflight.preceding.training.batches()[0])
    assert len(actual["tasks"]) == len(previous["tasks"]) == 28
    for current, baseline in zip(actual["tasks"], previous["tasks"]):
        normalized = deepcopy(current)
        normalized["gradient_seed"] = baseline["gradient_seed"]
        recipe = normalized["specification"].get("structured", normalized["specification"])
        recipe["seed"] = baseline["specification"].get("structured", baseline["specification"])["seed"]
        assert normalized == baseline
        assert current["gradient_seed"] != baseline["gradient_seed"]
    assert preflight.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.RATE == 2e-5
    seeds = preflight.training.alignment.fixed.seeds_in(actual)
    assert len(seeds) == 56
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(previous))


def test_actual_execution_and_independent_audit_use_paired_helpers_without_replacing_originals():
    assert "paired.protected_update(" in preflight.ADAPTED_SOURCE
    assert "training.guarded.protected_update(" not in preflight.ADAPTED_SOURCE
    assert "paired.check_record(" in preflight.ADAPTED_SOURCE
    assert "paired.check_endpoint(" in preflight.ADAPTED_SOURCE
    assert "decision = paired.metric_decision(" in preflight.ADAPTED_SOURCE
    assert preflight.ADAPTED_SOURCE.count("paired.validate_layout(") == 2
    assert preflight.training.guarded.protected_update is not preflight.paired.protected_update


def test_bounded_storage_floor_does_not_change_the_original_write_reserve(monkeypatch):
    class Usage:
        free = 24 * 1024**3 - 1

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage())
    with pytest.raises(OSError, match="18 GiB"):
        preflight.require_space()
    Usage.free += 1
    preflight.require_space()
    assert "wait_for_gpu(output, True, False)\n    require_space()" in preflight.ADAPTED_SOURCE


@pytest.mark.parametrize("mutation", ["incomplete", "qualified", "unconfirmed", "live", "campaign", "review"])
def test_no_probe_before_the_unchanged_predecessor_pipeline_finishes(monkeypatch, mutation):
    result = {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True},
              "campaign": None, "review": None}
    terminal = {"state": "complete", "pid": 999999999}
    if mutation == "incomplete":
        result["complete"] = False
    elif mutation == "qualified":
        result["gate"]["passed"] = True
    elif mutation == "unconfirmed":
        result["gate"]["confirmation_evaluated"] = False
    elif mutation == "live":
        terminal["state"] = "confirm"
    else:
        result[mutation] = "existing"
    monkeypatch.setattr(preflight, "read", lambda path: result if path.name == "result.json" else terminal)
    monkeypatch.setattr(preflight.preceding, "prerequisite", lambda path: pytest.fail("Never bypass terminal state"))
    with pytest.raises(ValueError, match="completed unsuccessful"):
        preflight.require_predecessor()
