"""Preserve all original training data and constraints when changing proposal weights."""

from copy import deepcopy
import math

import pytest

import preflight_single_model_legacy_strata as baseline
import single_model_legacy_heavy_mixture as mixture


def test_reweight_changes_only_weights_and_preserves_each_original_draw():
    batch = baseline.batch()
    unchanged = deepcopy(batch)
    result = mixture.reweight(batch)
    assert batch == unchanged
    assert result is not batch
    assert mixture.validate(result) == batch
    for original, current in zip(batch["tasks"], result["tasks"]):
        assert {name: value for name, value in original.items() if name != "weight"} == {
            name: value for name, value in current.items() if name != "weight"}
        assert current["weight"] == original["weight"] * mixture.FACTORS[mixture.group(original)]
    for group, mass in mixture.MASSES.items():
        assert math.isclose(math.fsum(task["weight"] for task in result["tasks"] if mixture.group(task) == group), mass,
                            rel_tol=0, abs_tol=1e-15)
    assert baseline.training.alignment.fixed.seeds_in(result) == baseline.training.alignment.fixed.seeds_in(batch)


def test_projection_layout_and_raw_objectives_do_not_change_with_weights():
    batch = baseline.batch()
    changed = mixture.reweight(batch)
    before = {f'{task["name"]}/{metric}': 1. for task in batch["tasks"]
              for metric in (("rmse", "mae") if mixture.group(task) == "recsys" else ("rmse", "common_nrmse"))}
    labels, groups, matrix = baseline.paired.gradient_layout(before, 28)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    assert len(labels) == 84 and len(groups) == 45
    assert (baseline.paired.validate_layout(entries, changed["tasks"], before) == matrix).all()
    assert (baseline.paired.validate_layout(entries, batch["tasks"], before) == matrix).all()
    assert 'backend.backward_episode(model, criterion, episode, teacher, 1.)' in baseline.replicated.CAPTURE_SOURCE
    assert [task["weight"] for task in changed["tasks"]] != [task["weight"] for task in batch["tasks"]]


@pytest.mark.parametrize("mutation", ["negative", "nan", "missing_draw", "mismatched_pair", "unstratified", "wrong_domain"])
def test_reweight_rejects_changed_source_scope(mutation):
    batch = baseline.batch()
    if mutation == "negative":
        batch["tasks"][0]["weight"] = -.125
    elif mutation == "nan":
        batch["tasks"][0]["weight"] = math.nan
    elif mutation == "missing_draw":
        batch["tasks"].pop()
    elif mutation == "mismatched_pair":
        batch["tasks"][14]["weight"] *= .5
    elif mutation == "unstratified":
        batch["tasks"][14]["specification"]["component"] = 0
    else:
        batch["tasks"][0]["specification"]["domain"] = "unknown"
    with pytest.raises(ValueError):
        mixture.reweight(batch)


def test_reweight_is_not_repeatable_and_validation_detects_changed_mass():
    batch = mixture.reweight(baseline.batch())
    with pytest.raises(ValueError):
        mixture.reweight(batch)
    batch["tasks"][0]["weight"] *= 2
    batch["tasks"][14]["weight"] *= 2
    with pytest.raises(ValueError):
        mixture.validate(batch)
