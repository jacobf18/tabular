from collections import Counter, defaultdict
from copy import deepcopy
import math

import pytest

import single_model_anchor_rehearsal as rehearsal
import single_model_balanced_structured_components as balanced
import single_model_legacy_heavy_mixture as mixture
import single_model_legacy_stratum_protection as protection
import single_model_replicated_decoder_batch as replicated
import train_single_matrixformer_targeted_decoder as schedule


def original_batch(step):
    return mixture.reweight(rehearsal.stratify(replicated.replicate_batch(
        schedule.batches()[step - 1], 611100901, 612100901)))


@pytest.mark.parametrize("step", range(1, 65))
def test_every_original_schedule_position_changes_only_structured_components(step):
    original = original_batch(step)
    saved = deepcopy(original)
    actual = balanced.balance(original)
    assert original == saved and actual is not original
    counts, weights = Counter(), defaultdict(list)
    for task, source in zip(actual["tasks"], original["tasks"]):
        restored = deepcopy(task)
        if task["cohort"] in balanced.FAMILIES:
            component = task["specification"]["structured"]["component"]
            counts[component] += 1
            weights[component].append(task["weight"])
            restored["specification"]["structured"]["component"] = source["specification"]["structured"]["component"]
        assert restored == source
    assert counts == {component: 6 for component in balanced.COMPONENTS}
    assert all(math.isclose(math.fsum(values), .0625, rel_tol=0, abs_tol=1e-15) for values in weights.values())
    assert schedule.original.alignment.fixed.seeds_in(actual) == schedule.original.alignment.fixed.seeds_in(original)
    assert balanced.balance(actual) == actual


@pytest.mark.parametrize("start", range(1, 64))
def test_every_family_covers_all_four_components_across_two_attempts(start):
    grouped = defaultdict(Counter)
    for step in (start, start + 1):
        for task in balanced.balance(original_batch(step))["tasks"]:
            if task["cohort"] in balanced.FAMILIES:
                grouped[task["cohort"]][task["specification"]["structured"]["component"]] += 1
    assert set(grouped) == set(balanced.FAMILIES)
    assert all(counts == {component: 1 for component in balanced.COMPONENTS} for counts in grouped.values())


@pytest.mark.parametrize("mutation", ["missing", "order", "step", "weight", "replicate", "name", "family",
    "component", "domain", "native_anchor", "rating_domain", "rating_structured", "captured"])
def test_invalid_batches_are_rejected_without_mutation(mutation):
    batch = original_batch(1)
    tasks = batch["tasks"]
    if mutation == "missing":
        tasks.pop()
    elif mutation == "order":
        tasks.reverse()
    elif mutation == "step":
        batch["step"] = True
    elif mutation == "weight":
        tasks[1]["weight"] = .01
    elif mutation == "replicate":
        tasks[1]["replicate"] = True
    elif mutation == "name":
        tasks[1]["name"] = "mcar_draw_1"
    elif mutation == "family":
        tasks[1]["specification"]["structured"]["family"] = "mar"
    elif mutation == "component":
        tasks[1]["specification"]["structured"]["component"] = "invented"
    elif mutation == "domain":
        tasks[1]["specification"]["domain"] = "recsys"
    elif mutation == "native_anchor":
        tasks[14]["specification"]["component"] = 7
    elif mutation == "rating_domain":
        tasks[13]["specification"]["domain"] = "imputation"
    elif mutation == "rating_structured":
        tasks[13]["specification"]["structured"] = None
    else:
        tasks[1]["artifact"] = "already_captured.pt"
    saved = deepcopy(batch)
    with pytest.raises(ValueError):
        balanced.balance(batch)
    assert batch == saved


def test_original_point_protection_layout_and_individual_diagnostics_are_preserved():
    original = original_batch(1)
    tasks = balanced.balance(original)["tasks"]
    before = {f'{task["name"]}/{metric}': 1. for task in tasks for metric in
        (("rmse", "mae") if task["cohort"] == "recsys" else ("rmse", "common_nrmse"))}
    labels, names, matrix = protection.gradient_layout(before, 28)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    assert protection.validate_layout(entries, tasks, before).shape == matrix.shape == (45, 84)
    assert len(names) == 45
    after = dict(before)
    after["legacy_draw_0/rmse"], after["legacy_draw_1/rmse"] = 1.1, .8
    assert not protection.metric_decision(before, after)["all_retained"]
    after = dict(before)
    after["mar_draw_0/rmse"], after["mar_draw_1/rmse"] = 1.1, .8
    result = protection.metric_decision(before, after)
    assert result["all_retained"] and not result["individual_draw_diagnostics"]["all_retained"]
