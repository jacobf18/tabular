from copy import deepcopy
import math

import pytest

import single_model_anchor_rehearsal as rehearsal
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import train_single_matrixformer_targeted_decoder as schedule


@pytest.mark.parametrize("step", range(1, 65))
def test_only_second_imputation_component_changes_in_every_original_schedule_position(step):
    original = replicated.replicate_batch(schedule.batches()[step - 1], 611100901, 612100901)
    saved = deepcopy(original)
    actual = rehearsal.stratify(original)
    assert original == saved and actual is not original
    assert actual["step"] == step and len(actual["tasks"]) == 28
    for task, source in zip(actual["tasks"], original["tasks"]):
        specification = task["specification"]
        restored = deepcopy(task)
        if task["replicate"] == 1 and specification["domain"] == "imputation":
            if "structured" in specification:
                assert specification["structured"]["component"] == "anchor_latent_mcar"
                restored["specification"]["structured"]["component"] = source["specification"]["structured"]["component"]
            else:
                assert specification["component"] == 6
                restored["specification"]["component"] = source["specification"]["component"]
        assert restored == source
        assert task["weight"] == actual["tasks"][(task["index"] + 14) % 28]["weight"]
    assert math.fsum(task["weight"] for task in actual["tasks"]) == 1.
    assert schedule.original.alignment.fixed.seeds_in(actual) == schedule.original.alignment.fixed.seeds_in(original)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "step", "negative_weight", "nonfinite_weight",
    "mass", "unequal_pair", "replicate", "cohort", "legacy_component", "structured_component", "family", "domain", "ratings"])
def test_invalid_or_relabelled_batch_is_rejected(mutation):
    batch = replicated.replicate_batch(schedule.batches()[0], 611100901, 612100901)
    tasks = batch["tasks"]
    if mutation == "missing":
        tasks.pop()
    elif mutation == "duplicate":
        tasks[1] = deepcopy(tasks[0])
    elif mutation == "order":
        tasks.reverse()
    elif mutation == "step":
        batch["step"] = True
    elif mutation == "negative_weight":
        tasks[0]["weight"] = -.125
    elif mutation == "nonfinite_weight":
        tasks[0]["weight"] = float("nan")
    elif mutation == "mass":
        tasks[0]["weight"] = tasks[14]["weight"] = .1
    elif mutation == "unequal_pair":
        tasks[0]["weight"] += .01
        tasks[14]["weight"] -= .01
    elif mutation == "replicate":
        tasks[0]["replicate"] = 1
    elif mutation == "cohort":
        tasks[0]["cohort"] = "mcar"
    elif mutation == "legacy_component":
        tasks[0]["specification"]["component"] = "anchor_latent_mcar"
    elif mutation == "structured_component":
        tasks[1]["specification"]["structured"]["component"] = "invented"
    elif mutation == "family":
        tasks[1]["specification"]["structured"]["family"] = "mar"
    elif mutation == "domain":
        tasks[1]["specification"]["domain"] = "real_data"
    else:
        tasks[13]["specification"]["structured"] = {"component": "anchor_latent_mcar"}
    with pytest.raises(ValueError):
        rehearsal.stratify(batch)


def test_original_and_anchor_strata_keep_paired_point_guards_and_individual_regressions():
    tasks = rehearsal.stratify(replicated.replicate_batch(schedule.batches()[0], 611100901, 612100901))["tasks"]
    before = {f'{task["name"]}/{metric}': 1. for task in tasks for metric in
              (("rmse", "mae") if task["cohort"] == "recsys" else ("rmse", "common_nrmse"))}
    labels, names, matrix = paired.gradient_layout(before, 28)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    assert paired.validate_layout(entries, tasks, before).shape == (42, 84)
    assert matrix.shape == (42, 84) and len(names) == 42
    after = dict(before)
    after["legacy_draw_0/rmse"] = 1.1
    after["legacy_draw_1/rmse"] = .8
    decision = paired.metric_decision(before, after)
    assert decision["all_retained"]
    assert not decision["individual_draw_diagnostics"]["all_retained"]
