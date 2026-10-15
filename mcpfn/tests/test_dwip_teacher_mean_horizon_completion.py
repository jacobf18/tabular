from copy import deepcopy

import numpy as np
import pytest
import torch

from audit_dwip_teacher_mean_horizon import (
    DATASETS, check_model_state, check_optimizer, complete_panels, expected_lr, matched_data_records, training_records,
)


def records():
    return [{"step": step, **dict.fromkeys(("loss_missing", "nll_missing", "teacher_point_loss_missing", "cdf_loss_missing",
             "rmse_missing_mean", "teacher_mean_rmse_missing", "grad_norm"), .1), "lr": expected_lr(step)} for step in [1, *range(25, 128001, 25)]]


def test_terminal_requires_complete_finite_honestly_labeled_training_stream():
    assert training_records(records()) == 5121


def test_shortened_schedule_is_not_a_horizon_run():
    rows = records()
    rows[40]["lr"] *= .8
    with pytest.raises(ValueError, match="learning rate"):
        training_records(rows)


def data_records():
    return [{"step": step, "matrix_rows": 80, "matrix_cols": 35, "batch_size": 2,
             "sampled_prior_index": 3, "sampled_prior_name": "generic", "support_missing_fraction": .5,
             "support_variance_df": 1., "lr": .01, "loss_missing": 1.} for step in [1, *range(25, 64001, 25)]]


def test_horizon_preserves_all2561_data_records_not_losses_or_learning_rates():
    previous = data_records()
    candidate = deepcopy(previous)
    for row in candidate:
        row["lr"], row["loss_missing"] = .03, .6
    candidate.append({"step": 64025})
    assert matched_data_records(candidate, previous) == 2561


@pytest.mark.parametrize("change", ("missing", "duplicate", "shape", "component", "support", "support_key"))
def test_data_prefix_mismatches_are_rejected(change):
    previous, candidate = data_records(), data_records()
    if change == "missing":
        candidate.pop()
    elif change == "duplicate":
        candidate[-1]["step"] = candidate[-2]["step"]
    elif change == "shape":
        candidate[30]["matrix_cols"] = 50
    elif change == "component":
        candidate[30]["sampled_prior_name"] = "another"
    elif change == "support":
        candidate[30]["support_variance_df"] = 3.
    else:
        candidate[30].pop("support_missing_fraction")
    with pytest.raises(ValueError):
        matched_data_records(candidate, previous)


def test_horizon_selection_requires65_records_and_earliest_strict_minimum():
    from audit_dwip_continuous_completion import fixed_selection
    validation = [{"step": step, "macro_rmse": 1. if step >= 64000 else 2.} for step in range(0, 128001, 2000)]
    assert fixed_selection(validation, validation[32], final_step=128000)["step"] == 64000
    with pytest.raises(ValueError, match="minimum"):
        fixed_selection(validation, validation[-1], final_step=128000)
    with pytest.raises(ValueError, match="schedule"):
        fixed_selection(validation[:-1], validation[32], final_step=128000)


def test_only_best_and_rolling_terminal_are_permitted():
    from audit_dwip_teacher_mean_horizon import MODEL_FILES, ROLLING
    assert MODEL_FILES == ("best_validation_checkpoint.pth",)
    assert ROLLING == "latest_checkpoint.pth"


def test_declared_learning_rates_match_native_scheduler_at_every_sampled_step():
    from dwip_teacher_mean_horizon import horizon_recipe
    from dwip_teacher_mean_training import base
    args = horizon_recipe()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=args.lr)
    scheduler = base.make_scheduler(optimizer, warmup_steps=600, stable_steps=0, decay_steps=127400, min_lr_ratio=.05)
    for row in records():
        assert row["lr"] == pytest.approx(args.lr * scheduler.lr_lambdas[0](row["step"]), rel=1e-14)


@pytest.mark.parametrize("change", ("missing", "nonfinite", "old_label", "duplicate"))
def test_incomplete_training_is_not_completion(change):
    rows = records()
    if change == "missing":
        rows.pop()
    elif change == "nonfinite":
        rows[10]["grad_norm"] = float("nan")
    elif change == "old_label":
        rows[10]["point_loss_missing"] = rows[10].pop("teacher_point_loss_missing")
    else:
        rows[-1]["step"] = rows[-2]["step"]
    with pytest.raises(ValueError):
        training_records(rows)


def model_state():
    return {f"parameter_{index}": torch.ones(2) for index in range(254)}


@pytest.mark.parametrize("change", ("extra_teacher", "shape", "precision", "nonfinite"))
def test_model_state_must_remain_the_original_finite_fp32_architecture(change):
    reference = model_state()
    candidate = deepcopy(reference)
    check_model_state({"model": candidate}, reference)
    if change == "extra_teacher":
        candidate["teacher.weight"] = torch.ones(2)
    elif change == "shape":
        candidate["parameter_0"] = torch.ones(3)
    elif change == "precision":
        candidate["parameter_0"] = candidate["parameter_0"].bfloat16()
    else:
        candidate["parameter_0"][0] = float("inf")
    with pytest.raises(ValueError):
        check_model_state({"model": candidate}, reference)


def optimizer_payload():
    return {"epoch": 128000, "scheduler": {"last_epoch": 128000}, "optimizer": {"state": {
        index: {"step": torch.tensor(128000.), "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)} for index in range(238)}}}


def test_terminal_optimizer_covers_the_complete_cosine_run():
    assert check_optimizer(optimizer_payload()) == 238


@pytest.mark.parametrize("change", ("epoch", "scheduler", "count", "moment_dtype", "moment_finite", "step"))
def test_partial_or_invalid_optimizer_is_rejected(change):
    payload = optimizer_payload()
    if change == "epoch":
        payload["epoch"] = 62000
    elif change == "scheduler":
        payload["scheduler"]["last_epoch"] = 63000
    elif change == "count":
        payload["optimizer"]["state"].pop(0)
    elif change == "moment_dtype":
        payload["optimizer"]["state"][0]["exp_avg"] = torch.ones(2, dtype=torch.bfloat16)
    elif change == "moment_finite":
        payload["optimizer"]["state"][0]["exp_avg_sq"][0] = float("nan")
    else:
        payload["optimizer"]["state"][0]["step"] = torch.tensor(63999.)
    with pytest.raises(ValueError):
        check_optimizer(payload)


def arrays(value):
    return {"benchmark_names": np.array(["one", "two"]), **{
        f"{method}_{metric}": np.full((10, 15), value) for method in ("entropy", "mi", "random") for metric in ("r2", "rmse")}}


def panel_fixture():
    references = {label: {dataset: arrays(.2) for dataset in DATASETS} for label in ("original", "primary_mean64")}
    selected = {dataset: arrays(.3) for dataset in DATASETS}
    return references, selected


def test_full_panel_contains_all_360_paired_mean_settings_and_all_folds():
    reports, rows = complete_panels(*panel_fixture())
    assert len(rows) == 360
    for panels in reports.values():
        assert sum(panel["mean_setting_counts"]["improved"] for panel in panels.values()) == 180
        assert all(len(panel["fold_arrays"]["mi_r2"]["selected"]) == 10 for panel in panels.values())


@pytest.mark.parametrize("change", ("dataset", "reference", "fold", "nonfinite", "identities"))
def test_incomplete_or_mismatched_benchmarks_are_rejected(change):
    references, selected = panel_fixture()
    if change == "dataset":
        selected.pop("mteb")
    elif change == "reference":
        references["original"].pop("mmlu")
    elif change == "fold":
        selected["mteb"]["mi_r2"] = np.zeros((9, 15))
    elif change == "nonfinite":
        selected["mteb"]["mi_r2"][0, 0] = float("nan")
    else:
        selected["mteb"]["benchmark_names"] = np.array(["other"])
    with pytest.raises(ValueError):
        complete_panels(references, selected)
