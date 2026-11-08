from copy import deepcopy
from types import SimpleNamespace

import pytest

import check_single_model_four_draw_lifecycle_capacity as capacity


def fixture():
    batches = [{"step": step, "tasks": [{"index": index, "specification": {"shape": [2000, 1000] if index < 4 else [100, 20]}}
        for index in range(56)]} for step in range(1, 129)]
    shapes = [[1, 240, 80] for index in range(152)]
    stages = {name: {"checkpoint": 16 * 1024**2, "root": 16 * 1024**2, "scratch": 32 * 1024**2} for name in capacity.STAGES}
    stages["replay"]["scratch"] = 768 * 1024**2
    config = {"storage_accounting": {volume: {"training": 1024**3} for volume in ("checkpoint", "root", "scratch")},
        "diagnostic_storage": {"stage_write_budget_bytes": deepcopy(stages)}}
    return batches, shapes, config, {"stage_write_budget_bytes": stages}


def test_every_requested_endpoint_and_selected_grid_is_accounted_without_mutation():
    batches, shapes, config, pipeline = fixture()
    saved = deepcopy((batches, shapes, config, pipeline))
    report = capacity.replay_accounting(batches, shapes)
    assert report["maximum_requested_endpoint_cells"] == 8104000
    assert report["selected_prediction_cells"] == 152 * 240 * 80
    assert report["float32_arrays_before_metadata"] == 21 * 8104000 + 4 * capacity.PARAMETERS + 4 * 152 * 240 * 80
    assert sum(report["conservative_replay_fields"].values()) == report["conservative_replay_bytes"]
    assert report["conservative_replay_bytes"] <= capacity.SUPPLEMENTAL_REPLAY_BYTES
    assert report["actual_replay_usage_measured"] is False
    lifecycle = capacity.lifecycle_accounting(config, pipeline, report)
    assert lifecycle["original_stage_admission_bytes"]["replay"]["scratch"] == 768 * 1024**2
    assert lifecycle["supplemental_stage_accounting_bytes"]["replay"]["scratch"] == 2 * 1024**3
    assert lifecycle["whole_lifecycle_additional_byte_bound"]["checkpoint"] == 1024**3 + 6 * 16 * 1024**2 + capacity.RECEIPT_ALLOWANCE
    assert lifecycle["already_written_bytes_credited"] == 0
    assert (batches, shapes, config, pipeline) == saved


@pytest.mark.parametrize("mutation", ["attempt", "ordinal", "task", "index", "validation", "dimension", "boolean", "negative", "many_matrices", "too_large"])
def test_no_missing_task_smaller_subset_or_inadequate_headroom_is_accepted(mutation):
    batches, shapes, config, pipeline = fixture()
    if mutation == "attempt":
        batches.pop()
    elif mutation == "ordinal":
        batches[-1]["step"] = 1
    elif mutation == "task":
        batches[0]["tasks"].pop()
    elif mutation == "index":
        batches[0]["tasks"][-1]["index"] = 0
    elif mutation == "validation":
        shapes.pop()
    else:
        batches[-1]["tasks"][-1]["specification"]["shape"] = {
            "dimension": [4], "boolean": [True, 2], "negative": [-1, 2],
            "many_matrices": [2, 3, 4], "too_large": [1000000, 1000000]}[mutation]
    with pytest.raises(ValueError):
        capacity.replay_accounting(batches, shapes)


def test_largest_possible_actual_endpoint_is_not_assumed_to_be_last():
    batches, shapes, config, pipeline = fixture()
    initial = capacity.replay_accounting(batches, shapes)
    batches[4]["tasks"][5]["specification"] = {"structured": {"shape": [2000, 2000]}}
    modified = capacity.replay_accounting(batches, shapes)
    assert modified["maximum_requested_endpoint_cells"] == initial["maximum_requested_endpoint_cells"] + 4000000 - 2000
    assert modified["conservative_replay_bytes"] > initial["conservative_replay_bytes"]


@pytest.mark.parametrize("mutation", ["missing_stage", "old_replay_changed", "declarations_differ"])
def test_original_declarations_and_all_later_stages_remain_bound(mutation):
    batches, shapes, config, pipeline = fixture()
    if mutation == "missing_stage":
        pipeline["stage_write_budget_bytes"].pop("confirm")
    elif mutation == "old_replay_changed":
        pipeline["stage_write_budget_bytes"]["replay"]["scratch"] += 1
    else:
        config["diagnostic_storage"]["stage_write_budget_bytes"]["confirm"]["checkpoint"] += 1
    with pytest.raises(ValueError):
        capacity.lifecycle_accounting(config, pipeline, capacity.replay_accounting(batches, shapes))


@pytest.mark.parametrize("failure", [None, "checkpoint", "root", "scratch", "omitted_volume"])
def test_full_remaining_bounds_and_original_reserves_require_actual_free_space(tmp_path, monkeypatch, failure):
    bounds = {name: 100 for name in ("checkpoint", "root", "scratch")}
    reserves = {name: 50 for name in bounds}
    volumes = {name: tmp_path / name for name in bounds}
    for directory in volumes.values():
        directory.mkdir()
    monkeypatch.setattr(capacity.shutil, "disk_usage", lambda path: SimpleNamespace(free=149 if path.name == failure else 150))
    if failure == "omitted_volume":
        volumes.pop("root")
    if failure:
        with pytest.raises((ValueError, OSError)):
            capacity.admit_space(bounds, reserves, volumes)
    else:
        assert all(row["surplus_bytes"] == 0 for row in capacity.admit_space(bounds, reserves, volumes).values())
