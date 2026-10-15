from copy import deepcopy

import pytest

import summarize_single_model_family_projection_boundary as boundary


def test_incomplete_future_log_row_does_not_invalidate_completed_prefix(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3')
    assert boundary.read_prefix(path, 2) == [{"step": 1}, {"step": 2}]
    with pytest.raises(ValueError, match="every complete"):
        boundary.read_prefix(path, 3)


@pytest.mark.parametrize("steps", [[1, 1, 2], [2, 1], [1, 3], []])
def test_missing_duplicate_or_reordered_update_is_not_hidden(tmp_path, steps):
    path = tmp_path / "training.jsonl"
    path.write_text("".join(f'{{"step": {step}}}\n' for step in steps))
    with pytest.raises(ValueError):
        boundary.read_prefix(path, 2)


def decisions():
    comparisons = [{"group": f"group_{index}", "metric": "common_nrmse", "initialization": 1., "candidate": 1., "retained": True} for index in range(22)]
    return {name: {"comparisons": deepcopy(comparisons)} for name in boundary.training.ARMS}


def test_table_preserves_every_guard_and_signed_change():
    source = decisions()
    source["projected"]["comparisons"][0].update(candidate=1.01, retained=False)
    table = boundary.decisions_table(source)
    assert len(table) == 22
    assert not table[0]["projected"]["retained"]
    assert table[0]["projected"]["change_percent"] == pytest.approx(1.)
    assert table[0]["control"]["change_percent"] == 0


@pytest.mark.parametrize("change", ["omit", "baseline", "group", "metric", "tolerance"])
def test_guard_identity_and_zero_tolerance_cannot_change(change):
    source = decisions()
    if change == "omit":
        source["control"]["comparisons"].pop()
    elif change == "baseline":
        source["projected"]["comparisons"][0]["initialization"] = 2.
    elif change in ("group", "metric"):
        source["projected"]["comparisons"][0][change] = "different"
    else:
        source["projected"]["comparisons"][0]["candidate"] += 1e-10
    with pytest.raises(ValueError):
        boundary.decisions_table(source)


def test_cpu_only_requirement_precedes_any_artifact_access(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        boundary.summarize(25, tmp_path)


def test_unregistered_boundary_cannot_be_used_for_early_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="registered completed"):
        boundary.summarize(10, tmp_path)
