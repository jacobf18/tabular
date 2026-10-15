from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import single_model_scratch_diagnostics as diagnostics


@pytest.fixture
def planned(tmp_path, monkeypatch):
    run, scratch = tmp_path / "checkpoints/trial", tmp_path / "scratch/trial"
    run.mkdir(parents=True)
    scratch.mkdir(parents=True)
    plan = diagnostics.layout(run, scratch, (0, 16, 32, 64, 96))
    monkeypatch.setattr(diagnostics, "require_filesystems", lambda current: None)
    return plan


def test_only_new_diagnostics_move_and_ordinary_checkpoints_stay(planned):
    run = Path(planned["run"])
    selected, latest = run / "selected_checkpoint.pth", run / "latest_checkpoint.pth"
    selected.write_bytes(b"selected ordinary model")
    latest.write_bytes(b"latest ordinary optimizer checkpoint")
    diagnostics.prepare(planned, "training")
    assert len(planned["directories"]["training"]) == 5
    assert all(not Path(record["canonical"]).exists() for record in planned["directories"]["qualification"])
    diagnostics.prepare(planned, "qualification")
    for phase in ("training", "qualification"):
        diagnostics.verify(planned, phase)
        for record in planned["directories"][phase]:
            path = Path(record["canonical"]) / "prediction.npz"
            path.write_bytes(b"new synthetic diagnostic")
            assert (Path(record["target"]) / path.name).read_bytes() == path.read_bytes()
        with pytest.raises(FileExistsError, match="never retry or relocate"):
            diagnostics.prepare(planned, phase)
    assert not selected.is_symlink() and selected.read_bytes() == b"selected ordinary model"
    assert not latest.is_symlink() and latest.read_bytes() == b"latest ordinary optimizer checkpoint"
    assert planned["scratch_is_provider_persistent"] is False


@pytest.mark.parametrize("kind", ["file", "directory", "broken_link", "target"])
def test_existing_evidence_is_never_replaced(planned, kind):
    record = planned["directories"]["training"][-1]
    path = Path(record["target"] if kind == "target" else record["canonical"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        path.write_bytes(b"earlier evidence")
    elif kind == "broken_link":
        path.symlink_to(path.parent / "missing")
    else:
        path.mkdir()
    with pytest.raises(FileExistsError):
        diagnostics.prepare(planned, "training")
    assert not Path(planned["directories"]["training"][0]["canonical"]).exists()


def test_changed_scope_or_target_or_budgets_are_rejected(planned):
    altered = deepcopy(planned)
    altered["directories"]["training"][0]["canonical"] = str(Path(planned["run"]) / "selected_checkpoint.pth")
    with pytest.raises(ValueError, match="scope"):
        diagnostics.prepare(altered, "training")
    altered = deepcopy(planned)
    altered["minimum_free_bytes"]["checkpoint"] -= 1
    with pytest.raises(ValueError, match="reserves"):
        diagnostics.prepare(altered, "training")
    diagnostics.prepare(planned, "training")
    path = Path(planned["directories"]["training"][0]["canonical"])
    path.unlink()
    path.symlink_to(Path(planned["scratch"]))
    with pytest.raises(ValueError, match="canonical"):
        diagnostics.verify(planned, "training")


def test_prepared_validation_directory_is_still_one_shot(planned):
    diagnostics.prepare(planned, "training")
    directory = Path(planned["directories"]["training"][0]["canonical"])
    diagnostics.require_empty_validation(planned, directory)
    with pytest.raises(ValueError, match="declared development"):
        diagnostics.require_empty_validation(planned, Path(planned["run"]))
    (directory / "student_predictions_000").mkdir()
    with pytest.raises(FileExistsError, match="Never reevaluate"):
        diagnostics.require_empty_validation(planned, directory)


@pytest.mark.parametrize("boundaries", [[], [1, 2], [0, 0], [0, 2, 1], [0, True], [0, -1]])
def test_layout_requires_ordered_original_boundaries(tmp_path, boundaries):
    with pytest.raises(ValueError, match="boundaries"):
        diagnostics.layout(tmp_path / "run", tmp_path / "scratch", boundaries)


def test_layout_disallows_nested_or_relative_storage_roots(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        diagnostics.layout(Path("relative"), tmp_path / "scratch", (0, 16))
    with pytest.raises(ValueError, match="distinct"):
        diagnostics.layout(tmp_path, tmp_path / "scratch", (0, 16))


@pytest.mark.parametrize("stage", list(diagnostics.WRITE_BUDGETS))
@pytest.mark.parametrize("volume", list(diagnostics.RESERVES))
def test_each_stage_accounts_for_its_actual_volume_without_spending_reserves(monkeypatch, stage, volume):
    volumes = {name: Path("/") / name for name in diagnostics.RESERVES}
    capacity = {path: diagnostics.RESERVES[name] + diagnostics.WRITE_BUDGETS[stage][name]
                for name, path in volumes.items()}
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda path: SimpleNamespace(free=capacity[path]))
    diagnostics.require_capacity(stage, volumes)
    capacity[volumes[volume]] -= 1
    with pytest.raises(OSError, match="Preserve"):
        diagnostics.require_capacity(stage, volumes)


def test_publication_remains_budgeted_on_checkpoint_storage():
    assert diagnostics.WRITE_BUDGETS["audit-confirmation"]["checkpoint"] >= 312096899
    assert diagnostics.WRITE_BUDGETS["training"]["checkpoint"] >= 2 * 468106547 + 312096899
    assert diagnostics.WRITE_BUDGETS["confirm"]["scratch"] == 2560 * 1024**2
    assert diagnostics.RESERVES == {"checkpoint": 18 * 1024**3, "root": 6 * 1024**3, "scratch": 50 * 1024**3}
    with pytest.raises(ValueError, match="per-volume"):
        diagnostics.require_capacity("undeclared")
    with pytest.raises(ValueError, match="omit"):
        diagnostics.require_capacity("training", {"checkpoint": Path("/")})
    with pytest.raises(ValueError, match="omit"):
        diagnostics.require_capacity("training", {})
