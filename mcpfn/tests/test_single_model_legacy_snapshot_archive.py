from copy import deepcopy

import pytest

import archive_single_model_legacy_snapshots as legacy


@pytest.mark.parametrize("name,epoch", [("checkpoint_4000.pth", 4000), ("checkpoint_phase1_24000.pth", 24000),
                                      ("latest_checkpoint.pth", 24000)])
def test_only_declared_snapshot_identities(name, epoch):
    assert legacy.snapshot_epoch(name, 24000) == epoch


@pytest.mark.parametrize("name", ["best_checkpoint.pth", "best_validation_checkpoint.pth", "checkpoint_24001.pth",
                                  "checkpoint_0.pth", "checkpoint_unknown.pth", "../../checkpoint_4000.pth"])
def test_selected_or_unknown_checkpoints_never_move(name):
    with pytest.raises(ValueError):
        legacy.snapshot_epoch(name, 24000)


def queue_fixture():
    name = legacy.NAMES[0]
    terminal = {"state": "complete", "active": "complete", "completed": ["first", "second"]}
    results = [{"tag": tag, "horizon": {"work_dir": str(legacy.WORK / f"tabimpute-{name}-{tag}")}}
               for tag in terminal["completed"]]
    return name, terminal, results


def test_completed_queue_and_exact_work_directory_bind_ownership():
    name, terminal, results = queue_fixture()
    assert legacy.queue_runs(name, terminal, results) == [legacy.WORK / f"tabimpute-{name}-{tag}" for tag in ("first", "second")]


@pytest.mark.parametrize("mutation", ["state", "active", "missing", "duplicate", "outside"])
def test_incomplete_or_ambiguous_queues_rejected(mutation):
    name, terminal, results = queue_fixture()
    if mutation in ("state", "active"):
        terminal[mutation] = "training"
    elif mutation == "missing":
        results.pop()
    elif mutation == "duplicate":
        results[1] = deepcopy(results[0])
    else:
        results[0]["horizon"]["work_dir"] = "/tmp/unrelated"
    with pytest.raises(ValueError):
        legacy.queue_runs(name, terminal, results)


def test_reused_control_is_not_double_counted():
    name, terminal, results = queue_fixture()
    results[0]["horizon"]["work_dir"] = str(legacy.WORK / "tabimpute-rowcol_controlled_20260906-mse_broad_support")
    assert legacy.queue_runs(name, terminal, results) == [legacy.WORK / f"tabimpute-{name}-second"]


def test_live_writer_detection_uses_actual_argv_tokens():
    assert legacy.writer_arguments(["python", "-u", "/repo/train_direct_pure_rabel_curriculum.py"])
    assert legacy.writer_arguments(["python", "scripts/run_spectral_lr_followups.py"])
    assert not legacy.writer_arguments(["bash", "-c", "echo train_direct_pure_rabel_curriculum.py"])
    assert not legacy.writer_arguments(["python", "scripts/archive_single_model_legacy_snapshots.py"])


def test_root_only_capacity_policy_keeps_original_checkpoint_volume_floor():
    assert legacy.MINIMUM_ROOT_FREE == 3 * 1024**3 and legacy.TARGET_ROOT_FREE == 24 * 1024**3
    assert legacy.archive.MINIMUM_FREE == 21 * 1024**3
    compressor = legacy.compressor()
    assert compressor.__globals__["ROOT"] == legacy.ROOT
    assert compressor.__globals__["MINIMUM_FREE"] == legacy.MINIMUM_ROOT_FREE
    assert "torch_mmap_readable" in legacy.COMPRESSOR_SOURCE
    assert "sha256(archived)" in legacy.COMPRESSOR_SOURCE
