import pytest
import torch

from dwip_distribution_compact_artifacts import (
    BEST, LATEST, MIB, DistributionCheckpointer, distribution_budget, measured_budget,
)
from run_dwip_budgeted_queue_v1 import check_storage, RESERVE_BYTES


def test_budget_counts_one_model_full_optimizer_largest_temporary_and_other_outputs():
    budget = distribution_budget(312096259, 936287343, 15 * MIB)
    assert budget["model_only_files"] == [BEST]
    assert budget["terminal_checkpoint"] == budget["rolling_optimizer_file"] == LATEST
    assert budget["other_artifacts_bytes"] == 128 * MIB
    assert budget["peak_additional_bytes"] == budget["model_file_ceiling_bytes"] + 2 * budget["latest_file_ceiling_bytes"] + 128 * MIB
    assert budget["model_file_ceiling_bytes"] >= 312096259 * 1.05
    assert budget["latest_file_ceiling_bytes"] >= 936287343 * 1.05
    assert budget["reserve_bytes"] == RESERVE_BYTES
    check_storage(budget, budget["required_free_bytes_at_launch"], launch=True)
    with pytest.raises(RuntimeError, match="storage guard"):
        check_storage(budget, budget["required_free_bytes_at_launch"] - 1, launch=True)


def test_other_artifacts_scale_up_without_weakening_reserve():
    budget = distribution_budget(MIB, 3 * MIB, 100 * MIB)
    assert budget["other_artifacts_bytes"] == 500 * MIB
    assert budget["reserve_bytes"] == 20 * 1024**3


@pytest.mark.parametrize("values", ((0, 1, 1), (1, 0, 1), (1, 1, -1), (True, 2, 1), (1., 3, 1)))
def test_invalid_budget_measurements_fail_closed(values):
    with pytest.raises(ValueError):
        distribution_budget(*values)


def test_reference_measurement_preserves_files_and_omits_only_known_checkpoint_bytes(tmp_path):
    root, work = tmp_path / "root", tmp_path / "work"
    root.mkdir()
    work.mkdir()
    for name, count in ((BEST, 100), (LATEST, 300), ("checkpoint_64000.pth", 110), ("metrics.json", 10)):
        (work / name).write_bytes(b"a" * count)
    (root / "results.json").write_bytes(b"x" * 20)
    (root / "shared.pth").symlink_to(work / BEST)
    budget = measured_budget(root, work, {"model": 105, "optimizer": 310})
    assert budget["reference_model_bytes"] == 110
    assert budget["reference_optimizer_bytes"] == 310
    assert budget["observed_other_artifacts_bytes"] == 30
    assert (work / BEST).stat().st_size == 100
    (work / f"{LATEST}.tmp").write_bytes(b"failed")
    with pytest.raises(ValueError, match="failed temporaries"):
        measured_budget(root, work, {"model": 105, "optimizer": 310})


@pytest.mark.parametrize("name", (BEST, LATEST))
def test_bounded_checkpoint_write_preserves_rng_and_payload(tmp_path, name):
    calls = []
    writer = DistributionCheckpointer(tmp_path, {"model_file_ceiling_bytes": 10000, "latest_file_ceiling_bytes": 10000}, lambda: calls.append(True))
    payload = {"model": {"weight": torch.ones(3)}}
    if name == LATEST:
        payload["optimizer"] = {"state": {}}
    before = torch.get_rng_state()
    writer.save(payload, name)
    assert torch.equal(before, torch.get_rng_state())
    assert len(calls) == 3 and not list(tmp_path.glob("*.tmp"))
    assert torch.equal(torch.load(tmp_path / name, weights_only=True)["model"]["weight"], torch.ones(3))


def test_oversize_is_rejected_before_any_temporary_is_written(tmp_path):
    (tmp_path / BEST).write_bytes(b"previous")
    writer = DistributionCheckpointer(tmp_path, {"model_file_ceiling_bytes": 1}, lambda: None)
    with pytest.raises(RuntimeError, match="no artifact was written"):
        writer.save({"model": torch.ones(100)}, BEST)
    assert (tmp_path / BEST).read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("name", ("checkpoint_64000.pth", "checkpoint_2000.pth", "../latest_checkpoint.pth"))
def test_redundant_or_unregistered_checkpoints_are_rejected(tmp_path, name):
    writer = DistributionCheckpointer(tmp_path, {}, lambda: None)
    with pytest.raises(ValueError, match="Only best-model"):
        writer.save({}, name)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,payload", ((BEST, {"optimizer": {}}), (LATEST, {"model": {}})))
def test_checkpoint_roles_cannot_be_confused(tmp_path, name, payload):
    writer = DistributionCheckpointer(tmp_path, {}, lambda: None)
    with pytest.raises(ValueError, match="role"):
        writer.save(payload, name)


def test_failed_write_preserves_temporary_and_previous_checkpoint(tmp_path, monkeypatch):
    def fail_copy(source, target, length):
        target.write(b"partial")
        raise OSError("interrupted")
    monkeypatch.setattr("dwip_distribution_compact_artifacts.shutil.copyfileobj", fail_copy)
    (tmp_path / BEST).write_bytes(b"previous")
    writer = DistributionCheckpointer(tmp_path, {"model_file_ceiling_bytes": 10000}, lambda: None)
    with pytest.raises(OSError, match="interrupted"):
        writer.save({"model": torch.ones(3)}, BEST)
    assert (tmp_path / BEST).read_bytes() == b"previous"
    assert (tmp_path / f"{BEST}.tmp").read_bytes() == b"partial"
    with pytest.raises(ValueError, match="Preserve"):
        writer.save({"model": torch.ones(3)}, BEST)
