import pytest
import torch

from dwip_teacher_compact_artifacts import CompactCheckpointer, MODEL_FILES, compact_budget, fixed_improves
from run_dwip_budgeted_queue_v1 import check_storage, RESERVE_BYTES


def test_compact_budget_counts_two_model_files_and_optimizer_temporary(tmp_path):
    (tmp_path / "best_validation_checkpoint.pth").write_bytes(b"model" * 100)
    (tmp_path / "latest_checkpoint.pth").write_bytes(b"optimizer" * 200)
    budget = compact_budget(tmp_path)
    assert budget["model_only_files"] == list(MODEL_FILES)
    expected = 2 * budget["model_file_ceiling_bytes"] + 2 * budget["latest_file_ceiling_bytes"] + budget["other_artifacts_bytes"]
    assert budget["peak_additional_bytes"] == expected
    assert budget["required_free_bytes_at_launch"] == expected + RESERVE_BYTES
    with pytest.raises(RuntimeError, match="storage guard"):
        check_storage(budget, expected + RESERVE_BYTES - 1, launch=True)


def test_compact_checkpoint_write_preserves_rng(tmp_path):
    calls = []
    writer = CompactCheckpointer(tmp_path, {"model_file_ceiling_bytes": 10000}, lambda: calls.append(True))
    rng = torch.get_rng_state()
    payload = {"model": {"weight": torch.ones(3)}}
    writer.save(payload, MODEL_FILES[0])
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(calls) == 2 and not list(tmp_path.glob("*.tmp"))
    assert torch.equal(torch.load(tmp_path / MODEL_FILES[0], weights_only=True)["model"]["weight"], torch.ones(3))


def test_oversize_temporary_is_preserved_without_replacing_previous_checkpoint(tmp_path):
    target = tmp_path / MODEL_FILES[0]
    target.write_bytes(b"previous")
    writer = CompactCheckpointer(tmp_path, {"model_file_ceiling_bytes": 1}, lambda: None)
    with pytest.raises(RuntimeError, match="ceiling"):
        writer.save({"model": torch.ones(10)}, MODEL_FILES[0])
    assert target.read_bytes() == b"previous"
    assert target.with_suffix(".pth.tmp").is_file()


@pytest.mark.parametrize("name", ("best_checkpoint.pth", "checkpoint_4000.pth", "../checkpoint_64000.pth"))
def test_no_unbudgeted_checkpoint_is_written(tmp_path, name):
    writer = CompactCheckpointer(tmp_path, {}, lambda: None)
    with pytest.raises(ValueError, match="Unregistered"):
        writer.save({}, name)
    assert not list(tmp_path.iterdir())


def test_existing_temporary_is_never_overwritten(tmp_path):
    temporary = tmp_path / f"{MODEL_FILES[0]}.tmp"
    temporary.write_bytes(b"failed evidence")
    writer = CompactCheckpointer(tmp_path, {}, lambda: None)
    with pytest.raises(ValueError, match="Preserve"):
        writer.save({}, MODEL_FILES[0])
    assert temporary.read_bytes() == b"failed evidence"


@pytest.mark.parametrize("score,expected", ((.9, True), (1., False), (1.1, False)))
def test_fixed_selection_is_strict_and_ignores_benchmarks(score, expected):
    assert fixed_improves({"macro_rmse": score, "mteb": 999.}, 1.) is expected


@pytest.mark.parametrize("score", (float("nan"), float("inf"), -1.))
def test_nonfinite_or_negative_selector_fails(score):
    with pytest.raises(ValueError):
        fixed_improves({"macro_rmse": score}, 1.)
