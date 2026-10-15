from types import SimpleNamespace

import pytest
import torch

import average_single_model_training_tail as averaging
from average_single_model_training_tail import WeightAverage, load_stable, require_live
from recsys_evaluation_common import sha256, write_json


def test_online_average_materializes_one_original_dtype_state_without_mutating_inputs():
    average = WeightAverage()
    states = [{"weight": torch.tensor([value, value + 1.]), "count": torch.tensor(3)} for value in (1., 4., 10.)]
    for state in states:
        average.add(state)
    expected = torch.stack([state["weight"] for state in states]).mean(0)
    result = average.state()
    torch.testing.assert_close(result["weight"], expected)
    assert result["weight"].dtype == torch.float32 and result["count"].dtype == torch.int64
    assert average.count == 3 and list(result) == ["weight", "count"]
    result["weight"].zero_()
    torch.testing.assert_close(average.state()["weight"], expected)
    assert states[0]["weight"].tolist() == [1., 2.]


@pytest.mark.parametrize("changed", [
    {"weight": torch.ones(3), "count": torch.tensor(1)},
    {"weight": torch.ones(2, dtype=torch.float64), "count": torch.tensor(1)},
    {"weight": torch.ones(2), "count": torch.tensor(2)},
    {"weight": torch.tensor([float("nan"), 1.]), "count": torch.tensor(1)},
])
def test_incompatible_states_fail_before_changing_the_average(changed):
    average = WeightAverage()
    average.add({"weight": torch.ones(2), "count": torch.tensor(1)})
    with pytest.raises(ValueError):
        average.add(changed)
    assert average.count == 1
    torch.testing.assert_close(average.state()["weight"], torch.ones(2))


def test_stable_reader_hashes_exact_bytes_loaded(tmp_path):
    import hashlib

    path = tmp_path / "checkpoint.pth"
    torch.save({"model": {"weight": torch.arange(3.)}, "step": 300}, path)
    payload, digest = load_stable(path)
    assert payload["step"] == 300 and digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_producer_guard_rejects_terminal_failure_and_missing_process():
    require_live({"state": "training_complete"})
    with pytest.raises(RuntimeError, match="stopped"):
        require_live({"state": "failed"})
    with pytest.raises(RuntimeError, match="missing or terminal"):
        require_live({"state": "training", "pid": 2**30})


def test_end_to_end_average_binds_every_update_and_contains_no_optimizer(tmp_path, monkeypatch):
    producer, output = tmp_path / "producer", tmp_path / "average"
    producer.mkdir()
    output.mkdir()
    write_json(producer / "config.json", {"steps": 310})
    fingerprint = sha256(producer / "config.json")
    config = {"producer": str(producer), "steps": [300, 310], "producer_config_sha256": fingerprint,
              "source_sha256": {}, "producer_inputs": {}, "producer_sources": {}, "producer_teacher_configs": {},
              "step_semantics": "Last contributing producer update"}
    write_json(output / "config.json", config)
    statuses = iter([{"state": "training", "step": 300}, {"state": "training_complete", "step": 310},
                     {"state": "training_complete", "step": 310}])
    monkeypatch.setattr(averaging, "read", lambda path: next(statuses))
    monkeypatch.setattr(averaging, "require_live", lambda status: None)
    payloads = iter([({"model": {"weight": torch.tensor([2.])}, "step": 300, "config_sha256": fingerprint}, "first"),
                     ({"model": {"weight": torch.tensor([4.])}, "step": 310, "config_sha256": fingerprint}, "second")])
    monkeypatch.setattr(averaging, "load_stable", lambda path: next(payloads))
    monkeypatch.setattr(averaging.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    averaging.execute(config, output)
    result = torch.load(output / "averaged_checkpoint.pth", weights_only=True)
    assert result["model"]["weight"].item() == 3.
    assert result["source_updates"] == [300, 310] and result["step"] == 310
    assert result["gradient_updates_performed_here"] == 0 and "optimizer" not in result
    assert result["receipts_sha256"] == sha256(output / "receipts.json")


def test_a_missing_required_checkpoint_aborts_instead_of_shrinking_the_window(tmp_path, monkeypatch):
    config = {"producer": str(tmp_path), "steps": [300, 310], "producer_config_sha256": "config"}
    monkeypatch.setattr(averaging, "read", lambda path: {"state": "training", "step": 310})
    monkeypatch.setattr(averaging, "require_live", lambda status: None)
    monkeypatch.setattr(averaging, "load_stable", lambda path: ({"step": 310}, "later"))
    with pytest.raises(ValueError, match="Missed required update 300"):
        averaging.execute(config, tmp_path)
    assert not (tmp_path / "averaged_checkpoint.pth").exists()
