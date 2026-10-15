from types import SimpleNamespace

import pytest
import torch

import dwip_bpr_experiment as experiment


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("step", [0, 50, 200])
def test_validation_details_only_change_the_output_directory(tmp_path, monkeypatch, enabled, step):
    args = SimpleNamespace() if enabled is None else SimpleNamespace(save_validation_details=enabled)
    imputer = object()
    metrics = {"synthetic_fixture": {"MAE": .5, "RMSE": .6, "NDCG@10": .1, "Recall@10": .2}}
    calls = []
    original_rng = torch.get_rng_state()

    def evaluate(candidate, partition="validation", output=None, audit_cornac=False):
        assert candidate is imputer
        assert partition == "validation"
        assert not torch.is_grad_enabled()
        assert not audit_cornac
        calls.append(output)
        return metrics

    monkeypatch.setattr(experiment, "evaluate_public", evaluate)
    with torch.enable_grad():
        actual = experiment.training_validation(imputer, tmp_path, step, args)
        assert torch.is_grad_enabled()
    assert actual is metrics
    assert calls == [tmp_path / f"validation_details_{step}" if enabled else None]
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert list(tmp_path.iterdir()) == []
