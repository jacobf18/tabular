import pytest
import torch

import check_single_model_decoder_finite_steps as finite
from single_model_parameter_subspace import inventory_ranges


def test_process_local_finite_recipe_restores_originals_even_after_failure():
    original = {name: getattr(finite.original, name) for name in ("ROLES", "POLICY", "role_parameters", "verify_registration")}
    with pytest.raises(RuntimeError):
        with finite.runtime():
            assert finite.original.ROLES == ("before", "decoder", "output_layer")
            assert finite.original.role_parameters is finite.role_parameters
            raise RuntimeError("test")
    for name, value in original.items():
        assert getattr(finite.original, name) is value


def test_decoder_replay_changes_only_audited_coordinates(monkeypatch, tmp_path):
    before = torch.tensor([1., 2., 3., 4.])
    ranges, total = inventory_ranges([["encoder.weight", [2], 2], ["decoder.layers.4.bias", [2], 2]], "output_layer")
    path = tmp_path / "subset.pt"
    torch.save({"ranges": ranges, "before": before[2:].clone(), "candidate": torch.tensor([5., 6.])}, path)
    record = {"decision": {"admissible": True}, "artifact": str(path), "total_parameters": total, "actual_products": [1.]}
    monkeypatch.setattr(finite.probe, "read", lambda path: {"scopes": {"output_layer": record}})
    candidate, products = finite.role_parameters({}, {"before": before}, "output_layer")
    assert torch.equal(candidate, torch.tensor([1., 2., 5., 6.]))
    assert products == [1.]
    changed = before.clone()
    changed[-1] += 1
    with pytest.raises(ValueError, match="identical"):
        finite.role_parameters({}, {"before": changed}, "output_layer")


def test_live_predecessor_cannot_be_preempted(monkeypatch, tmp_path):
    monkeypatch.setattr(finite, "verify", lambda output: None)
    monkeypatch.setattr(finite.probe, "read", lambda path: {"state": "registered"} if path.parent == tmp_path else {"state": "replay"})
    with pytest.raises(ValueError, match="preempt"):
        finite.run(tmp_path)
