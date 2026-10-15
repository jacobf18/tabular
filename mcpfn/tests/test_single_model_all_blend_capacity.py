from types import FunctionType

import pytest
import torch

import relocate_single_model_all_blend_capacity as relocation
import test_single_model_uniform_v8_capacity as capacity_test_blueprint


for name in ("test_only_complete_finite_probe_vectors_are_preserved",
        "test_copy_fsync_and_atomic_link_preserve_exact_original_serialization",
        "test_existing_preservation_is_not_restarted"):
    blueprint = getattr(capacity_test_blueprint, name)
    inherited = FunctionType(blueprint.__code__, {**blueprint.__globals__, "relocation": relocation},
        name, blueprint.__defaults__, blueprint.__closure__)
    inherited.__dict__.update(blueprint.__dict__)
    globals()[name] = inherited


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("mutation", [None, "shape", "dtype", "nan", "margin", "products", "norm", "adverse", "extra"])
def test_nonpromotable_point_proposals_keep_original_outcomes_and_full_vectors(accepted, mutation):
    payload = {"margin": .0001, "accepted": accepted, "parameters": torch.arange(5, dtype=torch.float32),
        "solution": {}, "actual_products": [0.] * 14, "actual_norm": 0., "adverse_tasks": 3}
    if mutation == "shape":
        payload["parameters"] = payload["parameters"][:-1]
    elif mutation == "dtype":
        payload["parameters"] = payload["parameters"].double()
    elif mutation == "nan":
        payload["parameters"][0] = float("nan")
    elif mutation == "margin":
        payload["margin"] = .1
    elif mutation == "products":
        payload["actual_products"].pop()
    elif mutation == "norm":
        payload["actual_norm"] = float("nan")
    elif mutation == "adverse":
        payload["adverse_tasks"] = -1
    elif mutation == "extra":
        payload["model"] = {}
    if mutation is None:
        relocation.validate_payload(payload, "point_proposal", full_parameters=5)
    else:
        with pytest.raises(ValueError):
            relocation.validate_payload(payload, "point_proposal", full_parameters=5)


@pytest.mark.parametrize("mutation", [None, "step", "updates", "best_step", "optimizer", "selection", "same_model", "selected_link"])
def test_only_the_exact_unselected_v44_tail_can_move(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "uniform_v8_retention_v44"
    directory.mkdir()
    config = directory / "config.json"
    config.write_text("{}")
    selected = directory / "selected_checkpoint.pth"
    torch.save({"step": 50, "model": {"weight": torch.ones(5)}}, selected)
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "SELECTED", selected)
    monkeypatch.setattr(relocation, "SELECTED_SHA256", relocation.sha256(selected))
    monkeypatch.setattr(relocation, "read", lambda path: {"best": {"score": 1.}})
    payload = {"step": 64, "attempted_batches": 64, "gradient_updates_performed": 57, "best_step": 50,
        "best_score": 1., "config_sha256": relocation.sha256(config), "independent_confirmation_passed": False,
        "preflight_only": False, "full_network_training": True, "model": {"weight": torch.zeros(5)},
        "optimizer": {"state": {}}}
    if mutation == "step":
        payload["step"] = 50
    elif mutation == "updates":
        payload["gradient_updates_performed"] = 56
    elif mutation == "best_step":
        payload["best_step"] = 25
    elif mutation == "optimizer":
        payload["optimizer"] = {}
    elif mutation == "selection":
        payload["independent_confirmation_passed"] = True
    elif mutation == "same_model":
        payload["model"]["weight"] = torch.ones(5)
    elif mutation == "selected_link":
        target = directory / "moved.pth"
        selected.rename(target)
        selected.symlink_to(target)
    if mutation is None:
        relocation.validate_payload(payload, "tail")
    else:
        with pytest.raises(ValueError):
            relocation.validate_payload(payload, "tail")


def test_declared_scope_preserves_every_selected_checkpoint_and_original_reserve():
    assert len(relocation.TARGETS) == 6
    assert list(relocation.TARGETS.values()).count("point_proposal") == 4
    assert all("selected_checkpoint" not in name and "best_validation_checkpoint" not in name for name in relocation.TARGETS)
    assert relocation.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    assert relocation.PROBE_CONFIG_SHA256 == "18ef8a6008c355889d6b622d4c2e04010292e239a2127d20772016bf95a39c0f"
    assert relocation.SELECTED.name == "selected_checkpoint.pth"
