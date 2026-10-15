from copy import deepcopy

import pytest
import torch

import audit_single_model_targeted_decoder as audit


def test_inventory_requires_full_original_network_and_exact_decoder():
    decoder = [[f"decoder.tensor_{index}", [1], 1] for index in range(5)]
    decoder.append(["decoder.tensor_5", [11227011], 11227011])
    parameters = [["encoder.weight", [4], 4], *decoder]
    audit.check_inventories(parameters, decoder, parameters)
    with pytest.raises(ValueError, match="exactly"):
        audit.check_inventories(parameters, parameters, parameters)
    with pytest.raises(ValueError, match="exactly"):
        audit.check_inventories(parameters, decoder[::-1], parameters)
    short = deepcopy(parameters)
    short[-1][-1] -= 1
    with pytest.raises(ValueError, match="count"):
        audit.check_inventories(short, short[1:], short)


def test_initialization_requires_every_v14_grid_not_just_equal_aggregate():
    parent = {"student": {"score": 1.08}, "cases": [
        {"index": index, "student": {"rmse": 1.0}, "prediction_file": f"old/{index}", "prediction_sha256": str(index)}
        for index in range(152)]}
    current = deepcopy(parent)
    for item in current["cases"]:
        item["prediction_file"] = f'new/{item["index"]}'
    audit.check_initialization(current, parent)
    current["cases"][-1]["prediction_sha256"] = "changed"
    with pytest.raises(ValueError, match="prediction grids"):
        audit.check_initialization(current, parent)
    with pytest.raises(ValueError, match="152"):
        audit.check_initialization({**parent, "cases": parent["cases"][:-1]}, parent)


def checkpoint_fixture():
    training = audit.training
    parent = {"encoder.weight": torch.ones(2), "decoder.weight": torch.ones(3), "buffer": torch.zeros(1)}
    model = deepcopy(parent)
    model["decoder.weight"] += .1
    selected = [["decoder.weight", [3], 3]]
    config = {"input_sha256": {str(training.PARENT): "v14", str(training.original.alignment.fixed.PARENT): "v8"}}
    best = {"step": 10, "score": 1.0}
    saved = {"model": model, "step": 10, "epoch": 10, "attempted_batches": 10, "gradient_updates_performed": 9,
             "parent_training_updates": 95, "config_sha256": "config", "bank_sha256": "bank",
             "independent_confirmation_passed": False, "preflight_only": False, "decoder_only_training": True,
             "parent_checkpoint_sha256": "v14", "retention_baseline_sha256": "v8", "best_step": 10, "best_score": 1.0}
    return saved, parent, selected, 10, 9, best, config, "config", "bank"


def test_checkpoint_checks_frozen_backbone_and_separate_baseline():
    values = checkpoint_fixture()
    result = audit.check_checkpoint(*values)
    assert result["backbone_and_buffers_bitwise_unchanged"]
    assert result["changed_decoder_tensors"] == ["decoder.weight"]
    values[0]["model"]["encoder.weight"][0] += .01
    with pytest.raises(ValueError, match="Frozen"):
        audit.check_checkpoint(*values)


@pytest.mark.parametrize("field,value", [("retention_baseline_sha256", "v14"), ("parent_checkpoint_sha256", "v8"),
    ("gradient_updates_performed", 10), ("attempted_batches", 9), ("parent_training_updates", 0),
    ("decoder_only_training", False), ("preflight_only", True), ("independent_confirmation_passed", True),
    ("best_step", 0), ("best_score", .9), ("epoch", 9), ("bank_sha256", "different")])
def test_checkpoint_rejects_wrong_provenance(field, value):
    values = checkpoint_fixture()
    values[0][field] = value
    with pytest.raises(ValueError, match="provenance"):
        audit.check_checkpoint(*values)
