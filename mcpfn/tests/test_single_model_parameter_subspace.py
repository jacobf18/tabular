import pytest
import torch

from probe_single_model_decoder_subspaces import geometry
from single_model_parameter_subspace import extract, inventory_ranges, replace


def test_subspace_replacement_leaves_every_other_coordinate_bitwise_unchanged():
    inventory = [["encoder.weight", [3], 3], ["decoder.layers.0.weight", [2], 2], ["decoder.layers.4.bias", [1], 1]]
    ranges, elements = inventory_ranges(inventory, "decoder")
    original = torch.arange(6, dtype=torch.float32)
    selected = extract(original, ranges, elements)
    assert torch.equal(selected, torch.tensor([3., 4., 5.]))
    candidate = replace(original, selected + 10, ranges, elements)
    assert torch.equal(candidate[:3], original[:3])
    assert torch.equal(candidate[3:], torch.tensor([13., 14., 15.]))
    output_ranges, _ = inventory_ranges(inventory, "output_layer")
    assert torch.equal(extract(original, output_ranges, elements), torch.tensor([5.]))


@pytest.mark.parametrize("inventory,scope", [([["encoder.weight", [2], 2]], "decoder"),
    ([["decoder.weight", [2], 3]], "decoder"),
    ([["decoder.bias", [1], 1], ["decoder.bias", [1], 1]], "decoder"),
    ([["decoder.weight", [2], 2]], "all")])
def test_missing_or_changed_inventory_is_rejected(inventory, scope):
    with pytest.raises(ValueError):
        inventory_ranges(inventory, scope)


def test_restricted_projection_preserves_all_declared_gradient_directions():
    gradients = torch.tensor([[1., 0.], [0., 1.]])
    before = torch.tensor([1., 1.])
    tentative = torch.tensor([.99, 1.001])
    record, candidate = geometry(gradients, before, tentative)
    assert record["adverse_before"] == 1
    assert record["feasible_projection"] and record["decision"]["admissible"]
    assert candidate.dtype == torch.float32
    assert all(value > 0 for value in record["actual_products"])


def test_conflicting_restricted_gradients_do_not_fabricate_feasibility():
    gradients = torch.tensor([[1.], [-1.]])
    record, candidate = geometry(gradients, torch.tensor([1.]), torch.tensor([.99]))
    assert not record["feasible_projection"]
    assert candidate is None


def test_zero_displacement_never_counts_as_an_admissible_update():
    record, candidate = geometry(torch.tensor([[1.]]), torch.tensor([1.]), torch.tensor([1.]))
    assert not record["feasible_projection"]
    assert record["error"] == "zero_restricted_adam_proposal"
    assert candidate is None
