from copy import deepcopy

import pytest

import single_model_full_network_cap90_recipe as recipe
import train_single_matrixformer_full_network_cap90 as failed


def test_exact_allocator_only_change_does_not_mutate_either_configuration():
    original = {"memory_fraction": .8, "seed": 91, "objective": {"cdf_weight": .05}}
    before = deepcopy(original)
    actual = recipe.expected_base(original)
    calls = []
    recipe.validate_base(actual, original, calls.append)
    assert calls == [original]
    assert original == before and actual["memory_fraction"] == .9
    calls[0]["objective"]["cdf_weight"] = 0.
    assert actual["objective"]["cdf_weight"] == .05 and original == before


@pytest.mark.parametrize("mutation", ["memory_fraction", "seed", "objective", "added", "removed"])
def test_no_other_setting_can_drift(mutation):
    original = {"memory_fraction": .8, "seed": 91, "objective": {"cdf_weight": .05}}
    actual = recipe.expected_base(original)
    if mutation == "removed":
        del actual["objective"]
    elif mutation == "objective":
        actual["objective"]["cdf_weight"] = .1
    else:
        actual[mutation] = .95 if mutation == "memory_fraction" else 92
    calls = []
    with pytest.raises(ValueError, match="allocator fraction"):
        recipe.validate_base(actual, original, calls.append)
    assert calls == []


def test_unchanged_original_scientific_validator_still_rejects():
    original = {"memory_fraction": .8}
    def reject(config):
        raise ValueError("original scientific gate")
    with pytest.raises(ValueError, match="original scientific gate"):
        recipe.validate_base(recipe.expected_base(original), original, reject)


def test_adaptation_replaces_only_two_explicit_base_checks():
    updated = recipe.adapt_verifier(failed.ADAPTED_SOURCE)
    restored = updated.replace(
        'config["base"] != cap90_recipe.expected_base(read(original.alignment.fixed.PRODUCER / "config.json"))',
        'config["base"] != read(original.alignment.fixed.PRODUCER / "config.json")').replace(
        'cap90_recipe.validate_base(config["base"], read(original.alignment.fixed.PRODUCER / "config.json"), original.alignment.fixed.validate_recipe)',
        'original.alignment.fixed.validate_recipe(config["base"])')
    assert restored == failed.ADAPTED_SOURCE
    with pytest.raises(ValueError):
        recipe.adapt_verifier(updated)
