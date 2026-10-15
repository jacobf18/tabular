"""Validate a prospectively declared allocator-only change without weakening scientific recipe checks."""

from copy import deepcopy

from single_model_registered_adaptation import replace_once


def expected_base(original):
    if original["memory_fraction"] != .8:
        raise ValueError("Bind the original.8 allocation setting before deriving the.9 recipe")
    updated = deepcopy(original)
    updated["memory_fraction"] = .9
    return updated


def validate_base(actual, original, scientific_validator):
    if actual != expected_base(original):
        raise ValueError("Only the prospectively registered allocator fraction may differ from the original base")
    scientific_validator(deepcopy(original))


def adapt_verifier(source):
    source = replace_once(source,
        'config["base"] != read(original.alignment.fixed.PRODUCER / "config.json")',
        'config["base"] != cap90_recipe.expected_base(read(original.alignment.fixed.PRODUCER / "config.json"))')
    return replace_once(source, 'original.alignment.fixed.validate_recipe(config["base"])',
        'cap90_recipe.validate_base(config["base"], read(original.alignment.fixed.PRODUCER / "config.json"), '
        'original.alignment.fixed.validate_recipe)')
