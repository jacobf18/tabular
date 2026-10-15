"""Unchanged reserved284-case independent gate for the V11 rehearsal experiment."""

import confirm_single_matrixformer_structured_mixture as original
from train_single_matrixformer_family_rehearsal import policy_runtime, validate_recipe


def main():
    previous = original.validate_recipe
    original.validate_recipe = validate_recipe
    try:
        with policy_runtime():
            original.main()
    finally:
        original.validate_recipe = previous


if __name__ == "__main__":
    main()
