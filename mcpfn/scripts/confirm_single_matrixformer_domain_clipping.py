"""Use the unchanged independent284-case confirmation for the paired optimizer sibling."""

import confirm_single_matrixformer_structured_mixture as original
from train_single_matrixformer_domain_clipping import validate_recipe


def main():
    previous = original.validate_recipe
    original.validate_recipe = validate_recipe
    try:
        original.main()
    finally:
        original.validate_recipe = previous


if __name__ == "__main__":
    main()
