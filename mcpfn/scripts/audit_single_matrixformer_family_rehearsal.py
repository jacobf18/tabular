"""Reuse full raw-prediction and gradient audits with the prospectively fixed V11 schedule."""

import audit_single_matrixformer_domain_clipping as original
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
