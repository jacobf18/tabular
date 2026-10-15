"""Bounded diagnostic refinement against actual FP32 parameter displacements."""

import numpy as np
import torch

from preflight_single_model_family_projection import verify_certificate
import single_model_family_projection as projection


MARGINS = (1e-4, 1e-3, 1e-2, 1e-1)


@torch.no_grad()
def realized_projection(before, tentative, gradients):
    if (before.shape != tentative.shape or before.dtype != torch.float32 or tentative.dtype != torch.float32
            or before.device != tentative.device or not torch.isfinite(before).all()):
        raise ValueError("Require the original aligned finite FP32 parameter vectors")
    gram, products, norm = projection.gram_and_products(gradients, before.double() - tentative.double())
    attempts = []
    for margin in MARGINS:
        record = {"margin": margin, "accepted": False, "parameters": None}
        try:
            solution = projection.solve_projection(gram, products, norm, margin_ratio=margin)
            verify_certificate(gram, products, norm, solution, margin)
        except (ValueError, AssertionError) as error:
            record["solver_error"] = repr(error)
            attempts.append(record)
            continue
        after = projection.projected_parameters(tentative, gradients, solution["raw_gradient_coefficients"])
        _, realized_products, realized_norm = projection.gram_and_products(gradients, before.double() - after.double())
        accepted = bool(np.all(realized_products >= 0) and np.any(realized_products > 0))
        record.update(solution=solution, parameters=after, actual_products=realized_products.tolist(),
                      actual_norm=realized_norm, adverse_tasks=int(np.sum(realized_products < 0)), accepted=accepted)
        attempts.append(record)
        if accepted:
            return {"accepted_index": len(attempts) - 1, "attempts": attempts}
    return {"accepted_index": None, "attempts": attempts}
