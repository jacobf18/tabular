"""Small-dual projection of a parameter update against synthetic task gradients."""

import math

import numpy as np
from scipy.optimize import minimize
import torch


SPECIFICATION = {"relative_inward_margin": 1e-4, "gram_relative_rank_cutoff": 1e-12,
                 "scaled_kkt_tolerance": 1e-8, "chunk_elements": 1_000_000,
                 "policy": "Closest Euclidean update to the actual AdamW step, subject to positive first-order descent margins for every nonzero sampled task gradient; numerical solver tolerances are not validation tolerances"}


def gram_and_products(task_gradients, direction, chunk_elements=SPECIFICATION["chunk_elements"]):
    if (task_gradients.ndim != 2 or direction.ndim != 1 or task_gradients.shape[1] != direction.numel()
            or not direction.numel() or not task_gradients.shape[0] or task_gradients.device != direction.device
            or chunk_elements < 1):
        raise ValueError("Require aligned nonempty task gradients and an update on one device")
    count = task_gradients.shape[0]
    gram = torch.zeros((count, count), device=direction.device, dtype=torch.float64)
    products = torch.zeros(count, device=direction.device, dtype=torch.float64)
    square = torch.zeros((), device=direction.device, dtype=torch.float64)
    for start in range(0, direction.numel(), chunk_elements):
        tasks = task_gradients[:, start:start + chunk_elements].double()
        current = direction[start:start + chunk_elements].double()
        if not torch.isfinite(tasks).all() or not torch.isfinite(current).all():
            raise ValueError("Every gradient and update coordinate must be finite")
        gram.add_(tasks @ tasks.T)
        products.add_(tasks @ current)
        square.add_(current @ current)
    return gram.cpu().numpy(), products.cpu().numpy(), math.sqrt(max(float(square), 0.))


def solve_projection(gram, products, direction_norm, margin_ratio=SPECIFICATION["relative_inward_margin"]):
    gram, products = np.asarray(gram, dtype=np.float64), np.asarray(products, dtype=np.float64)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or products.shape != (gram.shape[0],)
            or not products.size or not np.isfinite(gram).all() or not np.isfinite(products).all()
            or not math.isfinite(direction_norm) or direction_norm < 0 or not math.isfinite(margin_ratio) or margin_ratio < 0):
        raise ValueError("Require finite complete projection geometry")
    if not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-15) or np.any(np.diag(gram) < 0):
        raise ValueError("Task Gram matrix must be symmetric with nonnegative diagonal")
    norms = np.sqrt(np.diag(gram))
    active = norms > 0
    if np.any(products[~active] != 0) or np.any(gram[~active] != 0):
        raise ValueError("A zero-gradient task cannot have a nonzero product")
    multipliers, coefficients = np.zeros_like(products), np.zeros_like(products)
    before, after = np.zeros_like(products), np.zeros_like(products)
    rank, violation, complementarity = 0, 0., 0.
    if direction_norm == 0:
        if np.any(products != 0):
            raise ValueError("A zero update cannot have a nonzero task product")
    elif active.any():
        normalized = gram[np.ix_(active, active)] / np.outer(norms[active], norms[active])
        before[active] = products[active] / (norms[active] * direction_norm)
        linear = before[active] - margin_ratio
        eigenvalues = np.linalg.eigvalsh(normalized)
        if eigenvalues.min() < -1e-9:
            raise ValueError("Task Gram matrix must be positive semidefinite")
        retained = eigenvalues > max(1., float(eigenvalues.max())) * SPECIFICATION["gram_relative_rank_cutoff"]
        rank = int(retained.sum())
        result = minimize(lambda values: .5 * values @ normalized @ values + linear @ values,
                          np.zeros_like(linear), jac=lambda values: normalized @ values + linear,
                          bounds=[(0., None)] * len(linear), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
        solution = result.x
        if not result.success or not np.isfinite(solution).all() or np.any(solution < 0):
            raise ValueError("The inward projection is infeasible or its dual did not converge")
        face = solution > 1e-10
        if face.any():
            refined = np.zeros_like(solution)
            refined[face] = np.linalg.lstsq(normalized[np.ix_(face, face)], -linear[face], rcond=1e-12)[0]
            if np.all(refined >= 0) and np.min(normalized @ refined + linear) >= -1e-10:
                solution = refined
        multipliers[active] = solution
        coefficients[active] = solution * direction_norm / norms[active]
        after[active] = before[active] + normalized @ solution
        slack = after[active] - margin_ratio
        violation = max(0., float(-slack.min()))
        complementarity = float(np.max(np.abs(solution * slack)))
        if violation > SPECIFICATION["scaled_kkt_tolerance"] or complementarity > SPECIFICATION["scaled_kkt_tolerance"] * max(1., float(solution.max())):
            raise ValueError("Projection KKT conditions did not converge")
    return {"raw_gram": gram.tolist(), "raw_products_before": products.tolist(), "gradient_norms": norms.tolist(),
            "direction_norm": direction_norm, "relative_inward_margin": margin_ratio, "nonzero_tasks": int(active.sum()),
            "gram_rank": rank, "scaled_products_before": before.tolist(), "scaled_products_after": after.tolist(),
            "dual_multipliers": multipliers.tolist(), "raw_gradient_coefficients": coefficients.tolist(),
            "maximum_scaled_constraint_violation": violation, "maximum_scaled_complementarity": complementarity,
            "research_selection_tolerance": 0, "finite_step_loss_decrease_not_guaranteed": True}


def projected_parameters(tentative, task_gradients, coefficients, chunk_elements=SPECIFICATION["chunk_elements"]):
    if (tentative.ndim != 1 or task_gradients.ndim != 2 or task_gradients.shape[1] != tentative.numel()
            or tentative.device != task_gradients.device or tentative.dtype != torch.float32 or chunk_elements < 1):
        raise ValueError("Require the original flat FP32 parameter inventory")
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64, device=tentative.device)
    if coefficients.shape != (task_gradients.shape[0],) or not torch.isfinite(coefficients).all():
        raise ValueError("Require one finite coefficient per task")
    result = torch.empty_like(tentative)
    for start in range(0, tentative.numel(), chunk_elements):
        tasks = task_gradients[:, start:start + chunk_elements].double()
        original = tentative[start:start + chunk_elements].double()
        if not torch.isfinite(tasks).all() or not torch.isfinite(original).all():
            raise ValueError("Nonfinite projection input")
        result[start:start + chunk_elements] = (original - tasks.T @ coefficients).float()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite projected parameters")
    return result


def combined_gradient(task_gradients, weights, maximum=.5, chunk_elements=SPECIFICATION["chunk_elements"]):
    if task_gradients.ndim != 2 or not task_gradients.shape[0] or not task_gradients.shape[1]:
        raise ValueError("Require a nonempty matrix of task gradients")
    weights = torch.as_tensor(weights, dtype=torch.float64, device=task_gradients.device)
    if (weights.shape != (task_gradients.shape[0],) or not torch.isfinite(weights).all() or (weights < 0).any()
            or not math.isclose(float(weights.sum()), 1., abs_tol=1e-12) or not math.isfinite(maximum) or maximum <= 0 or chunk_elements < 1):
        raise ValueError("Require nonnegative task weights summing to one and a positive clip")
    zeros = torch.zeros(task_gradients.shape[1], dtype=torch.float32, device=task_gradients.device)
    gram, _, _ = gram_and_products(task_gradients, zeros, chunk_elements)
    norms = np.sqrt(np.diag(gram))
    coefficients = weights * torch.as_tensor(np.minimum(1., maximum / (norms + 1e-6)), device=weights.device)
    for start in range(0, zeros.numel(), chunk_elements):
        zeros[start:start + chunk_elements] = (task_gradients[:, start:start + chunk_elements].double().T @ coefficients).float()
    return zeros, {"weights": weights.cpu().tolist(), "raw_gradient_norms": norms.tolist(),
                   "weighted_clip_coefficients": coefficients.cpu().tolist(), "unweighted_task_clip": maximum}
