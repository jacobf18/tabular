import numpy as np
import pytest
import torch

import single_model_family_projection as projection


def project(tasks, direction, margin=0.):
    gram, products, norm = projection.gram_and_products(tasks, direction, chunk_elements=2)
    result = projection.solve_projection(gram, products, norm, margin)
    updated = direction.double() + tasks.double().T @ torch.tensor(result["raw_gradient_coefficients"], dtype=torch.float64)
    return updated, result


def test_orthogonal_projection_matches_analytic_solution():
    tasks = torch.eye(3)
    direction = torch.tensor([-2., 3., -4.])
    updated, record = project(tasks, direction)
    torch.testing.assert_close(updated, torch.tensor([0., 3., 0.], dtype=torch.float64))
    assert record["nonzero_tasks"] == record["gram_rank"] == 3
    assert record["maximum_scaled_constraint_violation"] < 1e-10
    assert record["maximum_scaled_complementarity"] < 1e-10


def test_already_feasible_update_is_unchanged():
    tasks = torch.tensor([[1., 0.], [1., 1.]])
    direction = torch.tensor([2., 1.])
    updated, record = project(tasks, direction, 1e-4)
    torch.testing.assert_close(updated, direction.double())
    assert record["raw_gradient_coefficients"] == [0., 0.]


@pytest.mark.parametrize("tasks", [torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([[1., 0.], [-1., 0.]])])
def test_rank_deficient_zero_margin_constraints(tasks):
    direction = torch.tensor([-1., 2.])
    updated, record = project(tasks, direction)
    assert record["gram_rank"] == 1
    torch.testing.assert_close(updated, torch.tensor([0., 2.], dtype=torch.float64))


def test_redundant_constraints_can_have_feasible_positive_margin():
    tasks = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    direction = torch.tensor([-1., -1.])
    updated, record = project(tasks, direction, .01)
    norms = torch.linalg.vector_norm(tasks.double(), dim=1)
    assert torch.all((tasks.double() @ updated) / norms >= .01 * direction.double().norm() - 1e-10)
    assert record["gram_rank"] == 2


def test_impossible_strict_inward_margin_is_rejected():
    tasks = torch.tensor([[1., 0.], [-1., 0.]])
    with pytest.raises(ValueError, match="infeasible|converge"):
        project(tasks, torch.tensor([-1., 2.]), .01)


def test_positive_margin_survives_actual_fp32_parameter_rounding():
    tasks = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    before = torch.tensor([.2, -.3, 1.])
    tentative = before + torch.tensor([2e-5, 1e-5, 1e-5])
    direction = before.double() - tentative.double()
    gram, products, norm = projection.gram_and_products(tasks, direction)
    record = projection.solve_projection(gram, products, norm)
    result = projection.projected_parameters(tentative, tasks, record["raw_gradient_coefficients"], chunk_elements=2)
    realized = before.double() - result.double()
    assert torch.all(tasks.double() @ realized >= 0)
    assert result.dtype == before.dtype
    assert torch.equal(before, torch.tensor([.2, -.3, 1.]))


def test_zero_gradient_and_zero_update_are_explicit():
    updated, record = project(torch.zeros(2, 3), torch.zeros(3), .01)
    assert record["nonzero_tasks"] == 0 and record["gram_rank"] == 0
    assert torch.equal(updated, torch.zeros(3, dtype=torch.float64))
    updated, record = project(torch.tensor([[0., 0.], [1., 0.]]), torch.tensor([-1., 2.]))
    assert record["nonzero_tasks"] == 1
    torch.testing.assert_close(updated, torch.tensor([0., 2.], dtype=torch.float64))


def test_projection_invariant_to_positive_task_rescaling():
    tasks = torch.tensor([[1., 2., 0.], [-1., 0., 2.]])
    direction = torch.tensor([-2., -3., -1.])
    first, _ = project(tasks, direction, .001)
    second, _ = project(tasks * torch.tensor([[5.], [.25]]), direction, .001)
    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)


def test_chunked_products_reproduce_direct_float64_geometry():
    generator = torch.Generator().manual_seed(16)
    tasks = torch.randn((14, 57), generator=generator)
    direction = torch.randn(57, generator=generator)
    gram, products, norm = projection.gram_and_products(tasks, direction, 11)
    np.testing.assert_allclose(gram, (tasks.double() @ tasks.double().T).numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(products, (tasks.double() @ direction.double()).numpy(), rtol=1e-12, atol=1e-12)
    assert norm == pytest.approx(float(direction.double().norm()), rel=1e-12)
    updated, record = project(tasks, direction)
    assert torch.all(tasks.double() @ updated >= -1e-8)
    assert float(updated.norm()) <= float(direction.norm()) + 1e-6
    assert record["research_selection_tolerance"] == 0


def test_weighted_gradient_uses_per_task_caps_and_preserves_inputs():
    tasks = torch.tensor([[3., 4.], [0., 2.]])
    original = tasks.clone()
    vector, record = projection.combined_gradient(tasks, [.25, .75], chunk_elements=1)
    expected = .25 * tasks[0] * (.5 / (5. + 1e-6)) + .75 * tasks[1] * (.5 / (2. + 1e-6))
    torch.testing.assert_close(vector, expected)
    assert torch.equal(tasks, original) and float(vector.norm()) <= .5
    assert record["weights"] == [.25, .75]


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "indefinite", "asymmetric", "zero_product", "negative_margin"])
def test_invalid_projection_geometry_is_rejected(kind):
    gram, products, norm, margin = np.eye(2), np.array([-1., 1.]), 2., 0.
    if kind == "shape":
        products = np.zeros(3)
    elif kind == "nonfinite":
        products[0] = np.nan
    elif kind == "indefinite":
        gram = np.array([[1., 2.], [2., 1.]])
    elif kind == "asymmetric":
        gram[0, 1] = .5
    elif kind == "zero_product":
        gram[0, 0] = 0.
    else:
        margin = -1.
    with pytest.raises(ValueError):
        projection.solve_projection(gram, products, norm, margin)


@pytest.mark.parametrize("weights", [[.2, .2], [-.1, 1.1], [1.], [np.nan, 0.]])
def test_invalid_task_weights_cannot_change_the_objective_silently(weights):
    with pytest.raises(ValueError):
        projection.combined_gradient(torch.ones(2, 3), weights)
