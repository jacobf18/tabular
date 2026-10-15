from copy import deepcopy

import numpy as np
import pytest

from report_single_model_paired_projection import geometry


def fixture():
    tasks = [{"name": f"legacy_draw_{draw}", "cohort": "legacy", "replicate": draw, "weight": .5,
              "gradient_seed": 100 + draw, "specification": {"domain": "imputation", "seed": 200 + draw}}
             for draw in (0, 1)]
    entries = [{"task": task["name"], "kind": kind, "index": index}
               for index, (task, kind) in enumerate((task, kind) for task in tasks
                                                   for kind in ("composite", "rmse", "common_nrmse"))]
    gradients = np.asarray([[1., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 1.], [1., 1.]])
    direction = np.asarray([-1., -2.])
    return gradients, direction, entries, tasks


def test_identical_draws_reproduce_the_same_projection_and_parameter_space_geometry():
    gradients, direction, entries, tasks = fixture()
    before = deepcopy((entries, tasks))
    result = geometry(gradients @ gradients.T, gradients @ direction, np.linalg.norm(direction), entries, tasks)
    assert (entries, tasks) == before
    assert len(result["pairs"]) == 3
    individual, paired = result["summaries"].values()
    assert individual["constraints"] == 6 and paired["constraints"] == 3
    assert individual["correction_norm"] == pytest.approx(paired["correction_norm"])
    for summary in (individual, paired):
        actual = direction + gradients.T @ np.asarray(summary["raw_gradient_coefficients"])
        assert np.linalg.norm(actual) == pytest.approx(summary["projected_norm"])
        assert np.linalg.norm(actual - direction) == pytest.approx(summary["correction_norm"])
        assert summary["adverse_individual_directions"] == summary["adverse_pair_means"] == 0


def test_paired_protection_reports_rather_than_hides_adverse_individual_draws():
    gradients, direction, entries, tasks = fixture()
    gradients[:] = [[1., 0.], [1., 0.], [1., 0.], [-.8, 1.], [-.8, 1.], [-.8, 1.]]
    direction[:] = [-1., 1.]
    result = geometry(gradients @ gradients.T, gradients @ direction, np.linalg.norm(direction), entries, tasks)
    individual, paired = result["summaries"].values()
    assert paired["correction_norm"] < individual["correction_norm"]
    assert individual["adverse_individual_directions"] == 0
    assert paired["adverse_individual_directions"] == 3
    assert paired["adverse_pair_means"] == 0
    assert all(row["paired_mean_first_product"] < 0 < row["paired_mean_mean_product"]
               for row in result["pairs"])


@pytest.mark.parametrize("mutation", ["products", "nonfinite", "norm", "impossible", "duplicate", "recipe", "weight"])
def test_invalid_or_changed_evidence_cannot_be_silently_averaged(mutation):
    gradients, direction, entries, tasks = fixture()
    products, norm = gradients @ direction, np.linalg.norm(direction)
    if mutation == "products":
        products = products[:-1]
    elif mutation == "nonfinite":
        products[0] = np.nan
    elif mutation == "norm":
        norm = 0
    elif mutation == "impossible":
        products[0] = 100
    elif mutation == "duplicate":
        entries[0]["index"] = 1
    elif mutation == "recipe":
        tasks[0]["specification"]["extra"] = True
    else:
        tasks[0]["weight"] = .4
    with pytest.raises(ValueError):
        geometry(gradients @ gradients.T, products, norm, entries, tasks)
