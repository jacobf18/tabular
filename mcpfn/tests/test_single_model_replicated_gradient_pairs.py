from copy import deepcopy

import numpy as np
import pytest

from report_single_model_replicated_gradient_pairs import paired_geometry


def fixture():
    tasks = [{"name": f"legacy_draw_{draw}", "cohort": "legacy", "replicate": draw, "weight": .5,
              "gradient_seed": 100 + draw, "specification": {"domain": "imputation", "seed": 200 + draw, "shape": [20, 5]}}
             for draw in (0, 1)]
    entries = [{"task": task["name"], "kind": kind, "index": index}
               for index, (task, kind) in enumerate((task, kind) for task in tasks for kind in ("composite", "rmse", "common_nrmse"))]
    gradients = np.asarray([[1., 0.], [0., 1.], [1., 1.], [1., 0.], [0., -1.], [1., -1.]])
    return gradients @ gradients.T, entries, tasks


def test_identical_opposing_and_orthogonal_draws_have_the_expected_geometry():
    arguments = fixture()
    before = deepcopy(arguments[1:])
    rows = paired_geometry(*arguments)
    assert arguments[1:] == before
    assert [row["cosine"] for row in rows] == pytest.approx([1., -1., 0.])
    assert [row["mean_energy_fraction"] for row in rows] == pytest.approx([1., 0., .5])
    assert rows[1]["mean_norm"] == 0
    assert rows[1]["half_difference_norm"] == 1


def test_zero_gradient_pairs_remain_explicit_without_fabricated_correlations():
    gram, entries, tasks = fixture()
    gram.fill(0)
    rows = paired_geometry(gram, entries, tasks)
    assert all(row["cosine"] is None and row["mean_energy_fraction"] is None for row in rows)
    assert all(row["mean_norm"] == row["half_difference_norm"] == 0 for row in rows)


@pytest.mark.parametrize("mutation", ["shape", "nonfinite", "asymmetric", "negative", "product", "slot", "objective", "draw", "recipe", "seed", "weight"])
def test_changed_geometry_or_paired_coverage_is_rejected(mutation):
    gram, entries, tasks = fixture()
    if mutation == "shape":
        gram = gram[:-1]
    elif mutation == "nonfinite":
        gram[0, 0] = np.nan
    elif mutation == "asymmetric":
        gram[0, 1] = 2.
    elif mutation == "negative":
        gram[0, 0] = -1.
    elif mutation == "product":
        gram[0, 3] = gram[3, 0] = 2.
    elif mutation == "slot":
        entries[0]["index"] = 1
    elif mutation == "objective":
        entries[0]["kind"] = "unknown"
    elif mutation == "draw":
        tasks.pop()
    elif mutation == "recipe":
        tasks[0]["specification"]["shape"] = [30, 5]
    elif mutation == "seed":
        tasks[1]["specification"]["seed"] = tasks[0]["specification"]["seed"]
    else:
        tasks[1]["weight"] = .4
    with pytest.raises(ValueError):
        paired_geometry(gram, entries, tasks)
