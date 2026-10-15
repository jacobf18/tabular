from copy import deepcopy

import pytest
import torch

from evaluate_dwip_fresh_compression import (
    ALLOWANCE, CASES_PER_GROUP, PANELS, RESERVE_BYTES, storage_guard, validate_population,
)
from run_dwip_continuous_spectrum import case_digest


def cases():
    return [{"id": f"group_{replicate}", "group": "group", "target": torch.tensor([float(replicate)])}
            for replicate in range(CASES_PER_GROUP)]


def test_fresh_scope_is_balanced_and_has_no_public_or_selection_panel():
    assert set(PANELS) == {"independent_spectral", "independent_general"}
    assert sum(panel["case_count"] for panel in PANELS.values()) == 544
    assert all(panel["case_count"] == panel["groups"] * CASES_PER_GROUP for panel in PANELS.values())
    assert len({panel["seed"] for panel in PANELS.values()}) == 2


def test_population_audit_accepts_full_registered_group_count():
    population = cases()
    result = validate_population(population, {"case_count": 16, "groups": 1}, ["old"])
    assert result["case_count"] == 16
    assert result["group_counts"] == {"group": 16}


@pytest.mark.parametrize("change", ("count", "duplicate", "groups", "imbalanced", "old_digest"))
def test_population_drift_or_reuse_fails_closed(change):
    population = deepcopy(cases())
    old = ["old"]
    registration = {"case_count": 16, "groups": 1}
    if change == "count":
        population.pop()
    elif change == "duplicate":
        population[-1]["id"] = population[0]["id"]
    elif change == "groups":
        registration["groups"] = 2
    elif change == "imbalanced":
        population[-1]["group"] = "other"
    else:
        old.append(case_digest(population))
    with pytest.raises(ValueError):
        validate_population(population, registration, old)


@pytest.mark.parametrize("allocated", (0, ALLOWANCE // 2, ALLOWANCE))
def test_storage_reserves_the_whole_remaining_allowance(allocated):
    minimum = RESERVE_BYTES + ALLOWANCE - allocated
    assert storage_guard(minimum, allocated)["remaining_allowance_bytes"] == ALLOWANCE - allocated
    with pytest.raises(RuntimeError, match="reserve"):
        storage_guard(minimum - 1, allocated)


@pytest.mark.parametrize("allocated", (-1, ALLOWANCE + 1))
def test_storage_rejects_invalid_ownership_accounting(allocated):
    with pytest.raises(RuntimeError, match="allowance"):
        storage_guard(100 * ALLOWANCE, allocated)
