import copy

import pytest

from diagnose_dwip_gradient_reference_v2 import check_reproduction, reference_ids


def test_revised_audit_preserves_exact_reference_population():
    assert reference_ids() == [*range(64001, 64065), *range(64129, 65089)]


def test_backward_drift_is_reported_without_relaxing_loss_or_identity_checks():
    original = {"id": 64002, "cohort": "reference", "family": "spectral", "shape": [2, 80, 40],
                "support": {"support_episode": 1.}, "losses": {"mse": 1., "total": 2.}, "preclip_norm": 10.}
    current = copy.deepcopy(original)
    current["preclip_norm"] = 10.005
    assert check_reproduction(current, original)["relative_norm_delta"] == pytest.approx(.0005)
    current["losses"]["mse"] = 1.01
    with pytest.raises(ValueError):
        check_reproduction(current, original)
    current = copy.deepcopy(original)
    current["cohort"] = "probe"
    with pytest.raises(ValueError):
        check_reproduction(current, original)
    current = copy.deepcopy(original)
    current["preclip_norm"] = float("nan")
    with pytest.raises(ValueError):
        check_reproduction(current, original)
