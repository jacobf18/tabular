from copy import deepcopy

import pytest

import audit_single_matrixformer_domain_clipping as audit
import single_model_domain_clipping as clipping
import train_single_matrixformer_domain_clipping as training


def evidence():
    return {"complete": True, "pairs": 64, "gradient_vectors": 128, "all_original_coordinates_remeasured": True,
            "original_artifacts_unchanged": True, "cuda_initialized": False, "runtime_reported_model_unchanged": True,
            "groups": {"structured": {"defined_norm_ratios": 48, "median_imputation_to_rating_norm": 10.},
                       "all": {"cases": 64, "methods": {
                           "joint_clip": {"rating_negative_descent_cases": 20, "imputation_negative_descent_cases": 0},
                           "domain_clip_then_mean": {"rating_negative_descent_cases": 0, "imputation_negative_descent_cases": 0}}}}}


def test_admission_requires_both_scale_evidence_and_better_direction_counts():
    values = evidence()
    assert clipping.admitted(values)
    values["groups"]["structured"]["median_imputation_to_rating_norm"] = 2.
    assert not clipping.admitted(values)
    values = evidence()
    values["groups"]["all"]["methods"]["domain_clip_then_mean"]["rating_negative_descent_cases"] = 20
    assert not clipping.admitted(values)
    values = evidence()
    values["groups"]["all"]["methods"]["domain_clip_then_mean"]["imputation_negative_descent_cases"] = 1
    assert not clipping.admitted(values)
    values["pairs"] = 63
    with pytest.raises(ValueError):
        clipping.admitted(values)


def test_paired_recipe_cannot_change_old_data_objectives_or_fingerprints():
    base = {"version": 9, "steps": 400, "input_sha256": {"input": "fixed"}, "source_sha256": {"source": "fixed"},
            "experiment": "original", "validation_cases": [{"seed": 123}], "loss": {"weight": 1.}}
    candidate = training.make_config(base)
    training.validate_pair(candidate, base)
    for key, value in (("steps", 200), ("validation_cases", [{"seed": 124}]), ("loss", {"weight": 2.}),
                       ("input_sha256", {}), ("source_sha256", {}), ("domain_clipping", {})):
        changed = deepcopy(candidate)
        changed[key] = value
        with pytest.raises(ValueError):
            training.validate_pair(changed, base)


def log_row():
    records = []
    for domain, norm in (("imputation", 10.), ("recsys", .1)):
        coefficient = min(1., .25 / (norm + 1e-6))
        records.append({"domain": domain, "domain_gradient_clipping": {
            "domain_weight": .5, "weighted_clip": .25, "unweighted_clip": .5,
            "weighted_norm_before_clip": norm, "clip_coefficient": coefficient,
            "weighted_norm_after_clip": norm * coefficient, "gradient_isolated_before_backward": True}})
    return {"step": 1, "microsteps": records, "grad_norm": .3}


def test_clipping_log_audit_keeps_the_exact_policy_and_triangle_bound():
    row = log_row()
    audit.check_record(row)
    row["grad_norm"] = .6
    with pytest.raises(ValueError, match="triangle bound"):
        audit.check_record(row)
    row = log_row()
    row["microsteps"][0]["domain_gradient_clipping"]["clip_coefficient"] = .1
    with pytest.raises(ValueError, match="does not reproduce"):
        audit.check_record(row)


def test_gradient_bound_roundoff_is_not_a_selection_margin():
    row = log_row()
    row["grad_norm"] = sum(case["domain_gradient_clipping"]["weighted_norm_after_clip"] for case in row["microsteps"]) + 1e-6
    audit.check_record(row)
    assert clipping.SPECIFICATION["final_global_clip"] == .5
