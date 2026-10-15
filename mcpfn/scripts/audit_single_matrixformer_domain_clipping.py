"""Keep the full V9 prediction audit and additionally check the paired clipping log."""

import json
import math
from pathlib import Path

import audit_single_matrixformer_structured_training as original
from recsys_evaluation_common import sha256, write_json
import single_model_domain_clipping as clipping
from train_single_matrixformer_domain_clipping import validate_recipe


def check_record(row):
    if [case["domain"] for case in row["microsteps"]] != clipping.SPECIFICATION["domain_order"]:
        raise ValueError("The original domain pair changed")
    bound = 0.
    for case in row["microsteps"]:
        record = case["domain_gradient_clipping"]
        if (record["domain_weight"] != .5 or record["weighted_clip"] != .25 or record["unweighted_clip"] != .5
                or not record["gradient_isolated_before_backward"]):
            raise ValueError("The registered domain clipping policy changed")
        norm = record["weighted_norm_before_clip"]
        if not math.isfinite(norm) or norm < 0:
            raise ValueError("Require finite pre-clipping domain norms")
        coefficient = min(1., .25 / (norm + 1e-6))
        if record["clip_coefficient"] != coefficient or record["weighted_norm_after_clip"] != norm * coefficient:
            raise ValueError("Logged isolated-domain clipping does not reproduce")
        bound += norm * coefficient
    if not math.isfinite(row["grad_norm"]) or not 0 <= row["grad_norm"] <= bound + 2e-6:
        raise ValueError("Summed gradient violates the triangle bound for two clipped domains")


def augment(original_audit, run, output, through_step):
    original_audit(run, output, through_step)
    rows = []
    for line in (run / "training.jsonl").read_text().splitlines(keepends=True) if through_step else ():
        if line.endswith("\n"):
            row = json.loads(line)
            if row["step"] <= through_step:
                check_record(row)
                rows.append(row)
    if [row["step"] for row in rows] != list(range(1, through_step + 1)):
        raise ValueError("Every registered optimizer update must have both domain records")
    write_json(output / "domain_clipping_audit.json", {"complete": True, "through_step": through_step,
        "logged_domain_gradients_checked": 2 * len(rows), "clipping_specification": clipping.SPECIFICATION,
        "checked_rows": rows, "numerical_triangle_bound_slack": 2e-6, "research_selection_tolerance": 0,
        "raw_prediction_audit_sha256": sha256(output / "audit.json"), "source_sha256": sha256(Path(__file__)),
        "optimizer_trajectory_replayed": False, "goal_success_not_assessed": True})


def main():
    old_validator, old_audit = original.validate_recipe, original.audit
    original.validate_recipe = validate_recipe
    original.audit = lambda run, output, through_step: augment(old_audit, run, output, through_step)
    try:
        original.main()
    finally:
        original.validate_recipe, original.audit = old_validator, old_audit


if __name__ == "__main__":
    main()
