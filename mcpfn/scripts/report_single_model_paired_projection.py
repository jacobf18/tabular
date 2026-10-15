"""Compare two protection scopes for one audited Adam proposal without training or evaluation."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_replicated_decoder as preflight
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_guard_contributions import write_csv
from report_single_model_replicated_gradient_pairs import paired_geometry
from single_model_family_projection import solve_projection


POLICY = ("CPU-only counterfactual first-order geometry for the independently audited doubled-draw preflight. "
          "Hold its original clipped composite Adam proposal fixed. Compare projection with0.1 normalized "
          "inward margin against all84 individual gradients versus42 equally averaged within-cohort/objective "
          "pairs. Preserve every raw gradient and report individual adverse directions as well as pair means. "
          "No parameter update, FP32 realization, finite-loss evaluation, checkpoint selection, confirmation "
          "or real benchmark. This does not establish generalization or justify weakening any qualification gate.")


def geometry(gram, products, norm, entries, tasks, margin=.1):
    gram, products = np.asarray(gram, dtype=np.float64), np.asarray(products, dtype=np.float64)
    pairs = paired_geometry(gram, entries, tasks)
    if (products.shape != (len(entries),) or not np.isfinite(products).all()
            or not math.isfinite(norm) or norm <= 0
            or np.any(np.abs(products) > np.sqrt(np.diag(gram)) * norm + 1e-12)):
        raise ValueError("Require the original finite nonzero Adam proposal and physically consistent products")
    averaging = np.zeros((len(pairs), len(entries)))
    for index, pair in enumerate(pairs):
        averaging[index, [pair["first_index"], pair["second_index"]]] = .5
    if not np.array_equal(averaging.sum(axis=0), np.full(len(entries), .5)):
        raise ValueError("Each raw gradient must occur in exactly one equal-weight pair")
    reduced = averaging @ gram @ averaging.T
    reduced_products = averaging @ products
    solutions = {"individual": solve_projection(gram, products, norm, margin_ratio=margin),
                 "paired_mean": solve_projection(reduced, reduced_products, norm, margin_ratio=margin)}
    summaries, projected = {}, {}
    for name, solution in solutions.items():
        coefficients = np.asarray(solution["raw_gradient_coefficients"])
        if name == "paired_mean":
            coefficients = averaging.T @ coefficients
        correction_square = float(coefficients @ gram @ coefficients)
        projected_square = norm**2 + 2 * float(coefficients @ products) + correction_square
        if min(correction_square, projected_square) < -1e-12:
            raise ValueError("Counterfactual squared norms must be nonnegative")
        values = products + gram @ coefficients
        projected[name] = values
        summaries[name] = {"constraints": len(solution["raw_gradient_coefficients"]),
                           "correction_norm": math.sqrt(max(0., correction_square)),
                           "projected_norm": math.sqrt(max(0., projected_square)),
                           "norm_ratio": math.sqrt(max(0., projected_square)) / norm,
                           "adverse_individual_directions": int(np.count_nonzero(values < 0)),
                           "adverse_pair_means": int(np.count_nonzero(averaging @ values < 0)),
                           "raw_gradient_coefficients": coefficients.tolist(), "solver": solution}
    rows = []
    for index, pair in enumerate(pairs):
        row = {"cohort": pair["cohort"], "kind": pair["kind"],
               "first_index": pair["first_index"], "second_index": pair["second_index"]}
        for name, values in {"adam": products, **projected}.items():
            row.update({f"{name}_first_product": float(values[pair["first_index"]]),
                        f"{name}_second_product": float(values[pair["second_index"]]),
                        f"{name}_mean_product": float(averaging[index] @ values)})
        rows.append(row)
    if summaries["paired_mean"]["correction_norm"] > summaries["individual"]["correction_norm"] + 1e-8:
        raise ValueError("The relaxed paired cone cannot require a farther exact Euclidean projection")
    return {"adam_norm": norm, "margin": margin, "summaries": summaries, "pairs": rows}


def report(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use only CPU analysis of saved evidence")
    if output.exists():
        raise FileExistsError("Never overwrite a diagnostic")
    config = preflight.verify(preflight.OUTPUT)
    audit_path = preflight.OUTPUT / "audit.json"
    audit = preflight.read(audit_path)
    if (not audit["complete"] or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["cuda_initialized"]):
        raise ValueError("Require every independently audited original task, gradient and prediction grid")
    verify_hashes(audit["artifact_sha256"])
    result_path, manifest_path = [preflight.OUTPUT / name for name in ("result.json", "training_batch/manifest.json")]
    result, manifest = preflight.read(result_path), preflight.read(manifest_path)
    if (manifest["step"] != 1 or len(manifest["tasks"]) != 28 or len(manifest["entries"]) != 84
            or result["record"]["margin"] != .1):
        raise ValueError("Preserve the complete original protection scope and margin")
    sources = [Path(__file__).resolve(), REPO / "scripts/report_single_model_replicated_gradient_pairs.py",
               REPO / "scripts/single_model_family_projection.py", REPO / "tests/test_single_model_paired_projection.py"]
    specification = {"policy": POLICY, "input_sha256": {str(path): sha256(path) for path in
                     (audit_path, result_path, manifest_path, preflight.OUTPUT / "config.json")},
                     "source_sha256": {str(path): sha256(path) for path in sources}}
    output.mkdir(parents=True)
    write_json(output / "config.json", specification)
    record = result["record"]
    outcome = geometry(record["joint_gram"], record["joint_products_before"], record["adam_norm"],
                       manifest["entries"], config["batch"]["tasks"])
    np.testing.assert_allclose(outcome["summaries"]["individual"]["raw_gradient_coefficients"],
                               record["projection"]["raw_gradient_coefficients"], rtol=1e-9, atol=1e-12)
    write_csv(output / "pairs.csv", outcome["pairs"])
    write_json(output / "result.json", {**outcome, "complete": True, "prediction_performed": False,
               "training_performed": False, "selection_performed": False, "goal_success_not_assessed": True,
               "config_sha256": sha256(output / "config.json")})
    lines = ["# Paired protection geometry", "", POLICY, "",
             "Positive products indicate first-order descent, not verified finite-loss improvement.", "",
             "| Scope | Directions | Correction norm | Projected/Adam norm | Adverse draws | Adverse means |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, summary in outcome["summaries"].items():
        lines.append(f'| {name} | {summary["constraints"]} | {summary["correction_norm"]:.10f} | '
                     f'{summary["norm_ratio"]:.10f} | {summary["adverse_individual_directions"]} | '
                     f'{summary["adverse_pair_means"]} |')
    lines.extend(["", "All42 pairs and all84 individual products are preserved in `pairs.csv`.",
                  "A fresh actual optimizer preflight is still required; no checkpoint is produced.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    verify_hashes(specification["input_sha256"])
    verify_hashes(specification["source_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.output.resolve())
