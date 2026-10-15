"""Read-only case/column decomposition of the audited V10 synthetic trajectory."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from audit_single_matrixformer_structured_training import close_values
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
from report_single_model_domain_clipping import validate_clipping_audit
import single_model_structured_mixture as mixture


def column_changes(truth, mask, initial, final):
    arrays = [np.asarray(value) for value in (truth, mask, initial, final)]
    truth, mask, initial, final = arrays
    if (truth.ndim != 2 or any(value.shape != truth.shape for value in arrays)
            or mask.dtype != np.bool_ or not mask.any()
            or any(not np.isfinite(value[mask]).all() for value in (truth, initial, final))):
        raise ValueError("Require aligned finite queried cells and a Boolean mask")
    result = []
    for column in np.flatnonzero(mask.any(axis=0)):
        selected = mask[:, column]
        target = truth[selected, column].astype(np.float64)
        error = initial[selected, column].astype(np.float64) - target
        movement = final[selected, column].astype(np.float64) - initial[selected, column].astype(np.float64)
        scale = float(np.std(target))
        initial_mse = float(np.mean(error**2))
        final_mse = float(np.mean((error + movement)**2))
        linear = float(2 * np.mean(error * movement))
        quadratic = float(np.mean(movement**2))
        if not math.isclose(final_mse - initial_mse, linear + quadratic, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Squared-error displacement decomposition failed")
        defined = scale >= 1e-3
        result.append({"column": int(column), "queries": int(selected.sum()), "target_std": scale,
            "common_defined": defined, "initial_mse": initial_mse, "final_mse": final_mse,
            "mse_linear_change": linear, "mse_quadratic_change": quadratic,
            "initial_common_nrmse": math.sqrt(initial_mse) / scale if defined else None,
            "final_common_nrmse": math.sqrt(final_mse) / scale if defined else None})
    return result


def summarize_family(rows):
    if not rows or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Require every distinct family case exactly once")
    defined = [row for row in rows if row["initial_common_nrmse"] is not None]
    if any((row["initial_common_nrmse"] is None) != (row["final_common_nrmse"] is None) for row in rows) or not defined:
        raise ValueError("Keep fixed truth-defined coverage")
    initial = float(np.mean([row["initial_common_nrmse"] for row in defined]))
    final = float(np.mean([row["final_common_nrmse"] for row in defined]))
    contributions = {str(row["index"]): (row["final_common_nrmse"] - row["initial_common_nrmse"]) / len(defined)
                     for row in defined}
    return {"cases": len(rows), "defined_cases": len(defined), "initial": initial, "final": final,
            "delta": final - initial, "relative_percent": 100 * (final / initial - 1) if initial else None,
            "regressed_cases": sum(value > 0 for value in contributions.values()), "case_contributions": contributions}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Do not write an empty diagnostic")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(run, audit, output):
    if output.exists():
        raise FileExistsError("Never overwrite an earlier diagnostic")
    raw = json.loads((audit / "audit.json").read_text())
    auxiliary = json.loads((audit / "domain_clipping_audit.json").read_text())
    validate_clipping_audit(auxiliary, sha256(audit / "audit.json"))
    if not raw["complete"] or raw["through_step"] != 400 or not raw["original_artifacts_unchanged"]:
        raise ValueError("Require the complete immutable V10 audit")
    verify_hashes(raw["original_artifact_sha256"])
    inputs = {str(audit / name): sha256(audit / name) for name in ("audit.json", "domain_clipping_audit.json")}
    inputs.update(raw["original_artifact_sha256"])
    training_path = run / "training.jsonl"
    inputs[str(training_path)] = sha256(training_path)
    reports = [json.loads((run / f"validation_{step}.json").read_text()) for step in range(0, 401, 50)]
    if any(report["best_step"] != 0 for report in reports) or any(report["eligible"] for report in reports[1:]):
        raise ValueError("This diagnosis must label the V10 tail as unselected")
    bank = torch.load(run / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    training = [json.loads(line) for line in training_path.read_text().splitlines()]
    if [row["step"] for row in training] != list(range(1, 401)):
        raise ValueError("Retain all400 training records")
    exposure = Counter((row["microsteps"][0].get("family"), row["microsteps"][0].get("component"))
                       for row in training if row["microsteps"][0].get("structured"))
    output.mkdir(parents=True)
    write_json(output / "registration.json", {"scope": "Exploratory synthetic development diagnosis only",
        "input_sha256": inputs, "source_sha256": sha256(Path(__file__)), "training_performed": False,
        "selected_tail": False, "confirmation_bank_consumed": False})
    cases, columns, trajectory = [], [], []
    for first, last in zip(reports[0]["cases"], reports[-1]["cases"]):
        if not first.get("structured"):
            continue
        index = first["index"]
        episode = mixture.backend.restore_episode(bank[index], "cpu")
        with np.load(first["prediction_file"], allow_pickle=False) as saved:
            initial = saved["student"].copy()
        with np.load(last["prediction_file"], allow_pickle=False) as saved:
            final = saved["student"].copy()
        for prediction, record in ((initial, first), (final, last)):
            close_values(mixture.defined.episode_metrics(torch.from_numpy(prediction), episode), record["student"])
        decomposition = column_changes(episode.target[0].numpy(), episode.mask[0].numpy(), initial[0], final[0])
        for role, record in (("initial", first), ("final", last)):
            values = [column[f"{role}_common_nrmse"] for column in decomposition if column["common_defined"]]
            close_values(float(np.mean(values)) if values else None, record["student"]["common_nrmse"])
        case = {key: first[key] for key in ("index", "family", "component", "rows", "columns", "regime",
                                           "requested_missingness", "effective_missingness", "seed")}
        case.update(initial_common_nrmse=first["student"]["common_nrmse"], final_common_nrmse=last["student"]["common_nrmse"],
                    training_pair_exposures=exposure[first["family"], first["component"]])
        cases.append(case)
        columns.extend({"index": index, "family": first["family"], "component": first["component"], **row} for row in decomposition)
        for report in reports:
            record = report["cases"][index]
            if record["seed"] != first["seed"] or record["family"] != first["family"]:
                raise ValueError("Trajectory identities changed")
            trajectory.append({"index": index, "family": first["family"], "component": first["component"],
                               "step": report["step"], "common_nrmse": record["student"]["common_nrmse"]})
    if len(cases) != 96:
        raise ValueError("Every structured case must be reported")
    families = {family: summarize_family([row for row in cases if row["family"] == family]) for family in mixture.FAMILIES}
    for family, record in families.items():
        close_values(record["initial"], reports[0]["student"]["structured"]["families"][family]["common_nrmse"])
        close_values(record["final"], reports[-1]["student"]["structured"]["families"][family]["common_nrmse"])
    summary = {"complete": True, "scope": "Synthetic development data; unselected V10 update400, not a benchmark or confirmation",
               "families": families, "cases": len(cases), "columns": len(columns), "trajectory_records": len(trajectory),
               "original_artifacts_unchanged": True, "new_forward_passes": 0, "goal_success_not_assessed": True}
    verify_hashes(inputs)
    write_csv(output / "cases.csv", cases)
    write_csv(output / "columns.csv", columns)
    write_csv(output / "trajectory.csv", trajectory)
    write_json(output / "summary.json", summary)
    lines = ["# V10 case-level retention diagnosis", "", summary["scope"] + ".", "",
             "All96 structured cases, every queried column, and all nine development boundaries are retained.",
             "Saved predictions reproduce the original scores; no new forward pass or training is performed.", "",
             "| Family | Initial | Update400 | Change (%) | Regressed cases / defined cases |", "|---|---:|---:|---:|---:|"]
    for family, record in families.items():
        lines.append(f'| {family} | {record["initial"]:.9f} | {record["final"]:.9f} | {record["relative_percent"]:+.6f} | {record["regressed_cases"]}/{record["defined_cases"]} |')
    for family, record in families.items():
        if record["delta"] <= 0:
            continue
        lines.extend(["", f"## {family}: all cases", "", "| Case | Generator | Shape | Initial | Update400 | Contribution to family change | Training pair draws |",
                      "|---:|---|---|---:|---:|---:|---:|"])
        for row in cases:
            if row["family"] == family:
                contribution = record["case_contributions"].get(str(row["index"]))
                lines.append(f'| {row["index"]} | {row["component"]} | {row["rows"]}x{row["columns"]} | {row["initial_common_nrmse"]:.9f} | {row["final_common_nrmse"]:.9f} | {contribution:+.9f} | {row["training_pair_exposures"]} |')
    lines.extend(["", "Column squared-error changes are decomposed exactly as 2*mean(initial_error*prediction_movement) + mean(prediction_movement^2).",
                  "This describes saved outputs, not parameter-space gradients or the cause of training changes. Contributions are signed case-mean changes, not independent effect estimates.",
                  "The selection bank is development data. Concentrated regressions motivate hypotheses, not post-hoc gate relaxation or proof of generalization.",
                  "The unselected400 tail remains ineligible; no independent confirmation or real benchmark was consumed.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in summary.items() if key != "families"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run.resolve(), args.audit.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
