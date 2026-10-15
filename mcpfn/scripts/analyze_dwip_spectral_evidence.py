"""Read-only DWIP comparisons from cached predictions and fixed validation logs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import psutil
import torch

from analyze_rowcol_controlled_evidence import read_jsonl
from analyze_rowcol_stopping_curves import analyze


def compare_scores(baseline, candidate):
    if baseline["groups"].keys() != candidate["groups"].keys():
        raise ValueError("Evaluation groups differ")
    groups = {}
    for name, before in baseline["groups"].items():
        after = candidate["groups"][name]
        if any(before[key] != after[key] for key in ("n", "monitor_only")):
            raise ValueError(f"Evaluation population differs: {name}")
        if any(not math.isfinite(scores[key]) for scores in (before, after)
               for key in ("rmse", "nll")):
            raise ValueError(f"Nonfinite evaluation: {name}")
        groups[name] = {
            "baseline_rmse": before["rmse"], "candidate_rmse": after["rmse"],
            "rmse_delta": after["rmse"] - before["rmse"],
            "relative_rmse": after["rmse"] / before["rmse"] if before["rmse"] > 0 else None,
            "nll_delta": after["nll"] - before["nll"],
            "monitor_only": before["monitor_only"], "n": before["n"],
        }
    selected = [values for values in groups.values() if not values["monitor_only"]]
    relative = [values["relative_rmse"] for values in selected if values["relative_rmse"] is not None]
    if not selected:
        raise ValueError("No non-monitor evaluation groups")
    return {
        "baseline_macro_rmse": float(np.mean([values["baseline_rmse"] for values in selected])),
        "candidate_macro_rmse": float(np.mean([values["candidate_rmse"] for values in selected])),
        "mean_relative_group_rmse": float(np.mean(relative)) if relative else None,
        "groups_improved": sum(values["rmse_delta"] < 0 for values in selected),
        "group_count": len(selected), "groups": groups,
    }


def compare_predictions(baseline, candidate):
    before_predictions, after_predictions = baseline["predictions"], candidate["predictions"]
    if not before_predictions or before_predictions.keys() != after_predictions.keys():
        raise ValueError("Cached case identities differ or are empty")
    cases = {}
    for case_id, before in before_predictions.items():
        after = after_predictions[case_id]
        for key in ("target", "flat_indices"):
            if not torch.equal(before[key], after[key]):
                raise ValueError(f"Cached {key} differs: {case_id}")
        for prediction in (before, after):
            if prediction["mean"].shape != prediction["target"].shape or not prediction["target"].numel():
                raise ValueError(f"Invalid cached prediction shape: {case_id}")
            if not all(torch.isfinite(prediction[key]).all() for key in ("mean", "target")):
                raise ValueError(f"Nonfinite cached prediction: {case_id}")
        before_rmse = float((before["mean"].double() - before["target"].double()).square().mean().sqrt())
        after_rmse = float((after["mean"].double() - after["target"].double()).square().mean().sqrt())
        cases[case_id] = {"baseline_rmse": before_rmse, "candidate_rmse": after_rmse,
                          "rmse_delta": after_rmse - before_rmse, "n": before["target"].numel()}
    return {"case_count": len(cases), "cases_improved": sum(values["rmse_delta"] < 0 for values in cases.values()),
            "cases": cases,
            "uncertainty": "Descriptive paired differences only; masks share matrices, so cases are not independent."}


def cached_comparison(baseline_path, candidate_path):
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    return {"scores": compare_scores(baseline["metrics"], candidate["metrics"]),
            "paired_predictions": compare_predictions(baseline, candidate)}


def collect(root, work):
    training = read_jsonl(work / "metrics_stream.jsonl")
    validation = read_jsonl(work / "validation_stream.jsonl")
    diagnostics = [row for row in training if "logit_gradient_norm/nll" in row]
    live = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = process.info["cmdline"] or []
            if work.name in command and any(arg.endswith("train_dwip_spectral_geometry.py") for arg in command):
                live.append({"pid": process.pid, "created_unix": process.info["create_time"]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(),
              "queue_state": json.loads((root / "status.json").read_text()),
              "live_trainers": live, "holdouts": {}, "work": str(work),
              "selection": "Unchanged synthetic selector; all comparisons here are report-only."}
    if training:
        report["training"] = {
            "last_step": training[-1]["step"], "last_loss_ema": training[-1]["loss_missing_ema"],
            "last_lr": training[-1]["lr"], "sample_count": len(training),
            "nonfinite_sampled_steps": [row["step"] for row in training if any(
                not math.isfinite(row[key]) for key in ("loss_missing", "nll_missing", "grad_norm"))],
            "sampled_clip_fraction": float(np.mean([row["grad_was_clipped"] for row in training])),
            "latest_update_diagnostics": {key: value for key, value in (diagnostics[-1] if diagnostics else {}).items()
                if key == "step" or key.startswith(("sampled_update_ratio/", "logit_gradient"))},
        }
    if validation:
        report["validation"] = {"latest_step": validation[-1]["step"],
            "best_step": min(validation, key=lambda row: row["macro_rmse"])["step"],
            "latest_vs_initial": compare_scores(validation[0], validation[-1])}
        if len(validation) > 1:
            curves = root / "curve_analysis"
            curves.mkdir(exist_ok=True)
            report["horizon"] = analyze(work, curves)
    for path in sorted(root.glob("dwip1e5_*_predictions.pth")):
        label = path.name.removeprefix("dwip1e5_").removesuffix("_predictions.pth")
        if label.isdigit() or label == "selected":
            report["holdouts"][f"spectral_{label}"] = cached_comparison(root / "baseline_spectral_predictions.pth", path)
    general = root / "dwip1e5_general_predictions.pth"
    if general.exists():
        report["holdouts"]["general_selected"] = cached_comparison(root / "baseline_general_predictions.pth", general)
    baseline = root / "baseline_mteb.json"
    if baseline.exists():
        report["mteb_report_only"] = {"baseline": json.loads(baseline.read_text()), "reference_mi_r2_5": .7620}
        candidate = root / "dwip1e5_mteb.json"
        if candidate.exists():
            scores = json.loads(candidate.read_text())
            report["mteb_report_only"].update(candidate=scores,
                delta={key: value - report["mteb_report_only"]["baseline"][key] for key, value in scores.items()},
                mi_r2_5_gap_to_reference=scores["mi_r2_5"] - .7620)
    transfer = root / "transfer_evaluation"
    if (transfer / "status.json").exists():
        report["additional_realworld_report_only"] = {
            "status": json.loads((transfer / "status.json").read_text()),
            "results": json.loads((transfer / "results.json").read_text()) if (transfer / "results.json").exists() else {},
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="dwip_spectral_20260915")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "runs/objective_queues" / args.queue
    work = repo / "src/tabimpute/workdir" / f"tabimpute-{args.queue}-dwip1e5"
    report = collect(root, work)
    path = root / "evidence_comparison.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)
    print(json.dumps({"path": str(path), "step": report.get("training", {}).get("last_step"),
                      "live_trainers": report["live_trainers"], "holdouts": list(report["holdouts"])}))


if __name__ == "__main__":
    main()
