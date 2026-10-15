"""Inspect fixed residual tails before considering a robust point objective."""
import argparse
import importlib.resources
import json
import math
from pathlib import Path

import torch

from analyze_rowcol_controlled_evidence import read_jsonl
from diagnose_dwip_matched_objective_gradients import verify_hashes
from diagnose_dwip_training_objective import prepare_training_batch
from general_spectral_geometry import install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from train_direct_pure_rabel_curriculum import CurriculumPhase
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def residual_influence(errors):
    errors = errors.detach().double().flatten()
    if not errors.numel() or not torch.isfinite(errors).all():
        raise ValueError("Require finite nonempty residuals")
    absolute = errors.abs()
    mse_derivative = 2 * errors
    huber_derivative = errors.clamp(-1, 1)
    result = {"count": errors.numel(), "mse": float(errors.square().mean()),
        "huber_delta1": float(torch.where(absolute <= 1, .5 * errors.square(), absolute - .5).mean()),
        "fraction_absolute_error_above1": float((absolute > 1).double().mean()),
        "max_absolute_error": float(absolute.max())}
    for label, derivative in (("mse", mse_derivative), ("huber_delta1", huber_derivative)):
        squared = derivative.square()
        total = float(squared.sum())
        result[f"{label}_derivative_norm"] = math.sqrt(total) / errors.numel()
        for fraction in (.01, .05):
            count = max(1, math.ceil(errors.numel() * fraction))
            result[f"{label}_top{int(100 * fraction)}percent_derivative_energy_fraction"] = float(squared.topk(count).values.sum()) / total if total else 0.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if gpu_processes():
        raise RuntimeError("GPU busy")
    previous = REPO / "runs/objective_queues" / args.diagnostic
    if json.loads((previous / "status.json").read_text())["state"] != "complete":
        raise ValueError("Complete the registered training-shaped diagnostic first")
    registration = json.loads((previous / "registration.json").read_text())
    verify_hashes(registration["input_sha256"])
    control_stream = WORK / f"tabimpute-{args.control}-dwip1e5/metrics_stream.jsonl"
    rows = read_jsonl(control_stream)
    largest = max(rows, key=lambda row: row["point_loss_missing"])
    spectral = max((row for row in rows if not row["sampled_prior_name"].startswith("anchor_")), key=lambda row: row["point_loss_missing"])
    targeted = {row["step"]: row for row in (largest, spectral)}
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    if (output / "registration.json").exists():
        raise ValueError("Preserve existing diagnostic")
    cache = previous / "cases.pth"
    if sha256(cache) != json.loads((previous / "cases_manifest.json").read_text())["sha256"]:
        raise ValueError("Frozen32-batch cache changed")
    inputs = {**registration["input_sha256"], str(cache): sha256(cache), str(control_stream): sha256(control_stream),
              str(Path(__file__)): sha256(__file__), str(previous / "registration.json"): sha256(previous / "registration.json")}
    manifest = {"input_sha256": inputs, "targeted_steps": sorted(targeted), "huber_delta": 1.,
        "targeted_rule": "Post-hoc largest overall and largest spectral sampled MSE in original-loss64k TRAINING stream; not selected from any benchmark or held-out scores.",
        "scope": "All32 previously frozen training-shaped batches, plus two targeted stress batches. Compare terminal64k models, not historical step-specific predictions. No updates or objective sweep.",
        "limits": "Prediction-space derivative energy, not parameter gradient contribution or Adam weighting. Huber1 has half MSE curvature inside the unit region; no coefficient compensation. Robustness may bias conditional means and harm RMSE/R2. Targeted stress cases do not estimate prevalence."}
    write_json(output / "registration.json", manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    install_geometry_generator()
    try:
        cases = torch.load(cache, map_location="cpu", weights_only=True)
        cases = [{**case, "cohort": "frozen32"} for case in cases]
        with preserved_rng(registration["prior_init_seed"]):
            generator = ShapeAwareMixedMatrixPrior(CurriculumPhase(**registration["phase"]), "cuda", "profile_mixed", registration["prior"])
        for step, row in sorted(targeted.items()):
            with preserved_rng(registration["prior_init_seed"] + step):
                (source, target, _, _, _), _ = generator.get_batch()
            source, target, mask, stats = prepare_training_batch(source, target, registration,
                registration["prior_init_seed"] + 10000000 + step)
            if generator.last_prior_name != row["sampled_prior_name"] or list(source.shape[1:]) != [row["matrix_rows"], row["matrix_cols"]]:
                raise ValueError("Targeted historical training batch identity differs")
            if any(value != row.get(key) for key, value in stats.items()):
                raise ValueError("Targeted historical support statistics differ")
            cases.append({"id": step, "family": generator.last_prior_name, "cohort": "targeted_training_tail",
                          "source": source.cpu(), "target": target.cpu(), "mask": mask.cpu(), "support": stats})
        torch.save(cases, output / "cases.pth")
        write_json(output / "cases_sha256.json", {"sha256": sha256(output / "cases.pth")})
        criterion = FullSupportBarDistribution(torch.load(importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
        results = {}
        for label in ("control", "candidate"):
            checkpoint = Path(registration["checkpoint_paths"][label])
            model = DwipTrainingAdapter(**registration["model_config"]).cuda().train()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            records = []
            for case in cases:
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask]
                with torch.no_grad(), preserved_rng(1201 + case["id"]), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask)
                errors = criterion.mean(logits.float()) - target.float()
                records.append({"id": case["id"], "family": case["family"], "cohort": case["cohort"],
                    "shape": list(source.shape), "support": case["support"], "residuals": residual_influence(errors)})
                del logits, errors
            results[label] = records
            verify_hashes(inputs)
            write_json(output / "results.json", results)
            del model
            torch.cuda.empty_cache()
        write_json(output / "status.json", {"state": "complete", "cases_per_model": len(cases)})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
