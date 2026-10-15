#!/usr/bin/env python
"""Matched objective/support experiments, followed by validation-selected LR trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from rowcol_fixed_validation import evaluate_fixed_validation, load_or_build_validation
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid
from train_direct_pure_rabel_curriculum import PRIOR_CONFIG

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "src/tabimpute/workdir"
BENCH = REPO.parent / "benchmark-selection"
INIT = WORK / "tabimpute-direct-rowcol-svthybrid768-widehead4096-generalmix-30k-fromobservedattn/best_checkpoint.pth"
ANCHOR = WORK / "tabimpute-direct-rowcol-svthybrid768-amp_precision_only-12000steps/best_checkpoint.pth"


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True))
    temp.replace(path)


def run_logged(command, path, cwd=REPO, extra_env=None):
    env = {**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    env.update(extra_env or {})
    print(f"[queue] log={path}", flush=True)
    with path.open("a") as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def precision_preflight(root, cases):
    summary_path = root / "precision_preflight.json"
    if summary_path.exists():
        return
    cfg = json.loads((ANCHOR.parent / "config.json").read_text())["model"]
    state = torch.load(ANCHOR, map_location="cpu", weights_only=True)["model"]
    borders = torch.load(importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True)
    criterion = FullSupportBarDistribution(borders.cuda())
    results = {}
    for mode in ("float32", "amp_bfloat16", "full_bfloat16"):
        print(f"[queue] precision preflight: {mode}", flush=True)
        model = TabImputeModelRowColumnSVTHybrid(**cfg)
        model.load_state_dict(state, strict=True)
        model.to(device="cuda", dtype=torch.bfloat16 if mode == "full_bfloat16" else torch.float32)
        predictions = root / f"precision_{mode}_predictions.pth"
        results[mode] = evaluate_fixed_validation(model, criterion, cases, "cuda", mode, predictions)
        del model
        torch.cuda.empty_cache()
    reference = torch.load(root / "precision_float32_predictions.pth", weights_only=True)["predictions"]
    for mode in ("amp_bfloat16", "full_bfloat16"):
        pred = torch.load(root / f"precision_{mode}_predictions.pth", weights_only=True)["predictions"]
        diff = torch.cat([(pred[k]["mean"] - reference[k]["mean"]).flatten() for k in reference])
        results[mode]["prediction_rmse_vs_fp32"] = float(diff.square().mean().sqrt())
        results[mode]["prediction_max_abs_vs_fp32"] = float(diff.abs().max())
    write_json(summary_path, {"checkpoint": str(ANCHOR), "results": results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="rowcol_controlled_20260906")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1000 <= args.steps <= 200000:
        raise ValueError("Each run must be between 1000 and 200000 steps")
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    initial_config = json.loads((INIT.parent / "config.json").read_text())
    model_config = {**initial_config["model"], "svt_solver_precision": "float32",
                    "svt_mixture_cv_reencode": True}
    prior_config = {**PRIOR_CONFIG, **initial_config["prior"]}
    cache = root / "fixed_validation.pth"
    trials = [dict(tag=f"{loss}_{mask}", loss=loss, mask=mask, lr=6e-6, seed=1201)
              for mask in ("mixed", "broad_support") for loss in ("huber", "mse")]
    phases = [dict(name="stationary_rectangles", end_step=args.steps,
                   rows_low=80, rows_high=320, cols_low=35, cols_high=100, batch_size=2)]
    manifest = {"steps": args.steps, "initial_checkpoint": str(INIT), "prior": prior_config,
                "model": model_config, "phases": phases, "trials": trials,
                "validation_seed": 20260906, "selection": "synthetic macro_rmse only",
                "followups": "best objective/mask at LR 3e-6, 1e-5, 3e-5; then best recipe seed 1202",
                "objective": "NLL=1, point=0.2 (Huber delta=.05 or MSE), CDF=.05, distill=.05, CV regret=.05",
                "schedule": "600 warmup, remaining cosine decay, floor 5%"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing queue manifest differs; choose a new queue name")
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import fcntl
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.backends.cuda.enable_cudnn_sdp(False)
    status_path = root / "status.json"
    results = []
    active = "preflight"

    def status(state, **extra):
        write_json(status_path, {"state": state, "active": active, "updated_unix": time.time(),
                                 "completed": [r["tag"] for r in results], **extra})

    def trial_run(trial):
        nonlocal active
        active = trial["tag"]
        status("training")
        if shutil.disk_usage(WORK).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB disk free; stopping queue before next job")
        name = f"tabimpute-{args.queue_name}-{trial['tag']}"
        work = WORK / name
        final = work / f"checkpoint_{args.steps}.pth"
        warmup = min(600, args.steps // 10)
        command = [sys.executable, "scripts/train_direct_rowcol_svt_hybrid_random_rank_curriculum.py",
                   "--run-name", name, "--init-model-checkpoint", str(INIT),
                   "--model-overrides-json", json.dumps(model_config),
                   "--prior-overrides-json", json.dumps(prior_config),
                   "--prior-kind", "latent_factor", "--generator-type", "generator_mix",
                   "--missingness-type", "profile_mixed", "--phases-json", json.dumps(phases),
                   "--precision-mode", "amp_bfloat16", "--seed", str(trial["seed"]),
                   "--data-seed", str(700000 + trial["seed"]), "--support-mask-mode", trial["mask"],
                   "--lr", str(trial["lr"]), "--weight-decay", "0.02", "--grad-clip-norm", "0.5",
                   "--warmup-steps", str(warmup), "--stable-steps", "0",
                   "--decay-steps", str(args.steps - warmup), "--min-lr-ratio", "0.05",
                   "--nll-loss-weight", "1", "--point-loss-weight", "0.2",
                   "--point-loss-kind", trial["loss"], "--point-loss-prediction", "mean",
                   "--huber-delta", "0.05", "--cdf-loss-weight", "0.05",
                   "--svt-mixture-depth-distill-weight", "0.05",
                   "--svt-mixture-depth-distill-temperature", "0.08",
                   "--svt-mixture-cv-regret-weight", "0.05",
                   "--distribution-loss-chunk-size", "2048",
                   "--fixed-validation-cache", str(cache), "--validation-every", "2000",
                   "--validation-seed", "20260906", "--checkpoint-every", "2000",
                   "--metrics-jsonl-every", "25", "--update-diagnostics-every", "500"]
        write_json(root / f"{trial['tag']}_command.json", command)
        if not final.exists():
            if (work / "metrics_stream.jsonl").exists():
                raise RuntimeError(f"Partial run {work}; inspect before explicitly resuming")
            run_logged(command, root / f"{trial['tag']}_train.log")
        validation = json.loads((work / "best_validation_metrics.json").read_text())
        result = {**trial, "validation_rmse": validation["macro_rmse"],
                  "validation_step": validation["step"], "work_dir": str(work)}
        # Real-world scores are recorded after synthetic selection, never used to choose trials.
        status("evaluating")
        eval_done = root / f"{trial['tag']}_eval_done.json"
        if not eval_done.exists():
            label = f"{args.queue_name}_{trial['tag']}_amp"
            run_logged([sys.executable, "code/eval_tabimpute.py"], root / f"{trial['tag']}_mteb.log",
                       cwd=BENCH, extra_env={
                           "BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint",
                           "BENCHSELECT_TABIMPUTE_LABEL": label,
                           "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(work / "best_validation_checkpoint.pth"),
                           "BENCHSELECT_TABIMPUTE_DEVICE": "cuda",
                           "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
                           "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean",
                           "BENCHSELECT_DATASETS": "mteb", "BENCHSELECT_METHODS": "entropy,mi,random",
                           "BENCHSELECT_N_FOLDS": "10", "BENCHSELECT_K_MAX": "15",
                           "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
            data = np.load(BENCH / "logs" / f"tabimpute_{label}_mteb.npz")
            benchmark = {f"{m}_r2_{k}": float(np.nanmean(data[f"{m}_r2"][:, k-1]))
                         for m in ("entropy", "mi", "random") for k in (5, 10, 15)}
            write_json(eval_done, benchmark)
        result.update(json.loads(eval_done.read_text()))
        diag_done = root / f"{trial['tag']}_diagnostic_done.json"
        if not diag_done.exists():
            status("diagnosing")
            run_logged([sys.executable, "scripts/diagnose_dwip_vs_rowcol_mteb_information_flow.py",
                        "--spec", f"{trial['tag']}|{work / 'best_validation_checkpoint.pth'}|rowcol_svt_hybrid",
                        "--dataset", "mteb", "--fold", "0", "--k-selected", "5",
                        "--max-targets", "1024", "--precision-mode", "amp_bfloat16",
                        "--output-dir", str(root / "diagnostics"),
                        "--run-label", trial["tag"]], root / f"{trial['tag']}_diagnostic.log")
            write_json(diag_done, {"complete": True})
        results.append(result)
        write_json(root / "results.json", results)
        with (root / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)
        print(f"[queue] completed {trial['tag']}: validation RMSE={validation['macro_rmse']:.6f}", flush=True)
        return result

    try:
        status("preflight")
        cases = load_or_build_validation(cache, prior_config)
        precision_preflight(root, cases)
        del cases
        torch.cuda.empty_cache()
        source_paths = [Path(__file__), REPO / "scripts/rowcol_fixed_validation.py",
                        REPO / "scripts/train_direct_pure_rabel_curriculum.py",
                        REPO / "src/tabimpute/model/model_rowcol_svt_hybrid.py",
                        REPO / "src/tabimpute/prior/training_set_generation.py", REPO / "src/tabimpute/interface.py",
                        BENCH / "code/eval_tabimpute.py",
                        REPO / "scripts/diagnose_dwip_vs_rowcol_mteb_information_flow.py"]
        write_json(root / "source_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
        for trial in trials:
            trial_run(trial)
        winner = min(results, key=lambda r: r["validation_rmse"])
        for index, lr in enumerate((3e-6, 1e-5, 3e-5)):
            trial_run(dict(tag=f"lr_followup_{index}", loss=winner["loss"], mask=winner["mask"], lr=lr, seed=1201))
        winner = min(results, key=lambda r: r["validation_rmse"])
        trial_run(dict(tag="winner_replication", loss=winner["loss"], mask=winner["mask"],
                       lr=winner["lr"], seed=1202))
        active = "complete"
        status("complete", best=min(results, key=lambda r: r["validation_rmse"]))
    except Exception as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
