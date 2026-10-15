#!/usr/bin/env python
"""Matched magnitude/translated-density trials with cached independent probes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from probe_rowcol_solver_decoder import run_probe
from rowcol_fixed_validation import load_or_build_validation
from run_rowcol_controlled_experiments import REPO, WORK, BENCH, run_logged, write_json


def replace_arg(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="rowcol_output_remedies_paired_20260907")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    previous = REPO / "runs/objective_queues/rowcol_controlled_20260906"
    template = json.loads((previous / "mse_broad_support_command.json").read_text())
    initial = Path(template[template.index("--init-model-checkpoint") + 1])
    config = json.loads(template[template.index("--model-overrides-json") + 1])
    prior = json.loads(template[template.index("--prior-overrides-json") + 1])
    trials = [{"tag": f"{variant}_lr{lr:g}", "magnitude": magnitude, "correction": correction, "lr": lr}
              for lr in (6e-6, 3e-5)
              for variant, magnitude, correction in (("control", False, False),
                  ("magnitude", True, False), ("correction", False, True), ("both", True, True))]
    source_files = ["src/tabimpute/model/model_rowcol_svt_hybrid.py",
        "src/tabimpute/model/prediction_distribution.py", "src/tabimpute/interface.py",
        "scripts/train_direct_pure_rabel_curriculum.py", "scripts/rowcol_fixed_validation.py",
        "scripts/probe_rowcol_solver_decoder.py", "scripts/run_rowcol_output_remedies.py"]
    manifest = {"initial_checkpoint": str(initial), "steps_per_run": 12000, "trials": trials,
        "training": "Matched data/model seed 1201, broad support, AMP + FP32 SVT, cosine floor 5%",
        "objective": "NLL 1 + MSE .2 + CDF .05 + depth distill .05 + CV regret .05",
        "selection": "Original fixed synthetic macro RMSE. Independent cache and MTEB reporting only.",
        "controls": "Reuse completed matched mse_broad_support and lr_followup_2 checkpoints/MTEB; new holdout probes only.",
        "holdout": {"seed": 20260907, "cases_per_group": 4},
        "conditional_followups": "Initialization supervision/short differentiable suffix only after paired ablations.",
        "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in source_files}}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Manifest/source changed; use a new queue name")
    write_json(path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import fcntl
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.backends.cuda.enable_cudnn_sdp(False)
    holdout = root / "independent_validation.pth"
    completed = []
    active = "independent_validation"

    def status(state, **extra):
        write_json(root / "status.json", {"state": state, "active": active,
                   "completed": [r["tag"] for r in completed], "updated_unix": time.time(), **extra})

    try:
        status("preflight")
        load_or_build_validation(holdout, prior, seed=20260907, cases_per_group=4)
        active = "paired_initial_checkpoint_probes"
        status("probing")
        preflight = root / "initial_probes"
        if not (preflight / "results.json").exists():
            run_probe(initial, holdout, preflight, model_overrides=config)
        for trial in trials:
            active = trial["tag"]
            status("training")
            if shutil.disk_usage(WORK).free < 40 * 1024**3:
                raise RuntimeError("Less than 40 GiB free. Queue paused without deleting files.")
            command = list(template)
            name = f"tabimpute-{args.queue_name}-{trial['tag']}"
            work = WORK / name
            reused_tag = None
            if not trial["magnitude"] and not trial["correction"]:
                reused_tag = "mse_broad_support" if trial["lr"] == 6e-6 else "lr_followup_2"
                work = WORK / f"tabimpute-rowcol_controlled_20260906-{reused_tag}"
                if not (work / "checkpoint_12000.pth").exists():
                    raise RuntimeError(f"Missing completed matched control: {work}")
            model_config = {**config, "use_svt_magnitude_channel": trial["magnitude"],
                            "use_svt_output_correction": trial["correction"]}
            replace_arg(command, "--run-name", name)
            replace_arg(command, "--model-overrides-json", json.dumps(model_config))
            replace_arg(command, "--lr", trial["lr"])
            if trial["magnitude"] or trial["correction"]:
                command.append("--allow-partial-init")
            write_json(root / f"{active}_command.json", command)
            if not (work / "checkpoint_12000.pth").exists():
                if work.exists():
                    raise RuntimeError(f"Partial run {work}; inspect before explicitly resuming")
                run_logged(command, root / f"{active}_train.log")
            checkpoint = work / "best_validation_checkpoint.pth"
            status("probing")
            probe_dir = root / f"{active}_probes"
            if not (probe_dir / "results.json").exists():
                run_probe(checkpoint, holdout, probe_dir)
            status("evaluating_mteb")
            done = root / f"{active}_mteb.json"
            if reused_tag is not None and not done.exists():
                write_json(done, json.loads((previous / f"{reused_tag}_eval_done.json").read_text()))
            if not done.exists():
                label = f"{args.queue_name}_{active}"
                run_logged([sys.executable, "code/eval_tabimpute.py"], root / f"{active}_mteb.log",
                    cwd=BENCH, extra_env={"BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint",
                    "BENCHSELECT_TABIMPUTE_LABEL": label, "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(checkpoint),
                    "BENCHSELECT_TABIMPUTE_DEVICE": "cuda", "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
                    "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean", "BENCHSELECT_DATASETS": "mteb",
                    "BENCHSELECT_METHODS": "entropy,mi,random", "BENCHSELECT_N_FOLDS": "10",
                    "BENCHSELECT_K_MAX": "15", "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
                with np.load(BENCH / "logs" / f"tabimpute_{label}_mteb.npz") as data:
                    scores = {f"{m}_r2_{k}": float(np.nanmean(data[f"{m}_r2"][:, k - 1]))
                              for m in ("entropy", "mi", "random") for k in (5, 10, 15)}
                write_json(done, scores)
            probe = json.loads((probe_dir / "results.json").read_text())["ablations"]["full"]
            baseline = json.loads((preflight / "results.json").read_text())["ablations"]["full"]
            ratios = {g: v["rmse"] / max(baseline["groups"][g]["rmse"], 1e-12)
                      for g, v in probe["groups"].items() if not v["monitor_only"]}
            completed.append({**trial, "work_dir": str(work),
                "reused_control": reused_tag,
                "independent_macro_rmse": probe["macro_rmse"],
                "independent_baseline_relative_macro_rmse": float(np.mean(list(ratios.values()))),
                "independent_group_rmse_ratios": ratios,
                "validation": json.loads((work / "best_validation_metrics.json").read_text()),
                "mteb": json.loads(done.read_text())})
            write_json(root / "results.json", completed)
        active = "complete"
        status("complete")
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
