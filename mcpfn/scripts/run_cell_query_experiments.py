"""Matched 36k cosine runs, checkpoint probes and cached real-world evaluation."""
import argparse
import hashlib
import importlib.resources
import json
import shutil
import sys
import time

import numpy as np
import torch

from analyze_rowcol_stopping_curves import analyze
from probe_rowcol_solver_decoder import run_probe
from rowcol_fixed_validation import evaluate_fixed_validation
from run_rowcol_controlled_experiments import REPO, WORK, BENCH, run_logged, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="cell_query_20260909")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-reviewed", action="store_true",
                        help="Recover a verified stopped queue; preserve original manifest and archive rolled-back logs")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    previous = REPO / "runs/objective_queues/rowcol_controlled_20260906"
    template = json.loads((previous / "mse_broad_support_command.json").read_text())
    base_config = json.loads(template[template.index("--model-overrides-json") + 1])
    trials = [{"tag": "query2", "layers": 2}, {"tag": "control", "layers": 0},
              {"tag": "query1", "layers": 1}]
    holdout = REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"
    sources = ["scripts/run_cell_query_experiments.py", "scripts/train_direct_pure_rabel_curriculum.py",
        "scripts/probe_rowcol_solver_decoder.py", "scripts/rowcol_fixed_validation.py",
        "src/tabimpute/model/model_rowcol_svt_hybrid.py", "src/tabimpute/model/cell_query_decoder.py",
        "src/tabimpute/interface.py"]
    manifest = {"trials": trials, "steps": 36000, "snapshot_steps": [12000, 24000, 36000],
        "lr": 6e-6, "warmup": 600, "cosine_decay": 35400, "floor": .05,
        "initial_checkpoint": template[template.index("--init-model-checkpoint") + 1],
        "objective": "Same NLL=1 MSE=.2 CDF=.05 depth=.05 CV regret=.05 as old control",
        "seed": 1201, "data_seed": 701201, "mask": "broad_support",
        "selection": "Original synthetic macro-RMSE only; per-group, independent holdout and MTEB reporting",
        "holdout": str(holdout), "template": template,
        "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources}}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        old = json.loads(path.read_text())
        if not args.resume_reviewed or {k: v for k, v in old.items() if k != "source_sha256"} != {
                k: v for k, v in manifest.items() if k != "source_sha256"}:
            raise ValueError("Queue source/config changed; use a new name")
        write_json(root / f"recovery_sources_{time.time_ns()}.json", manifest)
    else:
        write_json(path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import fcntl
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.backends.cuda.enable_cudnn_sdp(False)
    results = []
    active = "initializing"
    def status(state, **extra):
        write_json(root / "status.json", {"state": state, "active": active,
                   "completed": [r["tag"] for r in results], "updated_unix": time.time(), **extra})
    try:
        for trial in trials:
            active = trial["tag"]
            status("training")
            if shutil.disk_usage(WORK).free < 40 * 1024**3:
                raise RuntimeError("Less than 40 GiB free; no files deleted")
            name = f"tabimpute-{args.queue_name}-{active}"
            work = WORK / name
            config = {**base_config, "use_cell_query_decoder": bool(trial["layers"]),
                "cell_query_layers": max(1, trial["layers"]), "cell_query_width": 128,
                "cell_query_heads": 4, "cell_query_chunk_size": 128, "cell_query_checkpoint_chunks": True,
                "use_svt_output_correction": False, "use_svt_magnitude_channel": False}
            command = list(template)
            phases = json.loads(command[command.index("--phases-json") + 1])
            phases[-1]["end_step"] = 36000
            for flag, value in (("--run-name", name), ("--phases-json", json.dumps(phases)),
                                ("--model-overrides-json", json.dumps(config)), ("--decay-steps", 35400)):
                replace_arg(command, flag, value)
            command += ["--snapshot-steps", "12000", "24000"]
            if trial["layers"]:
                command += ["--allow-partial-init"]
            write_json(root / f"{active}_command.json", command)
            if not (work / "checkpoint_36000.pth").exists():
                if work.exists():
                    if not args.resume_reviewed:
                        raise RuntimeError(f"Partial run {work}; review before explicitly resuming")
                    latest = work / "latest_checkpoint.pth"
                    saved = torch.load(latest, map_location="cpu", weights_only=True)
                    if not all(key in saved for key in ("optimizer", "scheduler", "epoch", "model")):
                        raise ValueError("Recovery requires complete optimizer/scheduler state")
                    step = int(saved["epoch"])
                    if not 0 < step < 36000:
                        raise ValueError(f"Unexpected recovery step: {step}")
                    del saved
                    archive = work / f"interruption_{time.time_ns()}"
                    archive.mkdir()
                    for filename in ("metrics_stream.jsonl", "validation_stream.jsonl", "config.json"):
                        source = work / filename
                        if source.exists():
                            shutil.copy2(source, archive / filename)
                            if filename.endswith(".jsonl"):
                                records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
                                source.write_text("".join(json.dumps(record) + "\n" for record in records
                                    if int(record["step"]) < step))
                    i = command.index("--init-model-checkpoint")
                    del command[i:i + 2]
                    command += ["--resume", str(latest)]
                    write_json(root / f"{active}_recovery_command.json", command)
                run_logged(command, root / f"{active}_train.log")
            status("evaluating_horizon")
            curve_dir = root / "curve_analysis"
            curve_dir.mkdir(exist_ok=True)
            horizon = analyze(work, curve_dir)
            write_json(root / f"{active}_horizon.json", horizon)
            cases = torch.load(holdout, map_location="cpu", weights_only=True)["cases"]
            criterion = FullSupportBarDistribution(torch.load(
                importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
            checkpoints = {str(s): work / f"checkpoint_{s}.pth" for s in (12000, 24000, 36000)}
            checkpoints["selected"] = work / "best_validation_checkpoint.pth"
            scores = {}
            for label, checkpoint in checkpoints.items():
                metric_path = root / f"{active}_{label}_holdout.json"
                if not metric_path.exists():
                    model = TabImputeModelRowColumnSVTHybrid(**config).cuda().eval()
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                    metrics = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                        root / f"{active}_{label}_holdout_predictions.pth")
                    write_json(metric_path, metrics)
                    del model
                    torch.cuda.empty_cache()
                scores[label] = json.loads(metric_path.read_text())
            selected = checkpoints["selected"]
            status("probing")
            probe_dir = root / f"{active}_probes"
            if not (probe_dir / "results.json").exists():
                run_probe(selected, holdout, probe_dir)
            status("evaluating_mteb")
            done = root / f"{active}_mteb.json"
            if not done.exists():
                label = f"{args.queue_name}_{active}"
                run_logged([sys.executable, "code/eval_tabimpute.py"], root / f"{active}_mteb.log",
                    cwd=BENCH, extra_env={"BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint",
                        "BENCHSELECT_TABIMPUTE_LABEL": label, "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(selected),
                        "BENCHSELECT_TABIMPUTE_DEVICE": "cuda", "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
                        "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean", "BENCHSELECT_DATASETS": "mteb",
                        "BENCHSELECT_METHODS": "entropy,mi,random", "BENCHSELECT_N_FOLDS": "10",
                        "BENCHSELECT_K_MAX": "15", "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
                with np.load(BENCH / "logs" / f"tabimpute_{label}_mteb.npz") as data:
                    write_json(done, {f"{m}_r2_{k}": float(np.nanmean(data[f"{m}_r2"][:, k - 1]))
                                     for m in ("entropy", "mi", "random") for k in (5, 10, 15)})
            results.append({**trial, "work_dir": str(work), "holdout": scores,
                "validation": json.loads((work / "best_validation_metrics.json").read_text()),
                "horizon": horizon, "mteb": json.loads(done.read_text())})
            write_json(root / "results.json", results)
        active = "complete"
        status("complete")
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
