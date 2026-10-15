"""DWIP-only dispatcher with an explicit artifact budget, preserving the old queue."""
import fcntl
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

from analyze_rowcol_stopping_curves import analyze
from rowcol_fixed_validation import evaluate_fixed_validation, load_or_build_validation
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import compare_fold_metrics, evaluation_environment, gpu_processes, sha256
from run_rowcol_controlled_experiments import BENCH, REPO, WORK, write_json
from run_spectral_query_experiments import checkpoint_schedule
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable

VERSION = "dwip_artifact_budget_v1"
GIB = 1024**3
MIB = 1024**2
RESERVE_BYTES = 20 * GIB
OVERHEAD_BYTES = 2 * GIB
ANCHOR = Path("/root/tabimpute_anchor3_named/dwip_v2_t003_checkpoint_17000.pth")
SOURCES = ["scripts/run_dwip_budgeted_queue_v1.py", "scripts/run_spectral_query_experiments.py",
    "scripts/run_rowcol_controlled_experiments.py", "scripts/run_dwip_transfer_evaluation.py",
    "scripts/analyze_rowcol_stopping_curves.py", "scripts/rowcol_fixed_validation.py",
    "scripts/train_direct_pure_rabel_curriculum.py", "scripts/optimizer_continuation.py",
    "src/tabimpute/train/callbacks.py", "src/tabimpute/interface.py",
    "src/tabimpute/model/bar_distribution.py", "src/tabimpute/prior/spectrum_training_set_generation.py"]


def argument(command, flag):
    return command[command.index(flag) + 1]


def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Registered artifact changed; use fresh artifacts: {path}")
    if not path.exists():
        write_json(path, value)


def validate_command(command):
    if command[1] != "scripts/train_dwip_anchor_mix.py":
        raise ValueError("This budget only covers the reviewed DWIP anchor-mixture trainer")
    if Path(argument(command, "--init-model-checkpoint")).resolve() != ANCHOR.resolve():
        raise ValueError("Initialize from original DWIP weights only")
    if json.loads(argument(command, "--model-overrides-json")) != dwip_spec()["model_config"]:
        raise ValueError("Budget requires the unchanged DWIP architecture")
    forbidden = {"--resume", "--retain-init-optimizer-state", "--allow-partial-init"}
    if forbidden.intersection(command) or argument(command, "--precision-mode") != "amp_bfloat16":
        raise ValueError("Require strict fresh FP32 AdamW/BF16 initialization")
    phases = json.loads(argument(command, "--phases-json"))
    if len(phases) != 1 or checkpoint_schedule(command)[-1] >= 200000:
        raise ValueError("Budget covers one phase strictly below 200k steps")
    if int(argument(command, "--checkpoint-every")) <= 0 or int(argument(command, "--validation-every")) <= 0:
        raise ValueError("Require rolling and fixed validation checkpoints")


def artifact_budget(command, reference_work):
    validate_command(command)
    model_files = sorted(path for path in reference_work.glob("*.pth")
        if path.name != "latest_checkpoint.pth" and "checkpoint" in path.name)
    latest = reference_work / "latest_checkpoint.pth"
    if not model_files or not latest.is_file():
        raise ValueError("Require completed matched-run model and optimizer artifacts")
    observed_model = max(path.stat().st_size for path in model_files)
    observed_latest = latest.stat().st_size
    if observed_model <= 0 or observed_latest < observed_model:
        raise ValueError("Invalid reference checkpoint sizes")
    model_ceiling = ((observed_model * 5 + 4 * MIB - 1) // (4 * MIB)) * MIB
    latest_ceiling = max(3 * model_ceiling + MIB,
        ((observed_latest * 5 + 4 * MIB - 1) // (4 * MIB)) * MIB)
    phases = json.loads(argument(command, "--phases-json"))
    names = [f"checkpoint_{step}.pth" for step in checkpoint_schedule(command)]
    names += [f"checkpoint_phase{index}_{phase['end_step']}.pth" for index, phase in enumerate(phases, 1)]
    names += ["best_checkpoint.pth", "best_validation_checkpoint.pth"]
    persistent = len(names) * model_ceiling + latest_ceiling
    peak = persistent + latest_ceiling + OVERHEAD_BYTES
    return {"version": VERSION, "reference_work": str(reference_work),
        "reference_model_bytes": observed_model, "reference_latest_bytes": observed_latest,
        "model_file_ceiling_bytes": model_ceiling, "latest_file_ceiling_bytes": latest_ceiling,
        "model_only_files": names, "persistent_checkpoint_bytes": persistent,
        "atomic_temporary_bytes": latest_ceiling, "other_artifacts_bytes": OVERHEAD_BYTES,
        "peak_additional_bytes": peak, "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes_at_launch": peak + RESERVE_BYTES,
        "review": "Trainer snapshots plus final, one phase checkpoint, training-best, validation-best, rolling optimizer; saves are sequential atomic replacements. Callback has no model saves. Validation predictions, caches, metrics, logs and full transfer/retention outputs share2GiB allowance.",
        "headroom": "Each measured weight size rounded up to MiB with25% padding; optimizer at least3 padded model sizes plus1MiB. One largest atomic temporary, plus20GiB filesystem reserve.",
        "limits": "Conservative planning budget, not a hard filesystem quota. Poll every15s during owned subprocesses and before/after in-process evaluations; unrelated writers can race checks. Never delete artifacts or stop unrelated jobs. Frozen generic45GiB guard remains unchanged."}


def regular_bytes(path):
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, subdirectories, files in os.walk(path, followlinks=False):
        subdirectories[:] = [name for name in subdirectories if not (Path(directory) / name).is_symlink()]
        for name in files:
            item = Path(directory) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def check_storage(budget, free_bytes, allocated_bytes=0, launch=False):
    if allocated_bytes > budget["peak_additional_bytes"]:
        raise RuntimeError("DWIP artifact budget exceeded; preserve artifacts and inspect")
    remaining = budget["peak_additional_bytes"] if launch else max(
        budget["atomic_temporary_bytes"], budget["peak_additional_bytes"] - allocated_bytes)
    required = budget["reserve_bytes"] + remaining
    if free_bytes < required:
        raise RuntimeError(f"DWIP storage guard: {free_bytes} bytes free, {required} required; no files deleted")
    return {"free_bytes": free_bytes, "allocated_bytes": allocated_bytes, "required_free_bytes": required,
            "reserve_bytes": budget["reserve_bytes"], "launch": launch}


def storage_guard(root, work, budget, launch=False):
    if len({path.stat().st_dev for path in (root, WORK, BENCH)}) != 1:
        raise ValueError("Reviewed budget requires queue, work and benchmark on the same filesystem")
    benchmark = BENCH / "logs" / f"tabimpute_{root.name}_dwip1e5_mteb.npz"
    allocated = sum(regular_bytes(path) for path in (root, work, benchmark))
    result = check_storage(budget, shutil.disk_usage(WORK).free, allocated, launch)
    write_json(root / "storage_status.json", {**result, "updated_unix": time.time()})
    return result


def run_guarded(command, log_path, guard, cwd=REPO, environment=None):
    guard()
    if environment is None:
        environment = {key: value for key, value in os.environ.items() if not key.startswith("BENCHSELECT_")}
        environment.update({"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    guard()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            guard()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def run_queue(root, trials, commands, prior, anchor, budget, source_paths, dry_run=False):
    if len(trials) != 1 or trials[0]["tag"] != "dwip1e5" or set(commands) != {"dwip1e5"}:
        raise ValueError("One fixed DWIP arm only, not a sweep")
    command = commands["dwip1e5"]
    validate_command(command)
    sources = {str(path): sha256(REPO / path) for path in sorted(set(SOURCES + source_paths))}
    manifest = {"trials": trials, "commands": commands, "prior": prior,
        "evaluation_model": "TabImputeModelStable", "initial_checkpoint": str(anchor),
        "initial_checkpoint_sha256": sha256(anchor), "source_sha256": sources,
        "optimizer": "Fresh FP32 AdamW; cosine horizon defined by command",
        "selection": "Unchanged fixed synthetic macro RMSE only; all independent and real-world scores report-only",
        "steps_by_trial": {"dwip1e5": checkpoint_schedule(command)[-1]},
        "wait_for_idle_gpu": True, "dispatcher": VERSION, "storage_budget": budget,
        "limits": "One phase below200k steps; external24h timeout, preserve old queues and files"}
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_json(root / "manifest.json", manifest)
        immutable_json(root / "dwip1e5_command.json", command)
        immutable_json(root / "artifact_budget.json", budget)
        work = WORK / argument(command, "--run-name")
        if dry_run:
            storage_guard(root, work, budget, launch=True)
            print(json.dumps(manifest, indent=2))
            return
        if (root / "dwip1e5_result.json").exists():
            raise ValueError("Completed result exists; inspect rather than relaunch")

        def status(state, **extra):
            write_json(root / "status.json", {"state": state, "active": "dwip1e5",
                "completed": ["dwip1e5"] if state == "complete" else [], "updated_unix": time.time(), **extra})

        def guard():
            return storage_guard(root, work, budget)

        try:
            while active := gpu_processes():
                status("waiting_for_idle_gpu", observed_gpu_pids=active)
                time.sleep(60)
            if any(sha256(REPO / path) != expected for path, expected in sources.items()):
                raise ValueError("Registered source changed before execution")
            torch.backends.cuda.enable_cudnn_sdp(False)
            snapshots = checkpoint_schedule(command)
            if not (work / f"checkpoint_{snapshots[-1]}.pth").exists():
                if work.exists():
                    raise RuntimeError(f"Review partial training before resuming {work}")
                storage_guard(root, work, budget, launch=True)
                status("training")
                run_guarded(command, root / "dwip1e5_train.log", guard)
            status("evaluating")
            guard()
            curve = root / "curve_analysis"
            curve.mkdir(exist_ok=True)
            horizon = analyze(work, curve)
            config = json.loads(argument(command, "--model-overrides-json"))
            criterion = FullSupportBarDistribution(torch.load(
                importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
            cases = load_or_build_validation(root / "holdout.pth", prior, seed=20260911, cases_per_group=4)
            checkpoints = {str(step): work / f"checkpoint_{step}.pth" for step in snapshots}
            checkpoints["selected"] = work / "best_validation_checkpoint.pth"
            scores = {}

            def evaluate(checkpoint, samples, cache, predictions):
                guard()
                if not cache.exists():
                    model = TabImputeModelStable(**config).cuda().eval()
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
                    write_json(cache, evaluate_fixed_validation(model, criterion, samples, "cuda", "amp_bfloat16", predictions))
                    del model
                    torch.cuda.empty_cache()
                guard()
                return json.loads(cache.read_text())

            for label, checkpoint in checkpoints.items():
                scores[label] = evaluate(checkpoint, cases, root / f"dwip1e5_{label}_holdout.json",
                    root / f"dwip1e5_{label}_predictions.pth")
            general_cases = torch.load(REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth",
                map_location="cpu", weights_only=True)["cases"]
            scores["general_selected_report_only"] = evaluate(checkpoints["selected"], general_cases,
                root / "dwip1e5_general_holdout.json", root / "dwip1e5_general_predictions.pth")
            del criterion
            torch.cuda.empty_cache()
            status("evaluating_mteb")
            label = f"{root.name}_dwip1e5"
            benchmark = root / "dwip1e5_mteb.json"
            selected_npz = BENCH / "logs" / f"tabimpute_{label}_mteb.npz"
            if not benchmark.exists():
                if not selected_npz.exists():
                    run_guarded([sys.executable, "code/eval_tabimpute.py"], root / "dwip1e5_mteb.log", guard,
                        cwd=BENCH, environment=evaluation_environment(BENCH, checkpoints["selected"], label, "mteb"))
                baseline_npz = BENCH / "logs/tabimpute_dwip_spectral_20260915_baseline_mteb.npz"
                with np.load(baseline_npz, allow_pickle=True) as baseline, np.load(selected_npz, allow_pickle=True) as selected:
                    paired = compare_fold_metrics(baseline, selected)
                write_json(benchmark, {f"{method}_r2_{budget_step}": paired[f"{method}_r2_{budget_step}"]["selected"]
                    for method in ("entropy", "mi", "random") for budget_step in (5, 10, 15)})
            result = {"tag": "dwip1e5", "holdout": scores, "horizon": horizon, "mteb": json.loads(benchmark.read_text())}
            guard()
            write_json(root / "dwip1e5_result.json", result)
            write_json(root / "results.json", [result])
            status("complete")
        except Exception as error:
            status("error", error=repr(error))
            raise
