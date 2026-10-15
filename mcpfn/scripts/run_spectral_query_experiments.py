"""Controlled generic-spectrum continuations; benchmark results never select checkpoints."""
import argparse
import hashlib
import importlib.resources
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from analyze_rowcol_stopping_curves import analyze
from rowcol_fixed_validation import load_or_build_validation, evaluate_fixed_validation
from run_rowcol_controlled_experiments import REPO, WORK, BENCH, run_logged, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid

MODEL_CLASS = TabImputeModelRowColumnSVTHybrid
ADDITIONAL_SOURCES = []

def spectral_config():
    common = dict(spectrum_target_rank=10000, spectrum_rank_low=2, spectrum_rank_high=160,
        spectrum_scale_low=1., spectrum_scale_high=5., spectrum_noise_std_low=0.,
        spectrum_noise_std_high=.001, spectrum_head_weight_low=0., spectrum_head_weight_high=3.,
        spectrum_head_exp_scale_low=1., spectrum_head_exp_scale_high=30.,
        spectrum_head_exp_power_low=1., spectrum_head_exp_power_high=3.,
        spectrum_head_handoff_scale_low=1., spectrum_head_handoff_scale_high=20.,
        spectrum_tail_amplitude_low=.2, spectrum_tail_amplitude_high=1.,
        spectrum_taper_strength_low=0., spectrum_taper_strength_high=1.,
        spectrum_taper_power_low=1., spectrum_taper_power_high=4., p_missing=.4)
    families = {
        "low_rank": dict(spectrum_rank_low=2, spectrum_rank_high=16,
                         spectrum_alpha_low=.2, spectrum_alpha_high=2.),
        "slow_tail": dict(spectrum_alpha_low=0., spectrum_alpha_high=.5),
        "steep_tail": dict(spectrum_alpha_low=1., spectrum_alpha_high=2.5),
    }
    specs = [{"name": f"{name}_{mechanism}", "kind": "spectrum_decay", "weight": 1.,
              "missingness_type": mechanism, "config": {**common, **config}}
             for name, config in families.items() for mechanism in ("mcar", "profile_mixed")]
    # Intermediate exponents are excluded from the high-rank training families.
    validation = {name: {**common, **config, "missingness_type": mechanism}
        for name, config, mechanism in (
            ("spectral_low", families["low_rank"], "mcar"),
            ("spectral_mid", dict(spectrum_alpha_low=.6, spectrum_alpha_high=.9), "mcar"),
            ("spectral_steep", families["steep_tail"], "profile_mixed"))}
    return {"mixed_priors": specs, "validation_spectral_specs": validation}


def build_commands(root):
    previous = REPO / "runs/objective_queues/cell_query_20260909_checked"
    template = json.loads((previous / "query2_command.json").read_text())
    config = json.loads(template[template.index("--model-overrides-json") + 1])
    anchor = WORK / "tabimpute-cell_query_20260909_checked-query2/checkpoint_12000.pth"
    trials = [dict(tag="joint_low", lr=2e-6), dict(tag="frozen_low", lr=2e-6, freeze=True),
              dict(tag="axis_low", lr=2e-6, balanced=True), dict(tag="joint_reference", lr=6e-6)]
    prior = spectral_config()
    commands = {}
    for trial in trials:
        command = list(template)
        # Replace variable-length snapshot option inherited from the earlier queue.
        i = command.index("--snapshot-steps")
        j = i + 1
        while j < len(command) and not command[j].startswith("--"):
            j += 1
        del command[i:j]
        if "--allow-partial-init" in command:
            command.remove("--allow-partial-init")
        cfg = {**config, "cell_query_axis_balanced": bool(trial.get("balanced"))}
        phases = [dict(name="stationary_spectral", end_step=24000, rows_low=80,
                       rows_high=320, cols_low=35, cols_high=100, batch_size=2)]
        replacements = {"--run-name": f"tabimpute-{root.name}-{trial['tag']}",
            "--init-model-checkpoint": str(anchor), "--model-overrides-json": json.dumps(cfg),
            "--prior-kind": "mixed_matrix", "--generator-type": "latent_factor",
            "--prior-overrides-json": json.dumps(prior), "--phases-json": json.dumps(phases),
            "--lr": trial["lr"], "--warmup-steps": 600, "--decay-steps": 23400,
            "--data-seed": 901210, "--fixed-validation-cache": str(root / "validation.pth"),
            "--validation-seed": 20260910}
        for flag, value in replacements.items():
            replace_arg(command, flag, value)
        command += ["--snapshot-steps", "4000", "12000", "--validation-cases-per-group", "4"]
        if trial.get("freeze"):
            command += ["--freeze-param-substrings", "cell_query_decoder."]
        if trial.get("balanced"):
            command += ["--allow-partial-init", "--param-lr-multipliers-json",
                        json.dumps({"axis_blend": 100., "axis_gate": 100.})]
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def checkpoint_schedule(command):
    phases = json.loads(command[command.index("--phases-json") + 1])
    total_steps = phases[-1]["end_step"]
    if not isinstance(total_steps, int) or not 0 < total_steps <= 200000:
        raise ValueError("Training horizon must be between 1 and 200000 steps")
    snapshots = set()
    if "--snapshot-steps" in command:
        position = command.index("--snapshot-steps") + 1
        while position < len(command) and not command[position].startswith("--"):
            snapshots.add(int(command[position]))
            position += 1
    if any(not 0 < step <= total_steps for step in snapshots):
        raise ValueError("Snapshot steps must lie within the training horizon")
    return sorted(snapshots | {total_steps})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="spectral_query_20260910")
    parser.add_argument("--wait-for", default="cell_query_20260909_checked")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    trials, commands, prior, anchor = build_commands(root)
    sources = ["scripts/run_spectral_query_experiments.py", "scripts/rowcol_fixed_validation.py",
        "scripts/train_direct_pure_rabel_curriculum.py", "scripts/optimizer_continuation.py", "src/tabimpute/model/cell_query_decoder.py",
        "src/tabimpute/model/model_rowcol_svt_hybrid.py", "src/tabimpute/interface.py",
        "src/tabimpute/prior/spectrum_training_set_generation.py", *ADDITIONAL_SOURCES]
    manifest = {"trials": trials, "commands": commands, "prior": prior,
        "evaluation_model": MODEL_CLASS.__name__,
        "initial_checkpoint": str(anchor), "optimizer": (
            "Retained FP32 AdamW moments/steps, restarted cosine; see individual command flags"
            if any("--retain-init-optimizer-state" in command for command in commands.values())
            else "Fresh FP32 AdamW for every arm; cosine horizon defined by commands"),
        "selection": "Synthetic spectral macro RMSE only; MTEB reporting only",
        "axis_transition": "tanh blend starts zero; new gate/blend LR 100x backbone",
        "steps_by_trial": {tag: checkpoint_schedule(command)[-1] for tag, command in commands.items()},
        "wait_for_idle_gpu": args.wait_for_idle_gpu,
        "limits": "At most 200k steps per arm; configured horizons do not prove convergence",
        "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources}}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text()) != manifest:
        raise ValueError("Manifest changed; use a fresh queue name")
    write_json(root / "manifest.json", manifest)
    for tag, command in commands.items():
        write_json(root / f"{tag}_command.json", command)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import fcntl
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    results = []
    active = "waiting"
    def status(state, **extra):
        write_json(root / "status.json", dict(state=state, active=active,
            completed=[r["tag"] for r in results], updated_unix=time.time(), **extra))
    try:
        status("waiting", dependency=args.wait_for)
        while args.wait_for:
            dependency = json.loads((REPO / "runs/objective_queues" / args.wait_for / "status.json").read_text())
            if dependency["state"] == "complete":
                break
            if dependency["state"] == "error":
                raise RuntimeError(f"Dependency failed: {dependency}")
            time.sleep(60)
        if args.wait_for_idle_gpu:
            while True:
                gpu_pids = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                if not gpu_pids:
                    break
                status("waiting_for_idle_gpu", observed_gpu_pids=gpu_pids)
                time.sleep(60)
        torch.backends.cuda.enable_cudnn_sdp(False)
        cases = load_or_build_validation(root / "holdout.pth", prior, seed=20260911, cases_per_group=4)
        for trial in trials:
            active = trial["tag"]
            result_path = root / f"{active}_result.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                continue
            command = commands[active]
            snapshot_steps = checkpoint_schedule(command)
            total_steps = snapshot_steps[-1]
            work = WORK / command[command.index("--run-name") + 1]
            if shutil.disk_usage(WORK).free < 45 * 1024**3:
                raise RuntimeError("Less than 45 GiB free; no files deleted")
            status("training")
            if not (work / f"checkpoint_{total_steps}.pth").exists():
                if work.exists():
                    raise RuntimeError(f"Review partial training before resuming {work}")
                run_logged(command, root / f"{active}_train.log")
            status("evaluating")
            curve_dir = root / "curve_analysis"
            curve_dir.mkdir(exist_ok=True)
            horizon = analyze(work, curve_dir)
            config = json.loads(command[command.index("--model-overrides-json") + 1])
            criterion = FullSupportBarDistribution(torch.load(
                importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
            scores = {}
            checkpoints = {str(step): work / f"checkpoint_{step}.pth" for step in snapshot_steps}
            checkpoints["selected"] = work / "best_validation_checkpoint.pth"
            for label, checkpoint in checkpoints.items():
                cache = root / f"{active}_{label}_holdout.json"
                if not cache.exists():
                    model = MODEL_CLASS(**config).cuda().eval()
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                    write_json(cache, evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                        root / f"{active}_{label}_predictions.pth"))
                    del model
                    torch.cuda.empty_cache()
                scores[label] = json.loads(cache.read_text())
            general_cache = root / f"{active}_general_holdout.json"
            if not general_cache.exists():
                general_cases = torch.load(REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth",
                    map_location="cpu", weights_only=True)["cases"]
                model = MODEL_CLASS(**config).cuda().eval()
                model.load_state_dict(torch.load(checkpoints["selected"], map_location="cpu", weights_only=True)["model"])
                write_json(general_cache, evaluate_fixed_validation(model, criterion, general_cases, "cuda", "amp_bfloat16",
                    root / f"{active}_general_predictions.pth"))
                del model
                torch.cuda.empty_cache()
            scores["general_selected_report_only"] = json.loads(general_cache.read_text())
            status("evaluating_mteb")
            label = f"{root.name}_{active}"
            benchmark = root / f"{active}_mteb.json"
            if not benchmark.exists():
                run_logged([sys.executable, "code/eval_tabimpute.py"], root / f"{active}_mteb.log",
                    cwd=BENCH, extra_env={"BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint",
                        "BENCHSELECT_TABIMPUTE_LABEL": label,
                        "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(checkpoints["selected"]),
                        "BENCHSELECT_TABIMPUTE_DEVICE": "cuda", "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
                        "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean", "BENCHSELECT_DATASETS": "mteb",
                        "BENCHSELECT_METHODS": "entropy,mi,random", "BENCHSELECT_N_FOLDS": "10",
                        "BENCHSELECT_K_MAX": "15", "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
                with np.load(BENCH / "logs" / f"tabimpute_{label}_mteb.npz") as data:
                    write_json(benchmark, {f"{m}_r2_{k}": float(np.nanmean(data[f"{m}_r2"][:, k - 1]))
                        for m in ("entropy", "mi", "random") for k in (5, 10, 15)})
            result = dict(tag=active, holdout=scores, horizon=horizon,
                          mteb=json.loads(benchmark.read_text()))
            write_json(result_path, result)
            results.append(result)
            write_json(root / "results.json", results)
        active = "complete"
        status("complete")
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
