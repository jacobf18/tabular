"""Existing DWIP checkpoint, newest generic priors, cached baseline and final evals."""
import argparse
import hashlib
import importlib.resources
import json
from pathlib import Path
import sys

import numpy as np
import torch

import run_spectral_query_experiments as queue
from run_spectral_geometry_experiments import build_commands as geometry_commands
from run_rowcol_controlled_experiments import REPO, BENCH, run_logged, write_json
from run_rowcol_output_remedies import replace_arg
from rowcol_fixed_validation import load_or_build_validation, evaluate_fixed_validation
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model.bar_distribution import FullSupportBarDistribution

MANIFEST = REPO / "benchmark/movielens/configs/tabimpute_models.json"
SOURCES = ["scripts/run_dwip_spectral_geometry.py", "scripts/train_dwip_spectral_geometry.py",
           "src/tabimpute/model/model_new_stable.py", "scripts/general_spectral_geometry.py",
           "scripts/spectral_support_normalization.py"]


def dwip_spec():
    return next(m for m in json.loads(MANIFEST.read_text())["models"] if m["name"] == "dwip-v2-2")


def build_commands(root):
    _, originals, prior, _ = geometry_commands(root)
    spec = dwip_spec()
    anchor = Path(spec["checkpoint_path"])
    command = list(originals["geometry_lr1e5"])
    command[1] = "scripts/train_dwip_spectral_geometry.py"
    for flag, value in {
        "--run-name": f"tabimpute-{root.name}-dwip1e5",
        "--init-model-checkpoint": str(anchor),
        "--model-overrides-json": json.dumps(spec["model_config"]),
        "--svt-mixture-depth-distill-weight": 0,
        "--svt-mixture-cv-regret-weight": 0,
        "--data-seed": 1150924,
    }.items():
        replace_arg(command, flag, value)
    return [dict(tag="dwip1e5", lr=1e-5)], {"dwip1e5": command}, prior, anchor


def evaluate_baseline(root):
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    spec = dwip_spec()
    _, _, prior, anchor = build_commands(root)
    model = TabImputeModelStable(**spec["model_config"]).cuda().eval()
    model.load_state_dict(torch.load(anchor, map_location="cpu", weights_only=True)["model"], strict=True)
    criterion = FullSupportBarDistribution(torch.load(
        importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
    spectral = load_or_build_validation(root / "holdout.pth", prior, seed=20260911, cases_per_group=4)
    general = torch.load(REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth",
                         weights_only=True, map_location="cpu")["cases"]
    for name, cases in (("spectral", spectral), ("general", general)):
        path = root / f"baseline_{name}.json"
        if not path.exists():
            write_json(path, evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                                                       root / f"baseline_{name}_predictions.pth"))
    del model, criterion
    torch.cuda.empty_cache()
    # Give the existing direct-checkpoint evaluator a local config without
    # modifying or copying the original weights.
    directory = root / "baseline"
    directory.mkdir(exist_ok=True)
    link = directory / "checkpoint.pth"
    if not link.exists():
        link.symlink_to(anchor)
    elif link.resolve() != anchor.resolve():
        raise ValueError("Baseline checkpoint link changed")
    write_json(directory / "config.json", dict(model_family="new_stable", model=spec["model_config"]))
    score = root / "baseline_mteb.json"
    if not score.exists():
        label = f"{root.name}_baseline"
        run_logged([sys.executable, "code/eval_tabimpute.py"], root / "baseline_mteb.log", cwd=BENCH,
            extra_env={"BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint",
                "BENCHSELECT_TABIMPUTE_LABEL": label, "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(link),
                "BENCHSELECT_TABIMPUTE_DEVICE": "cuda", "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
                "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean", "BENCHSELECT_DATASETS": "mteb",
                "BENCHSELECT_METHODS": "entropy,mi,random", "BENCHSELECT_N_FOLDS": "10",
                "BENCHSELECT_K_MAX": "15", "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
        with np.load(BENCH / "logs" / f"tabimpute_{label}_mteb.npz") as data:
            write_json(score, {f"{method}_r2_{k}": float(np.nanmean(data[f"{method}_r2"][:, k - 1]))
                for method in ("entropy", "mi", "random") for k in (5, 10, 15)})


def smoke_command(root):
    _, commands, _, _ = build_commands(root)
    command = list(commands["dwip1e5"])
    for flag in ("--fixed-validation-cache",):
        i = command.index(flag)
        del command[i:i+2]
    i = command.index("--snapshot-steps")
    j = i + 1
    while j < len(command) and not command[j].startswith("--"):
        j += 1
    del command[i:j]
    for flag, value in {
        "--run-name": "dwip_spectral_preflight",
        "--phases-json": json.dumps([dict(name="max_training_shape", end_step=3,
             rows_low=320, rows_high=320, cols_low=100, cols_high=100, batch_size=2)]),
        "--warmup-steps": 1, "--decay-steps": 2, "--checkpoint-every": 0,
        "--metrics-jsonl-every": 1, "--update-diagnostics-every": 1,
    }.items():
        replace_arg(command, flag, value)
    command += ["--work-dir", str(root / "smoke")]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_20260915")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    if args.baseline_only:
        evaluate_baseline(root)
        return
    if args.smoke_only:
        run_logged(smoke_command(root), root / "smoke.log")
        return
    spec = dwip_spec()
    manifest = dict(source_sha256={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in SOURCES},
        original_checkpoint=spec["checkpoint_path"], model=spec["model_config"],
        objective="User replaced RowCol experiments with existing DWIP checkpoint fine-tuning",
        initialization="Strict model-only initialization; fresh FP32 AdamW; original checkpoint untouched",
        training="Latest general spectral geometry and df1 support normalization; no prior labels supplied to DWIP",
        schedule="24k steps, LR1e-5, 600 warmup, cosine to5%; FP32 weights/BF16 autocast; batch2",
        losses="NLL1 + MSE0.2 + CDF0.05; no inapplicable SVT auxiliary losses",
        selection="Synthetic validation only; baseline and selected-checkpoint MTEB are report-only",
        reporting="Cached spectral and broad predictions, curves, runtime and 10fold MTEB")
    path = root / "dwip_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Configuration changed; use a new queue name")
    write_json(path, manifest)
    if not args.dry_run:
        write_json(root / "status.json", dict(state="baseline_evaluation", active="baseline"))
        run_logged([sys.executable, "scripts/run_dwip_spectral_geometry.py", "--queue-name", args.queue_name,
                    "--baseline-only"], root / "baseline.log")
    queue.build_commands = build_commands
    queue.MODEL_CLASS = TabImputeModelStable
    queue.ADDITIONAL_SOURCES = SOURCES
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", ""]
    if args.dry_run:
        sys.argv.append("--dry-run")
    queue.main()


if __name__ == "__main__":
    main()
