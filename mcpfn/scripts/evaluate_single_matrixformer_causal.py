"""Evaluate the combined MatrixFormer on frozen causal placebo draws.

Reuses the original inference and metric implementations, never modifies the
benchmark directories, and holds the archived seven-method scale fixed.
"""

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_evaluation_common import atomic_npz, sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
CHECKPOINT = Path("/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth")
CHECKPOINT_SHA = "0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49"
ENSEMBLE = ROOT / "runs/benchmark_evaluations/causal_matrixformer_plus_20261005"
OUTPUT = ROOT / "runs/benchmark_evaluations/single_matrixformer_causal_20261102"
BASELINES = ("sdid", "sc", "did", "softimpute", "knn", "hyperimpute")
REFERENCES = ("matrixformer-plus", *BASELINES)
ORIENTATIONS = ("standard", "transpose", "average")
CANDIDATES = ("matrixformer", "matrixformer-T", "matrixformer-avg")
METRICS = ("treated_block_rmse", "tau_abs_error")
DESIGNS = [
    ("CPS", f"{outcome.replace('_', ' ')} / {treatment.replace('_', ' ')}",
     f"CPS/causal_eval/{outcome}_{treatment}", 200)
    for outcome in ("hours", "log_wage", "urate")
    for treatment in ("abort_ban", "min_wage", "open_carry")
] + [
    ("Kansas", "factor model", "kansas/causal_eval/kansas_factor_r3_200", 200),
    ("California smoking", "simultaneous", "prop99/causal_eval/ca_simul", 500),
    ("California smoking", "staggered", "prop99/causal_eval/ca_stagg", 500),
]


def causal_module():
    sys.path.insert(0, str(BENCHMARK))
    import eval_causal_new_architectures
    return eval_causal_new_architectures


def load_draw(draw):
    observed = np.load(draw / "observed.npy")
    true = np.load(draw / "true.npy")
    mask = np.load(draw / "treatment_mask.npy").astype(bool)
    if observed.shape != true.shape or mask.shape != true.shape:
        raise ValueError(f"Shape mismatch: {draw}")
    if not np.array_equal(mask, np.isnan(observed)) or not mask.any() or mask.all():
        raise ValueError(f"Invalid hidden block: {draw}")
    if not np.isfinite(true).all() or not np.array_equal(observed[~mask], true[~mask]):
        raise ValueError(f"Invalid observed values: {draw}")
    return observed, true, mask


def complete(impute, observed, standardize, transpose=False):
    panel = observed.T if transpose else observed
    normalized, means, scales = standardize(panel)
    prediction = (impute(normalized.copy()).astype(np.float32) * scales + means).astype(np.float32)
    if transpose:
        prediction = prediction.T.copy()
    prediction[~np.isnan(observed)] = observed[~np.isnan(observed)]
    if prediction.shape != observed.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid completion")
    return prediction


def average_predictions(standard, transposed, observed):
    average = (0.5 * (standard.astype(np.float32) + transposed.astype(np.float32))).astype(np.float32)
    average[~np.isnan(observed)] = observed[~np.isnan(observed)]
    return average


def summarize(frame):
    """Fixed-reference min-max scores, equal design weights, draw-level CIs."""
    keys = ["family", "path", "draw_name", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate method/draw rows")
    methods = list(REFERENCES) + [method for method in CANDIDATES if method in set(frame.method)]
    if set(frame.method) != set(methods):
        raise ValueError("Unexpected or missing methods")
    design_rows = []
    for (family, path), group in frame.groupby(["family", "path"], sort=False):
        for metric in METRICS:
            wide = group.pivot(index="draw_name", columns="method", values=metric).reindex(columns=methods)
            if len(wide) < 2 or not np.isfinite(wide.to_numpy()).all():
                raise ValueError(f"Missing/nonfinite method-draw entries: {path} {metric}")
            lower, upper = wide[list(REFERENCES)].min(axis=1), wide[list(REFERENCES)].max(axis=1)
            valid = upper > lower
            normalized = wide.sub(lower, axis=0).div(upper - lower, axis=0).loc[valid]
            if len(normalized) < 2:
                raise ValueError(f"Too few non-tied draws: {path} {metric}")
            for method in methods:
                design_rows.append({
                    "family": family, "path": path, "metric": metric, "method": method,
                    "score": float(normalized[method].mean()),
                    "se": float(normalized[method].std(ddof=1) / np.sqrt(len(normalized))),
                    "raw_mean": float(wide[method].mean()),
                    "raw_se": float(wide[method].std(ddof=1) / np.sqrt(len(wide))),
                    "n_draws": len(normalized), "n_tied_draws": int((~valid).sum()),
                })
    designs = pd.DataFrame(design_rows)
    rows = []
    for (family, metric, method), group in designs.groupby(["family", "metric", "method"], sort=False):
        error = float(np.sqrt((group.se ** 2).sum()) / len(group))
        rows.append({"family": family, "metric": metric, "method": method,
                     "score": float(group.score.mean()), "se": error, "ci95": 1.96 * error,
                     "n_scenarios": len(group), "n_draws": int(group.n_draws.sum()),
                     "shown": method in (*REFERENCES, "matrixformer")})
    return pd.DataFrame(rows), designs


def inventory():
    inputs, draws = {}, {}
    for family, _, path, count in DESIGNS:
        selected = sorted((BENCHMARK / path).glob("draw_[0-9][0-9][0-9][0-9]"))
        start = 1 if family == "CPS" else 0
        if [draw.name for draw in selected] != [f"draw_{index:04d}" for index in range(start, start + count)]:
            raise ValueError(f"Unexpected draw inventory: {path}")
        draws[path] = selected
        for draw in selected:
            load_draw(draw)
            for filename in ("observed.npy", "true.npy", "treatment_mask.npy", "metadata.json"):
                source = draw / filename
                inputs[str(source.relative_to(BENCHMARK))] = sha256(source)
    return draws, inputs


def references(draws):
    ensemble_file = ENSEMBLE / "per_draw_metrics.csv"
    ensemble = pd.read_csv(ensemble_file)
    frames, hashes = [], {str(ensemble_file): sha256(ensemble_file)}
    columns = ["family", "path", "draw_name", "method", *METRICS]
    for family, _, path, _ in DESIGNS:
        source = BENCHMARK / path / "results/per_draw_metrics.csv"
        baseline = pd.read_csv(source)
        baseline = baseline[baseline.method.isin(BASELINES)].copy()
        baseline["family"], baseline["path"] = family, path
        selected = pd.concat([baseline[columns], ensemble.loc[ensemble.path == path, columns]])
        if selected.duplicated(["draw_name", "method"]).any():
            raise ValueError(f"Duplicate reference: {path}")
        expected = {(draw.name, method) for draw in draws[path] for method in REFERENCES}
        if set(zip(selected.draw_name, selected.method)) != expected:
            raise ValueError(f"Reference coverage mismatch: {path}")
        frames.append(selected)
        hashes[str(source)] = sha256(source)
    return pd.concat(frames, ignore_index=True), hashes


def make_imputer(causal, checkpoint):
    return causal.make_imputer(causal.MethodSpec(
        name="matrixformer", kind="direct", checkpoint=checkpoint, model_family="new_stable",
        precision_mode="amp_bfloat16", prediction_stat="mean"), "cuda")


def preflight(causal, draws, output):
    import torch
    parent = causal.DWIP_SPECTRAL_TM128K_CHECKPOINT
    expected = "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d"
    if sha256(parent) != expected:
        raise ValueError("Parent checkpoint hash mismatch")
    impute = make_imputer(causal, parent)
    rows = []
    for _, _, path, _ in DESIGNS:
        draw = draws[path][0]
        observed, _, _ = load_draw(draw)
        standard = complete(impute, observed, causal.standardize_columns)
        transposed = complete(impute, observed, causal.standardize_columns, transpose=True)
        average = average_predictions(standard, transposed, observed)
        for suffix, predicted in zip(("", "-T", "-avg"), (standard, transposed, average)):
            source = draw / f"dwip-spectral-tm128k{suffix}.npy"
            saved = np.load(source)
            identical = np.array_equal(predicted, saved)
            rows.append({"path": path, "draw": draw.name, "suffix": suffix,
                         "bit_identical": identical, "max_abs_diff": float(np.max(np.abs(predicted - saved))),
                         "reference_sha256": sha256(source)})
    write_json(output / "control_parity.json", {"checkpoint_sha256": expected, "checks": rows})
    if not all(row["bit_identical"] for row in rows):
        raise ValueError("Control parity failed; inspect control_parity.json")
    del impute
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Parent control: {len(rows)}/{len(rows)} predictions bit-identical", flush=True)


def evaluate(causal, draws, output, checkpoint):
    import torch
    impute = make_imputer(causal, checkpoint)
    all_rows = []
    for family, label, path, _ in DESIGNS:
        stem = path.replace("/", "__")
        prediction_file = output / "predictions" / f"{stem}.npz"
        metrics_file = output / f"{stem}.csv"
        done = output / f"{stem}.done.json"
        if done.exists():
            for name, expected in json.loads(done.read_text()).items():
                if sha256(output / name) != expected:
                    raise ValueError(f"Resume output hash mismatch: {name}")
            all_rows.extend(pd.read_csv(metrics_file).to_dict("records"))
            continue
        predictions = {orientation: [] for orientation in ORIENTATIONS}
        rows = []
        warm, _, _ = load_draw(draws[path][0])
        for transpose in (False, True):
            complete(impute, warm, causal.standardize_columns, transpose)
        for draw in draws[path]:
            observed, true, mask = load_draw(draw)
            outputs, runtimes = [], []
            for transpose in (False, True):
                torch.cuda.synchronize()
                start = time.perf_counter()
                completed = complete(impute, observed, causal.standardize_columns, transpose)
                torch.cuda.synchronize()
                runtimes.append(time.perf_counter() - start)
                outputs.append(completed)
            outputs.append(average_predictions(*outputs, observed))
            runtimes.append(sum(runtimes))
            for method, orientation, completed, runtime in zip(CANDIDATES, ORIENTATIONS, outputs, runtimes):
                predictions[orientation].append(completed)
                rows.append({"family": family, "design": label, "path": path,
                             "draw_name": draw.name, "method": method, "runtime_seconds": runtime,
                             **causal.compute_metrics(true, completed, mask)})
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(prediction_file, draw_names=np.array([draw.name for draw in draws[path]]),
                   **{key: np.stack(value) for key, value in predictions.items()})
        pd.DataFrame(rows).to_csv(metrics_file, index=False)
        write_json(done, {str(source.relative_to(output)): sha256(source)
                          for source in (prediction_file, metrics_file)})
        all_rows.extend(rows)
        print(f"Completed {path}: {len(draws[path])} draws, three orientations", flush=True)
    return pd.DataFrame(all_rows)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Combined checkpoint hash mismatch")
    causal = causal_module()
    draws, inputs = inventory()
    reference, reference_hashes = references(draws)
    source_paths = [Path(__file__), Path(causal.__file__), BENCHMARK / "CPS/eval_cps_causal.py",
                    Path(__file__).with_name("recsys_evaluation_common.py")]
    source_paths += sorted((ROOT / "src/tabimpute").rglob("*.py"))
    identity = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": CHECKPOINT_SHA,
        "config_sha256": sha256(args.checkpoint.parent / "config.json"),
        "inputs": inputs, "references": reference_hashes,
        "sources": {str(source): sha256(source) for source in source_paths},
        "protocol": {"primary_orientation": "standard", "orientations": ORIENTATIONS,
                     "normalization_references": REFERENCES, "new_scores_may_exceed_unit_interval": True,
                     "column_statistics": "observed-only; separately for each orientation",
                     "model_family": "new_stable", "prediction_stat": "mean",
                     "precision": "amp_bfloat16", "chunking": False,
                     "cudnn_sdpa": torch.backends.cuda.cudnn_sdp_enabled(),
                     "timing": "warm, CUDA-synchronized completion including normalization; average is sum of two calls",
                     "selection": "fixed combined checkpoint; no test-outcome checkpoint or orientation selection"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    manifest_file = args.output / "manifest.json"
    identity = json.loads(json.dumps(identity))
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Resume identity mismatch; use a fresh output directory")
    else:
        manifest = {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(manifest_file, manifest)
    preflight(causal, draws, args.output)
    evaluated = evaluate(causal, draws, args.output, args.checkpoint)
    scores, designs = summarize(pd.concat([reference, evaluated], ignore_index=True))
    evaluated.to_csv(args.output / "per_draw_metrics.csv", index=False)
    reference.to_csv(args.output / "reference_per_draw_metrics.csv", index=False)
    scores.to_csv(args.output / "normalized_scores.csv", index=False)
    designs.to_csv(args.output / "per_design_scores.csv", index=False)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = {source.name: sha256(source) for source in args.output.glob("*.csv")}
    write_json(manifest_file, manifest)
    print(scores[scores.method.isin(CANDIDATES)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
