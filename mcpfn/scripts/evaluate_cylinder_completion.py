"""Frozen zero-shot cylinder-flow completion using PyDMD data and fancyimpute."""

import argparse
import hashlib
from importlib import metadata, resources
import json
import math
from pathlib import Path
import platform
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ndtr
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.routing import fit_routing, observed_holdout


ROOT = Path(__file__).resolve().parents[1]
DATA_REVISION = "024464ff667e4b0226f9ecbe933bc67f3fe21e81"
DATA_URL = (f"https://raw.githubusercontent.com/PyDMD/PyDMD/{DATA_REVISION}/"
            "tutorials/user-manual1/CYLINDER_ALL_LOW_RES.mat")
SINGLE = Path("/mnt/checkpoints_200/single_model_goal_20261005/"
              "four_draw_v57_user_stop_100/best_validation_checkpoint.pth")
SINGLE_SHA256 = "0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49"
GENERAL = ROOT / "runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth"
SPECIALIST = ROOT / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth"
METHODS = ["Column mean", "Temporal interpolation", "Iterative gappy POD", "SoftImpute",
           "MatrixFormer", "MatrixFormer*"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalize_observed(matrix):
    observed = np.asarray(matrix)[np.isfinite(matrix)]
    if not observed.size:
        raise ValueError("No observations")
    mean, scale = float(observed.mean()), float(observed.std())
    scale = scale if scale > 1e-8 else 1.0
    return (matrix - mean) / scale, mean, scale


def make_mask(shape, rate, pattern, seed):
    if not 0 < rate < 1 or min(shape) < 2:
        raise ValueError("Require partial missingness and nontrivial matrix dimensions")
    random = np.random.default_rng(seed)
    if pattern == "mcar":
        missing = random.random(shape) < rate
    elif pattern == "sensor_outage":
        duration = min(shape[1] - 1, max(1, round(shape[1] * rate)))
        starts = random.integers(shape[1], size=(shape[0], 1))
        missing = (np.arange(shape[1])[None, :] - starts) % shape[1] < duration
    else:
        raise ValueError(pattern)
    if missing.all(axis=0).any() or missing.all(axis=1).any():
        raise ValueError("Mask leaves an unsupported row or column; do not silently repair it")
    return missing


def point_metrics(truth, prediction, missing, scale):
    error = np.asarray(prediction)[missing] - np.asarray(truth)[missing]
    if not error.size or not np.isfinite(error).all():
        raise ValueError("Invalid predictions or empty test set")
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"rmse": rmse, "nrmse": rmse / scale, "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error ** 2) / np.sum((truth[missing] - truth[missing].mean()) ** 2))}


def full_support_quantiles(probabilities, borders, levels):
    """Invert the actual uniform-interior/half-normal-tail predictive density."""
    cumulative = probabilities.cumsum(-1)
    cumulative[..., -1] = 1.0
    requested = levels.expand(*probabilities.shape[:-1], len(levels)).contiguous()
    indices = torch.searchsorted(cumulative.contiguous(), requested).clamp_max(probabilities.shape[-1] - 1)
    previous = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], dim=-1)
    fraction = ((requested - previous.gather(-1, indices)) /
                probabilities.gather(-1, indices).clamp_min(1e-30)).clamp(1e-7, 1 - 1e-7)
    quantiles = borders[indices] + fraction * (borders[indices + 1] - borders[indices])
    halfnormal_median = 0.6744897501960817
    left_scale = (borders[1] - borders[0]) / halfnormal_median
    right_scale = (borders[-1] - borders[-2]) / halfnormal_median
    left = borders[1] - left_scale * math.sqrt(2) * torch.erfinv(1 - fraction)
    right = borders[-2] + right_scale * math.sqrt(2) * torch.erfinv(fraction)
    return torch.where(indices == 0, left,
                       torch.where(indices == probabilities.shape[-1] - 1, right, quantiles))


def quantile_crps(quantiles, target, levels):
    residual = target.unsqueeze(-1) - quantiles
    return 2 * torch.maximum(levels * residual, (levels - 1) * residual).mean(-1)


def gaussian_metrics(target, prediction, deviation, scale):
    deviation = max(float(deviation), 1e-8)
    standardized = (target - prediction) / deviation
    density = np.exp(-.5 * standardized ** 2) / math.sqrt(2 * math.pi)
    crps = deviation * (standardized * (2 * ndtr(standardized) - 1) + 2 * density - 1 / math.sqrt(math.pi))
    result = {"crps": float(crps.mean()), "normalized_crps": float(crps.mean() / scale),
              "nll": float(np.mean(.5 * standardized ** 2 + math.log(deviation * math.sqrt(2 * math.pi)))),
              "uncertainty_cells": len(target), "uncertainty_type": "observed-validation Gaussian residual"}
    for coverage, multiplier in [(50, .6744897501960817), (90, 1.6448536269514722), (95, 1.959963984540054)]:
        result[f"coverage_{coverage}"] = float(np.mean(np.abs(standardized) <= multiplier))
        result[f"width_{coverage}"] = float(2 * multiplier * deviation)
    return result


def fit_baseline(matrix, method, parameter, seed):
    from fancyimpute import IterativeSVD, SoftImpute

    normalized, mean, scale = normalize_observed(matrix)
    np.random.seed(seed)
    if method == "Iterative gappy POD":
        solver = IterativeSVD(rank=int(parameter), max_iters=200, convergence_threshold=1e-5,
                              gradual_rank_increase=False, verbose=False)
    elif method == "SoftImpute":
        singular_max = float(np.linalg.svd(np.nan_to_num(normalized), compute_uv=False)[0])
        solver = SoftImpute(shrinkage_value=float(parameter) * singular_max, max_iters=200,
                            max_rank=32, convergence_threshold=1e-5, n_power_iterations=3, verbose=False)
    else:
        raise ValueError(method)
    return solver.fit_transform(normalized) * scale + mean


def classical_predictions(matrix, seed):
    observed = np.isfinite(matrix)
    predictions = {"Column mean": np.where(observed, matrix, np.nanmean(matrix, axis=0))}
    interpolated = matrix.copy()
    times = np.arange(matrix.shape[1])
    for row in range(len(matrix)):
        interpolated[row] = np.interp(times, times[observed[row]], matrix[row, observed[row]])
    predictions["Temporal interpolation"] = interpolated
    validation = observed_holdout(matrix, fraction=.1, seed=seed + 104729)
    reduced = matrix.copy()
    reduced[validation] = np.nan
    tuning = {}
    for method, grid in [("Iterative gappy POD", [2, 4, 8, 16, 32]),
                         ("SoftImpute", [.005, .02, .1])]:
        started = time.perf_counter()
        trials = []
        for parameter in grid:
            candidate = fit_baseline(reduced, method, parameter, seed)
            error = candidate[validation] - matrix[validation]
            trials.append({"parameter": parameter, "validation_rmse": float(np.sqrt(np.mean(error ** 2)))})
        selected = min(trials, key=lambda trial: trial["validation_rmse"])
        predictions[method] = fit_baseline(matrix, method, selected["parameter"], seed)
        tuning[method] = {"selected": selected, "trials": trials, "validation_cells": int(validation.sum()),
                          "seconds_including_tuning": time.perf_counter() - started}
    return predictions, tuning, validation


def load_models(checkpoints):
    models = {}
    for name, checkpoint in checkpoints.items():
        config = json.loads(checkpoint.with_name("config.json").read_text())["model"]
        model = TabImputeModelStable(**config).cuda().eval()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights["model"], strict=True)
        models[name] = model
    with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as stream:
        borders = torch.load(stream, weights_only=True).cuda()
    return models, FullSupportBarDistribution(borders)


@torch.inference_mode()
def neural_predictions(matrix, models, distribution):
    normalized, mean, scale = normalize_observed(matrix)
    values = torch.from_numpy(normalized.astype(np.float32)).cuda().unsqueeze(0)
    features, raw, timings = {}, {}, {}
    for name, model in models.items():
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model(values, return_embeddings=True)[0]
            features[name] = encoded.reshape(-1, encoded.shape[-1])
        point = torch.empty(matrix.size, device="cuda", dtype=torch.float32)
        for start in range(0, matrix.size, 2048):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.decoder(features[name][start:start + 2048])
            point[start:start + len(logits)] = distribution.mean(logits.float())
        raw[name] = point.cpu().numpy().reshape(matrix.shape).astype(np.float64) * scale + mean
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
    decision = fit_routing(matrix, raw["general"], raw["specialist"], np.isfinite(matrix),
                           np.full(matrix.shape[1], scale))
    weight = decision["specialist_weight"]
    predictions = {"MatrixFormer": raw["single"],
                   "MatrixFormer*": (1 - weight) * raw["general"] + weight * raw["specialist"]}
    return predictions, features, decision, timings


@torch.inference_mode()
def neural_uncertainty(models, features, distribution, matrix, truth, indices, weight):
    _, mean, scale = normalize_observed(matrix)
    targets = torch.from_numpy(((truth.ravel()[indices] - mean) / scale).astype(np.float32)).cuda()
    index_tensor = torch.as_tensor(indices, device="cuda")
    levels = (torch.arange(199, device="cuda") + .5) / 199
    interval_levels = torch.tensor([.025, .05, .25, .75, .95, .975], device="cuda")
    all_levels = torch.cat([levels, interval_levels])
    saved = {method: {key: [] for key in ("crps", "nll", "quantiles", "std")}
             for method in ("MatrixFormer", "MatrixFormer*")}
    for start in range(0, len(indices), 1024):
        selection = index_tensor[start:start + 1024]
        current_target = targets[start:start + 1024]
        probabilities = {}
        for name, model in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.decoder(features[name][selection])
            probabilities[name] = logits.float().softmax(-1)
        for method, probability in [("MatrixFormer", probabilities["single"]),
                                     ("MatrixFormer*", (1 - weight) * probabilities["general"] + weight * probabilities["specialist"])]:
            quantiles = full_support_quantiles(probability, distribution.borders, all_levels)
            log_probability = probability.clamp_min(1e-38).log()
            values = {"crps": quantile_crps(quantiles[:, :199], current_target, levels) * scale,
                      "nll": distribution(log_probability, current_target) + math.log(scale),
                      "quantiles": quantiles[:, 199:] * scale + mean,
                      "std": distribution.variance(log_probability).clamp_min(0).sqrt() * scale}
            for key, value in values.items():
                saved[method][key].append(value.cpu().numpy())
    records, arrays = {}, {}
    for method, chunks in saved.items():
        values = {key: np.concatenate(parts) for key, parts in chunks.items()}
        quantiles = values["quantiles"]
        target = truth.ravel()[indices]
        record = {"crps": float(values["crps"].mean()), "normalized_crps": float(values["crps"].mean() / scale),
                  "nll": float(values["nll"].mean()), "uncertainty_cells": len(indices),
                  "uncertainty_type": "native full-support bars" if method == "MatrixFormer" else "convex mixture of native bars"}
        for coverage, lower, upper in [(50, 2, 3), (90, 1, 4), (95, 0, 5)]:
            record[f"coverage_{coverage}"] = float(np.mean((target >= quantiles[:, lower]) & (target <= quantiles[:, upper])))
            record[f"width_{coverage}"] = float(np.mean(quantiles[:, upper] - quantiles[:, lower]))
        records[method], arrays[method] = record, values
    return records, arrays


def summarize(output):
    records = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    rows = [{**{key: record[key] for key in ("case_id", "pattern", "rate", "seed", "actual_missing_fraction", "observed_std")},
             "method": method, **metrics} for record in records for method, metrics in record["metrics"].items()]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_case.csv", index=False)
    metrics = ["rmse", "nrmse", "mae", "r2", "crps", "normalized_crps", "nll",
               "coverage_50", "coverage_90", "coverage_95", "width_50", "width_90", "width_95"]
    summary = frame.groupby(["pattern", "rate", "method"], sort=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    return frame, summary.reset_index()


def make_figures(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font="STIXGeneral", font_scale=.85)
    plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stix"})
    patterns, rates = list(summary.pattern.unique()), sorted(summary.rate.unique())
    figure, axes = plt.subplots(len(patterns), len(rates), figsize=(11, 3.1 * len(patterns)), squeeze=False)
    for row, pattern in enumerate(patterns):
        for column, rate in enumerate(rates):
            axis = axes[row, column]
            data = summary[(summary.pattern == pattern) & (summary.rate == rate)].sort_values("nrmse_mean")
            colors = ["#2A6FBB" if method.startswith("MatrixFormer") else "#b9bec8" for method in data.method]
            axis.barh(data.method, data.nrmse_mean, color=colors, xerr=data.nrmse_std.fillna(0), capsize=2)
            for position, value in enumerate(data.nrmse_mean):
                axis.text(value + .01 * max(data.nrmse_mean), position, f"{value:.3f}", va="center", fontsize=8,
                          fontweight="bold" if position == 0 else "normal")
            axis.set_xlim(0, max(data.nrmse_mean + data.nrmse_std.fillna(0)) * 1.22)
            axis.invert_yaxis()
            axis.set_title(f"{pattern.replace('_', ' ').upper()} · {rate:.0%} missing")
            axis.set_xlabel("NRMSE ↓ (observed global SD)")
            axis.grid(axis="y", visible=False)
    sns.despine(fig=figure)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"accuracy.{extension}", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(patterns), figsize=(9, 3.3), squeeze=False)
    for axis, pattern in zip(axes[0], patterns):
        data = summary[summary.pattern == pattern]
        for method, color, marker in [("MatrixFormer", "#2A6FBB", "o"), ("MatrixFormer*", "#174778", "s"),
                                      ("Iterative gappy POD", "#777777", "^"), ("SoftImpute", "#aaaaaa", "d")]:
            current = data[data.method == method].sort_values("rate")
            axis.errorbar(current.rate * 100, current.coverage_90_mean * 100,
                          yerr=current.coverage_90_std.fillna(0) * 100, color=color, marker=marker, label=method, capsize=2)
        axis.axhline(90, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
        axis.set(title=pattern.replace("_", " ").upper(), xlabel="Missing entries (%)", ylabel="90% interval coverage (%)", ylim=(0, 101))
    axes[0, -1].legend(fontsize=8, loc="lower left")
    sns.despine(fig=figure)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"calibration.{extension}", dpi=180)
    plt.close(figure)


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output.parent / "data/CYLINDER_ALL_LOW_RES.mat"
    data_path.parent.mkdir(exist_ok=True)
    if not data_path.exists():
        urllib.request.urlretrieve(DATA_URL, data_path)
    loaded = loadmat(data_path)
    original = loaded["VORTALL"]
    if not np.isfinite(original).all() or np.max(np.abs(original.imag)) > 1e-10:
        raise ValueError("Expected finite real-valued vorticity")
    if not 1 < arguments.rows <= len(original):
        raise ValueError("Invalid spatial sample size")
    spatial_indices = np.sort(np.random.default_rng(20261027).choice(len(original), arguments.rows, replace=False))
    truth = original.real[spatial_indices].astype(np.float64)
    checkpoints = {"single": arguments.checkpoint.resolve(), "general": GENERAL, "specialist": SPECIALIST}
    if sha256(checkpoints["single"]) != SINGLE_SHA256:
        raise ValueError("This evaluation is frozen to the published MatrixFormer checkpoint")
    sources = [Path(__file__).resolve(), *sorted((ROOT / "src/tabimpute/model").glob("*.py")), ROOT / "src/tabimpute/routing.py"]
    from fancyimpute import IterativeSVD, SoftImpute
    import inspect
    sources += [Path(inspect.getfile(IterativeSVD)), Path(inspect.getfile(SoftImpute))]
    manifest = {"dataset": "PyDMD low-resolution Re=100 cylinder wake, noiseless vorticity", "data_url": DATA_URL,
                "data_revision": DATA_REVISION, "data_sha256": sha256(data_path), "source_shape": list(original.shape),
                "matrix_shape": list(truth.shape), "spatial_sampling_seed": 20261027,
                "rows": "spatial locations sorted in source order", "columns": "all 151 chronological snapshots",
                "grid_shape_from_mat": [int(loaded["ny"].item()), int(loaded["nx"].item())],
                "patterns": arguments.patterns, "rates": arguments.rates, "mask_seeds": arguments.seeds,
                "outage_definition": "One circular contiguous temporal interval per sensor; wrap at recording boundary to preserve snapshot support",
                "normalization": "Global mean and population SD from outer observations only; refitted on inner observations for baseline tuning",
                "nrmse_definition": "RMSE on outer hidden entries / global population SD of outer observations; not per-column MissBench NRMSE",
                "baseline_selection": "10% of outer observations hidden with row/column support preserved; all outer observations restored for final fit",
                "pod_ranks": [2, 4, 8, 16, 32], "softimpute_shrinkage_fractions": [.005, .02, .1], "baseline_max_iters": 200,
                "ensemble": "Existing observed-reconstruction convex router; both specialist checkpoints use the same physics global normalization. Not the recommender preprocessing recipe.",
                "ensemble_distribution": "Convex mixture with the same observed-only weight; not a distribution of independent averaged draws",
                "uncertainty_max_cells": arguments.uncertainty_cells, "uncertainty_sampling": "Uniform subset of outer hidden cells, seed=mask_seed+65537",
                "crps": "199 midpoint quantiles; numerical pinball integral; full-support half-normal tails explicitly inverted",
                "baseline_uncertainty": "Homoscedastic Gaussian with SD equal to selected inner-validation RMSE; not a Bayesian POD posterior",
                "checkpoint_sha256": {str(path): sha256(path) for path in checkpoints.values()},
                "config_sha256": {str(path.with_name('config.json')): sha256(path.with_name('config.json')) for path in checkpoints.values()},
                "source_sha256": {str(path): sha256(path) for path in sources},
                "packages": {name: metadata.version(name) for name in ["torch", "numpy", "scipy", "fancyimpute", "scikit-learn", "pandas"]},
                "python": platform.python_version(), "device": torch.cuda.get_device_name(), "precision": "FP32 weights, BF16 autocast",
                "context": "Whole sampled matrix in one forward pass, no context chunking, native orientation only; decoder microbatches of 2048 cells",
                "training": False, "checkpoint_selection_on_physics": False, "added_noise": False,
                "expected_cases": len(arguments.patterns) * len(arguments.rates) * len(arguments.seeds)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing experiment differs; use a new output directory")
    write_json(manifest_path, manifest)
    np.savez_compressed(output / "dataset.npz", truth=truth, spatial_indices=spatial_indices)
    (output / "cases").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    models, distribution = load_models(checkpoints)
    completed = 0
    for pattern in arguments.patterns:
        for rate in arguments.rates:
            for seed in arguments.seeds:
                case_id = f"{pattern}_{round(rate * 100):02d}_seed{seed}"
                record_path = output / "cases" / f"{case_id}.json"
                prediction_path = output / "predictions" / f"{case_id}.npz"
                if record_path.exists():
                    previous = json.loads(record_path.read_text())
                    if previous["prediction_sha256"] != sha256(prediction_path):
                        raise ValueError("Saved prediction changed")
                    completed += 1
                    continue
                print(f"Starting {case_id}", flush=True)
                write_json(output / "status.json", {"state": "running", "case": case_id, "completed": completed, "expected": manifest["expected_cases"]})
                started = time.perf_counter()
                missing = make_mask(truth.shape, rate, pattern, seed)
                matrix = np.where(missing, np.nan, truth)
                _, mean, scale = normalize_observed(matrix)
                predictions, tuning, validation = classical_predictions(matrix, seed)
                neural, features, decision, timings = neural_predictions(matrix, models, distribution)
                predictions.update(neural)
                hidden_indices = np.flatnonzero(missing)
                indices = np.sort(np.random.default_rng(seed + 65537).choice(hidden_indices,
                                 min(arguments.uncertainty_cells, len(hidden_indices)), replace=False))
                uncertainty, uncertainty_arrays = neural_uncertainty(models, features, distribution, matrix, truth,
                                                                     indices, decision["specialist_weight"])
                del features
                metrics = {method: point_metrics(truth, prediction, missing, scale) for method, prediction in predictions.items()}
                for method, values in uncertainty.items():
                    metrics[method].update(values)
                for method, details in tuning.items():
                    metrics[method].update(gaussian_metrics(truth.ravel()[indices], predictions[method].ravel()[indices],
                                                          details["selected"]["validation_rmse"], scale))
                arrays = {"missing": missing, "validation": validation, "uncertainty_indices": indices}
                for method, prediction in predictions.items():
                    arrays[f"prediction_{METHODS.index(method)}"] = prediction.astype(np.float32)
                for method, fields in uncertainty_arrays.items():
                    arrays.update({f"uncertainty_{METHODS.index(method)}_{key}": value for key, value in fields.items()})
                np.savez_compressed(prediction_path, **arrays)
                record = {"case_id": case_id, "pattern": pattern, "rate": rate, "seed": seed,
                          "actual_missing_fraction": float(missing.mean()), "test_cells": int(missing.sum()),
                          "observed_mean": mean, "observed_std": scale, "metrics": metrics, "baseline_tuning": tuning,
                          "routing": decision, "neural_seconds": timings, "elapsed_seconds": time.perf_counter() - started,
                          "prediction_keys": {f"prediction_{METHODS.index(method)}": method for method in predictions},
                          "prediction_sha256": sha256(prediction_path)}
                write_json(record_path, record)
                completed += 1
                print(json.dumps({"case": case_id, "seconds": record["elapsed_seconds"],
                                  "nrmse": {method: values["nrmse"] for method, values in metrics.items()}}), flush=True)
                summarize(output)
    _, summary = summarize(output)
    make_figures(output, summary)
    write_json(output / "status.json", {"state": "complete", "completed": completed, "expected": manifest["expected_cases"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/physics_cylinder_20261027/pilot")
    parser.add_argument("--checkpoint", type=Path, default=SINGLE)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--rates", type=float, nargs="+", default=[.2, .5, .8])
    parser.add_argument("--patterns", nargs="+", choices=["mcar", "sensor_outage"], default=["mcar", "sensor_outage"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--uncertainty-cells", type=int, default=10000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
