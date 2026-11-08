"""Frozen MatrixFormer on raw and lifted phantom-MRI k-space."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from skimage.data import shepp_logan_phantom
from skimage.metrics import structural_similarity
from skimage.transform import resize
import torch

import evaluate_cylinder_completion as shared


ROOT = Path(__file__).resolve().parents[1]
SPECTRUM_SCRIPT = Path("/root/matrix-completion-benchmarks/scripts/plot_phantom_mri_spectra.py")
METHODS = ["Zero-fill", "Structured low-rank S", "MatrixFormer raw", "MatrixFormer S"]
KEYS = dict(zip(METHODS, ["zero", "lowrank", "raw", "lifted"]))
RANKS = [4, 8, 12, 18, 24]


def fft_image(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))


def ifft_kspace(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))


def make_phantom(size):
    amplitude = resize(shepp_logan_phantom(), (size, size), anti_aliasing=True, preserve_range=True)
    horizontal, vertical = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    phase = (.6 * horizontal + .4 * vertical + .3 * horizontal * vertical
             + .2 * (horizontal ** 2 - vertical ** 2))
    image = amplitude * np.exp(1j * phase)
    return image, fft_image(image)


class SLift:
    """Real-linear valid-neighborhood LORAKS S map and averaging pseudoinverse."""

    def __init__(self, size, radius=2):
        if size % 2 or size < 2 * radius + 4 or radius < 1:
            raise ValueError("Use an even image size with a nonempty symmetric valid-center grid")
        self.size = size
        self.radius = radius
        offsets = np.array([(row, column) for row in range(-radius, radius + 1)
                            for column in range(-radius, radius + 1)
                            if row ** 2 + column ** 2 <= radius ** 2])
        bound = size // 2 - radius - 1
        centers = np.array([(row, column) for row in range(-bound, bound + 1)
                            for column in range(-bound, bound + 1)])
        positive = centers[:, None] + offsets[None] + size // 2
        negative = -centers[:, None] + offsets[None] + size // 2
        self.positive = positive[..., 0] * size + positive[..., 1]
        self.negative = negative[..., 0] * size + negative[..., 1]
        self.centers, self.neighbors = self.positive.shape
        self.shape = (2 * self.centers, 2 * self.neighbors)
        self.counts = np.bincount(np.concatenate([self.positive.ravel(), self.negative.ravel()]),
                                  minlength=size ** 2)

    def forward(self, kspace):
        positive = np.asarray(kspace).ravel()[self.positive]
        negative = np.asarray(kspace).ravel()[self.negative]
        difference, total = positive - negative, positive + negative
        return np.block([[difference.real, -difference.imag], [total.imag, total.real]])

    def observed(self, acquired):
        known = acquired.ravel()[self.positive] & acquired.ravel()[self.negative]
        return np.tile(known, (2, 2))

    def inverse(self, matrix):
        if matrix.shape != self.shape or not np.isfinite(matrix).all():
            raise ValueError("Invalid completed S-matrix")
        difference = matrix[:self.centers, :self.neighbors] - 1j * matrix[:self.centers, self.neighbors:]
        total = matrix[self.centers:, self.neighbors:] + 1j * matrix[self.centers:, :self.neighbors]
        proposals = np.concatenate([((total + difference) / 2).ravel(), ((total - difference) / 2).ravel()])
        indices = np.concatenate([self.positive.ravel(), self.negative.ravel()])
        real = np.bincount(indices, weights=proposals.real, minlength=self.size ** 2)
        imaginary = np.bincount(indices, weights=proposals.imag, minlength=self.size ** 2)
        return ((real + 1j * imaginary) / np.maximum(self.counts, 1)).reshape(self.size, self.size)


def acquisition_mask(size, rate, pattern, seed, calibration=8):
    if not 0 < rate < 1 or not 0 < calibration < size:
        raise ValueError("Invalid missingness or calibration size")
    center_start = size // 2 - calibration // 2
    calibration_indices = slice(center_start, center_start + calibration)
    central = np.zeros((size, size), dtype=bool)
    random = np.random.default_rng(seed)
    if pattern == "points":
        central[calibration_indices, calibration_indices] = True
        budget = round((1 - rate) * size ** 2) - int(central.sum())
        candidates = np.flatnonzero(~central)
        acquired = central.copy()
        if not 0 <= budget < len(candidates):
            raise ValueError("Acquisition budget must include calibration and omit some samples")
        acquired.ravel()[random.permutation(candidates)[:budget]] = True
    elif pattern == "lines":
        central[calibration_indices, :] = True
        known_rows = central[:, 0]
        budget = round((1 - rate) * size) - int(known_rows.sum())
        candidates = np.flatnonzero(~known_rows)
        if not 0 <= budget < len(candidates):
            raise ValueError("Line budget must include calibration and omit some lines")
        selected = known_rows.copy()
        selected[random.permutation(candidates)[:budget]] = True
        acquired = np.broadcast_to(selected[:, None], central.shape).copy()
    else:
        raise ValueError(pattern)
    return acquired, central


def observed_inputs(measured, lift):
    acquired = np.isfinite(measured)
    raw = np.concatenate([measured.real, measured.imag], axis=1)
    known = lift.observed(acquired)
    lifted = lift.forward(np.nan_to_num(measured))
    lifted[~known] = np.nan
    return raw, lifted


def preserve_measurements(prediction, measured):
    result = prediction.copy()
    acquired = np.isfinite(measured)
    result[acquired] = measured[acquired]
    return result


@torch.inference_mode()
def neural_complete(matrix, model, distribution):
    normalized, mean, scale = shared.normalize_observed(matrix)
    values = torch.from_numpy(normalized.astype(np.float32)).cuda().unsqueeze(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = model(values, return_embeddings=True)[0]
        features = encoded.reshape(-1, encoded.shape[-1])
    point = torch.empty(matrix.size, device="cuda", dtype=torch.float32)
    for start in range(0, matrix.size, 1024):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.decoder(features[start:start + 1024])
        point[start:start + len(logits)] = distribution.mean(logits.float())
    raw = point.cpu().numpy().reshape(matrix.shape).astype(np.float64) * scale + mean
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite model output")
    completed = np.where(np.isfinite(matrix), matrix, raw)
    torch.cuda.synchronize()
    diagnostics = {"seconds": time.perf_counter() - started, "observed_mean": mean, "observed_std": scale,
                   "matrix_shape": list(matrix.shape), "observed_fraction": float(np.isfinite(matrix).mean()),
                   "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
    return completed, raw, diagnostics


def truncated_rank(matrix, rank):
    _, vectors = np.linalg.eigh(matrix.T @ matrix)
    basis = vectors[:, -rank:]
    return (matrix @ basis) @ basis.T


def structured_lowrank(measured, lift, rank, max_iterations=100, tolerance=1e-5):
    if not 0 < rank < lift.shape[1]:
        raise ValueError("Invalid S-matrix rank")
    estimate = np.nan_to_num(measured)
    history = []
    for iteration in range(max_iterations):
        lowrank = truncated_rank(lift.forward(estimate), rank)
        candidate = preserve_measurements(lift.inverse(lowrank), measured)
        change = float(np.linalg.norm(candidate - estimate) / max(np.linalg.norm(estimate), 1e-12))
        history.append(change)
        estimate = candidate
        if change < tolerance:
            break
    return estimate, {"rank": rank, "iterations": iteration + 1, "relative_updates": history,
                      "converged": change < tolerance}


def validation_mask(acquired, central, pattern, seed):
    eligible = acquired & ~central
    random = np.random.default_rng(seed + 1000)
    heldout = np.zeros_like(acquired)
    if pattern == "lines":
        candidates = np.flatnonzero(eligible.all(axis=1))
        if not len(candidates):
            raise ValueError("No noncalibration lines for rank validation")
        selected = random.permutation(candidates)[:max(1, round(.1 * len(candidates)))]
        heldout[selected] = True
    else:
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError("No noncalibration samples for rank validation")
        selected = random.permutation(candidates)[:max(1, round(.1 * len(candidates)))]
        heldout.ravel()[selected] = True
    return heldout


def fit_lowrank(measured, central, pattern, seed, lift):
    started = time.perf_counter()
    heldout = validation_mask(np.isfinite(measured), central, pattern, seed)
    reduced = measured.copy()
    reduced[heldout] = complex(np.nan, np.nan)
    trials = []
    for rank in RANKS:
        candidate, diagnostics = structured_lowrank(reduced, lift, rank)
        error = candidate[heldout] - measured[heldout]
        trials.append({**diagnostics, "validation_rmse": float(np.sqrt(np.mean(np.abs(error) ** 2)))})
    selected = min(trials, key=lambda trial: trial["validation_rmse"])
    prediction, final = structured_lowrank(measured, lift, selected["rank"])
    return prediction, heldout, {"trials": trials, "selected_rank": selected["rank"], "final": final,
                                "seconds_including_selection": time.perf_counter() - started}


def score_reconstruction(truth_image, truth_kspace, prediction, acquired):
    image = ifft_kspace(prediction)
    target_magnitude, magnitude = np.abs(truth_image), np.abs(image)
    magnitude_error = magnitude - target_magnitude
    missing = ~acquired
    hidden_error = prediction[missing] - truth_kspace[missing]
    data_error = float(np.max(np.abs(prediction[acquired] - truth_kspace[acquired])))
    if data_error > 1e-10 or not np.isfinite(prediction).all():
        raise ValueError("Reconstruction does not preserve acquired measurements")
    data_range = float(np.ptp(target_magnitude))
    magnitude_mse = float(np.mean(magnitude_error ** 2))
    return {"complex_image_nrmse": float(np.linalg.norm(image - truth_image) / np.linalg.norm(truth_image)),
            "magnitude_image_nrmse": float(np.linalg.norm(magnitude_error) / np.linalg.norm(target_magnitude)),
            "magnitude_ssim": float(structural_similarity(target_magnitude, magnitude, data_range=data_range)),
            "magnitude_psnr_db": float(10 * np.log10(data_range ** 2 / max(magnitude_mse, 1e-30))),
            "hidden_kspace_rmse": float(np.sqrt(np.mean(np.abs(hidden_error) ** 2))),
            "hidden_kspace_relative_l2": float(np.linalg.norm(hidden_error) / np.linalg.norm(truth_kspace[missing])),
            "max_observed_error": data_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/physics_mri_20261027/pilot")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rates", nargs="+", type=float, default=[.5, .75])
    parser.add_argument("--patterns", nargs="+", choices=["points", "lines"], default=["points", "lines"])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if any(len(set(values)) != len(values) for values in [args.seeds, args.rates, args.patterns]):
        raise ValueError("Duplicate cases")
    lift = SLift(args.size)
    for pattern in args.patterns:
        for rate in args.rates:
            acquired, central = acquisition_mask(args.size, rate, pattern, args.seeds[0])
            validation_mask(acquired, central, pattern, args.seeds[0])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sources = [Path(__file__).resolve(), Path(shared.__file__).resolve(), ROOT / "PHYSICS_MRI.md", SPECTRUM_SCRIPT,
               ROOT / "src/tabimpute/model/model_new_stable.py", ROOT / "src/tabimpute/model/bar_distribution.py",
               ROOT / "src/tabimpute/data/borders.pt", shared.SINGLE, shared.SINGLE.with_name("config.json")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "size": args.size,
                "radius": 2, "calibration": 8, "seeds": args.seeds, "rates": args.rates, "patterns": args.patterns,
                "methods": METHODS, "rank_grid": RANKS, "s_matrix_shape": list(lift.shape),
                "checkpoint": str(shared.SINGLE), "normalization": "observed-only global mean/population SD",
                "file_sha256": {str(path): shared.sha256(path) for path in sources},
                "packages": {name: metadata.version(name) for name in ["numpy", "scipy", "scikit-image", "torch", "pandas"]},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gpu": torch.cuda.get_device_name()}
    if manifest["file_sha256"][str(shared.SINGLE)] != shared.SINGLE_SHA256:
        raise ValueError("Unexpected checkpoint")
    shared.write_json(args.output / "manifest.json", manifest)
    total = len(args.seeds) * len(args.rates) * len(args.patterns)
    shared.write_json(args.output / "status.json", {"status": "running", "completed": 0, "total": total})
    image, kspace = make_phantom(args.size)
    truth_s = lift.forward(kspace)
    np.save(args.output / "image.npy", image)
    np.save(args.output / "raw_kspace.npy", kspace)
    np.save(args.output / "lifted_s_matrix.npy", truth_s)
    subprocess.run([sys.executable, str(SPECTRUM_SCRIPT), "--raw-matrix-path", str(args.output / "raw_kspace.npy"),
                    "--lifted-matrix-path", str(args.output / "lifted_s_matrix.npy"),
                    "--output-dir", str(args.output / "spectra_truth")], check=True)
    models, distribution = shared.load_models({"single": shared.SINGLE})
    rows, completed_cases = [], 0
    for pattern in args.patterns:
        for seed in args.seeds:
            for rate in args.rates:
                case_id = f"{pattern}_seed{seed}_missing{rate:g}"
                print(f"Running {case_id}", flush=True)
                acquired, central = acquisition_mask(args.size, rate, pattern, seed)
                measured = np.where(acquired, kspace, complex(np.nan, np.nan))
                raw_input, lifted_input = observed_inputs(measured, lift)
                lowrank, validation, tuning = fit_lowrank(measured, central, pattern, seed, lift)
                raw_completed, raw_prediction, raw_diagnostics = neural_complete(raw_input, models["single"], distribution)
                lifted_completed, lifted_prediction, lifted_diagnostics = neural_complete(lifted_input, models["single"], distribution)
                predictions = {"Zero-fill": np.nan_to_num(measured), "Structured low-rank S": lowrank,
                               "MatrixFormer raw": preserve_measurements(raw_completed[:, :args.size] + 1j * raw_completed[:, args.size:], measured),
                               "MatrixFormer S": preserve_measurements(lift.inverse(lifted_completed), measured)}
                arrays = {"acquired": acquired, "central": central, "validation": validation,
                          "raw_input": raw_input, "lifted_input": lifted_input,
                          "raw_prediction": raw_prediction, "lifted_prediction": lifted_prediction}
                metrics = {}
                for method, prediction in predictions.items():
                    key = KEYS[method]
                    arrays[f"{key}_kspace"] = prediction
                    arrays[f"{key}_s_singular_values"] = np.linalg.svd(lift.forward(prediction), compute_uv=False)
                    arrays[f"{key}_raw_singular_values"] = np.linalg.svd(prediction, compute_uv=False)
                    metrics[method] = score_reconstruction(image, kspace, prediction, acquired)
                    rows.append({"case_id": case_id, "pattern": pattern, "rate": rate, "seed": seed,
                                 "method": method, **metrics[method]})
                artifact = args.output / "cases" / f"{case_id}.npz"
                np.savez_compressed(artifact, **arrays)
                record = {"case_id": case_id, "pattern": pattern, "rate": rate, "seed": seed, "metrics": metrics,
                          "tuning": tuning, "raw_inference": raw_diagnostics, "lifted_inference": lifted_diagnostics,
                          "actual_kspace_missing_fraction": float((~acquired).mean()),
                          "actual_s_missing_fraction": float(np.isnan(lifted_input).mean()),
                          "uncovered_missing_kspace": int(((lift.counts.reshape(acquired.shape) == 0) & ~acquired).sum()),
                          "arrays_sha256": shared.sha256(artifact)}
                shared.write_json(artifact.with_suffix(".json"), record)
                frame = pd.DataFrame(rows)
                frame.to_csv(args.output / "per_case.csv", index=False)
                completed_cases += 1
                shared.write_json(args.output / "status.json", {"status": "running", "completed": completed_cases, "total": total})
                print({method: round(scores["complex_image_nrmse"], 5) for method, scores in metrics.items()}, flush=True)
    metric_names = list(metrics[METHODS[0]])
    summary = frame.groupby(["pattern", "rate", "method"])[metric_names].agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    shared.write_json(args.output / "status.json", {"status": "complete", "completed": completed_cases, "total": total})
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
