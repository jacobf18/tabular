"""Frozen MatrixFormer on the acquired, native-resolution USC multi-coil phantom."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.metrics import structural_similarity
import torch

import evaluate_cylinder_completion as shared
import evaluate_mri_completion as synthetic


ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "8137c274b6e2cc2829c97321e270e1933197bde70873e03739b116950b8c1184"
METHODS = synthetic.METHODS
KEYS = synthetic.KEYS
RANKS = [8, 16, 32, 64, 128]
METRICS = ["rss_image_nrmse", "complex_image_nrmse", "rss_ssim", "rss_psnr_db",
           "hidden_kspace_rmse", "hidden_kspace_relative_l2", "max_observed_error"]


def load_phantom(path):
    if shared.sha256(path) != DATA_SHA256:
        raise ValueError("Unexpected phantom checksum; this protocol binds the supplied acquisition")
    kspace = loadmat(path)["kData"]
    if kspace.shape != (256, 340, 16) or not np.iscomplexobj(kspace) or not np.isfinite(kspace).all():
        raise ValueError("Unexpected phantom array")
    return kspace.astype(np.complex128)


def coil_images(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(0, 1)),
                                      axes=(0, 1), norm="ortho"), axes=(0, 1))


def rss_image(kspace):
    return np.sqrt(np.sum(np.abs(coil_images(kspace)) ** 2, axis=2))


class MultiCoilSLift:
    """Rectangular, joint-coil extension of the independently implemented S map."""

    def __init__(self, spatial_shape, coils, radius=2):
        if len(spatial_shape) != 2 or any(size % 2 or size < 2 * radius + 4 for size in spatial_shape):
            raise ValueError("Require two even spatial dimensions with nonempty symmetric centers")
        if coils < 1 or radius < 1:
            raise ValueError("Invalid coils or radius")
        self.spatial_shape = tuple(spatial_shape)
        self.coils = coils
        self.radius = radius
        offsets = np.array([(row, column) for row in range(-radius, radius + 1)
                            for column in range(-radius, radius + 1)
                            if row ** 2 + column ** 2 <= radius ** 2])
        bounds = np.array(spatial_shape) // 2 - radius - 1
        centers = np.array([(row, column) for row in range(-bounds[0], bounds[0] + 1)
                            for column in range(-bounds[1], bounds[1] + 1)])
        positive = centers[:, None] + offsets[None] + np.array(spatial_shape) // 2
        negative = -centers[:, None] + offsets[None] + np.array(spatial_shape) // 2
        self.positive = positive[..., 0] * spatial_shape[1] + positive[..., 1]
        self.negative = negative[..., 0] * spatial_shape[1] + negative[..., 1]
        self.centers, self.neighbors = self.positive.shape
        self.width = self.neighbors * coils
        self.shape = (2 * self.centers, 2 * self.width)
        self.counts = np.bincount(np.concatenate([self.positive.ravel(), self.negative.ravel()]),
                                  minlength=int(np.prod(spatial_shape)))

    def forward(self, kspace):
        flat = np.asarray(kspace).reshape(-1, self.coils)
        positive = flat[self.positive].reshape(self.centers, self.width)
        negative = flat[self.negative].reshape(self.centers, self.width)
        difference, total = positive - negative, positive + negative
        return np.block([[difference.real, -difference.imag], [total.imag, total.real]])

    def observed(self, acquired):
        known = acquired.ravel()[self.positive] & acquired.ravel()[self.negative]
        return np.tile(np.repeat(known, self.coils, axis=1), (2, 2))

    def inverse(self, matrix):
        if matrix.shape != self.shape or not np.isfinite(matrix).all():
            raise ValueError("Invalid completed S-matrix")
        difference = matrix[:self.centers, :self.width] - 1j * matrix[:self.centers, self.width:]
        total = matrix[self.centers:, self.width:] + 1j * matrix[self.centers:, :self.width]
        positive = ((total + difference) / 2).reshape(-1, self.coils)
        negative = ((total - difference) / 2).reshape(-1, self.coils)
        result = np.zeros((int(np.prod(self.spatial_shape)), self.coils), dtype=np.complex128)
        for coil in range(self.coils):
            for indices, proposals in [(self.positive, positive), (self.negative, negative)]:
                result[:, coil] += np.bincount(indices.ravel(), weights=proposals[:, coil].real,
                                               minlength=len(result))
                result[:, coil] += 1j * np.bincount(indices.ravel(), weights=proposals[:, coil].imag,
                                                    minlength=len(result))
        result /= np.maximum(self.counts[:, None], 1)
        return result.reshape(*self.spatial_shape, self.coils)


class TorchSLift:
    def __init__(self, lift, device="cuda"):
        self.lift = lift
        self.positive = torch.as_tensor(lift.positive, device=device)
        self.negative = torch.as_tensor(lift.negative, device=device)
        self.counts = torch.as_tensor(np.maximum(lift.counts, 1), device=device)[:, None]

    def forward(self, kspace):
        flat = kspace.reshape(-1, self.lift.coils)
        positive = flat[self.positive].reshape(self.lift.centers, self.lift.width)
        negative = flat[self.negative].reshape(self.lift.centers, self.lift.width)
        difference, total = positive - negative, positive + negative
        return torch.cat([torch.cat([difference.real, -difference.imag], dim=1),
                          torch.cat([total.imag, total.real], dim=1)], dim=0)

    def inverse(self, matrix):
        centers, width = self.lift.centers, self.lift.width
        difference = torch.complex(matrix[:centers, :width], -matrix[:centers, width:])
        total = torch.complex(matrix[centers:, width:], matrix[centers:, :width])
        result = torch.zeros((len(self.lift.counts), self.lift.coils), dtype=total.dtype, device=matrix.device)
        result.index_add_(0, self.positive.ravel(), ((total + difference) / 2).reshape(-1, self.lift.coils))
        result.index_add_(0, self.negative.ravel(), ((total - difference) / 2).reshape(-1, self.lift.coils))
        return (result / self.counts).reshape(*self.lift.spatial_shape, self.lift.coils)


def acquisition_mask(shape, rate, pattern, seed, calibration=8):
    if not 0 < rate < 1 or not 0 < calibration < min(shape):
        raise ValueError("Invalid acquisition settings")
    central = np.zeros(shape, dtype=bool)
    row_start, column_start = [size // 2 - calibration // 2 for size in shape]
    random = np.random.default_rng(seed)
    if pattern == "points":
        central[row_start:row_start + calibration, column_start:column_start + calibration] = True
        candidates = np.flatnonzero(~central)
        budget = round((1 - rate) * np.prod(shape)) - int(central.sum())
        acquired = central.copy()
        if not 0 <= budget < len(candidates):
            raise ValueError("Infeasible point budget")
        acquired.ravel()[random.permutation(candidates)[:budget]] = True
    elif pattern == "lines":
        central[row_start:row_start + calibration] = True
        candidates = np.flatnonzero(~central[:, 0])
        budget = round((1 - rate) * shape[0]) - calibration
        if not 0 <= budget < len(candidates):
            raise ValueError("Infeasible line budget")
        acquired = central.copy()
        acquired[random.permutation(candidates)[:budget]] = True
    else:
        raise ValueError(pattern)
    return acquired, central


def observed_inputs(measured, acquired, lift):
    flat = measured.reshape(measured.shape[0], -1)
    raw = np.concatenate([flat.real, flat.imag], axis=1)
    lifted = lift.forward(np.nan_to_num(measured))
    lifted[~lift.observed(acquired)] = np.nan
    return raw, lifted


def axis_tiles(length, maximum, paired, seed):
    if maximum < 2 or (paired and length % 2):
        raise ValueError("Invalid chunk dimensions")
    if length <= maximum:
        return [np.arange(length)]
    units = length // 2 if paired else length
    step = maximum // 2 if paired else maximum
    order = np.random.default_rng(seed).permutation(units)
    return [np.concatenate([order[start:start + step], order[start:start + step] + units])
            if paired else order[start:start + step] for start in range(0, units, step)]


def observed_statistics(matrix):
    observed = matrix[np.isfinite(matrix)]
    if not observed.size:
        raise ValueError("No acquired entries")
    mean, scale = float(observed.mean()), float(observed.std())
    return mean, scale if scale > 0 else 1.0


@torch.inference_mode()
def neural_complete(matrix, model, distribution, lifted, maximum=1024):
    mean, scale = observed_statistics(matrix)
    rows = axis_tiles(matrix.shape[0], maximum, lifted, 20261027)
    columns = axis_tiles(matrix.shape[1], maximum, not lifted, 20261028)
    completed = matrix.copy()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    empty_tiles = 0
    for row_number, row_indices in enumerate(rows):
        for column_indices in columns:
            selection = np.ix_(row_indices, column_indices)
            tile = matrix[selection]
            missing = ~np.isfinite(tile)
            if not missing.any():
                continue
            if missing.all():
                completed[selection] = mean
                empty_tiles += 1
                continue
            values = torch.as_tensor(((tile - mean) / scale).astype(np.float32), device="cuda")[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model(values, return_embeddings=True)[0]
            features = encoded.reshape(-1, encoded.shape[-1])
            query_indices = torch.as_tensor(np.flatnonzero(missing), device="cuda")
            point = torch.empty(len(query_indices), device="cuda", dtype=torch.float32)
            for start in range(0, len(query_indices), 1024):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.decoder(features[query_indices[start:start + 1024]])
                point[start:start + len(logits)] = distribution.mean(logits.float())
            tile[missing] = point.cpu().numpy().astype(np.float64) * scale + mean
            completed[selection] = tile
            del values, encoded, features, logits, point, query_indices
        if lifted and (row_number + 1) % 20 == 0:
            print(f"  S inference tiles {row_number + 1}/{len(rows)}", flush=True)
    if not np.isfinite(completed).all():
        raise ValueError("Nonfinite neural prediction")
    torch.cuda.synchronize()
    return completed, {"seconds": time.perf_counter() - started, "mean": mean, "scale": scale,
                       "row_tiles": len(rows), "column_tiles": len(columns), "empty_tiles": empty_tiles,
                       "maximum_tile_shape": [max(map(len, rows)), max(map(len, columns))],
                       "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}


@torch.inference_mode()
def structured_lowrank(measured, lift, rank, max_iterations=100, tolerance=1e-5):
    if not 0 < rank < lift.lift.shape[1]:
        raise ValueError("Invalid rank")
    values = torch.as_tensor(measured, device=lift.positive.device)
    acquired = torch.isfinite(values)
    estimate = torch.where(acquired, values, 0)
    history = []
    for iteration in range(max_iterations):
        matrix = lift.forward(estimate)
        _, vectors = torch.linalg.eigh(matrix.T @ matrix)
        basis = vectors[:, -rank:]
        candidate = torch.where(acquired, values, lift.inverse((matrix @ basis) @ basis.T))
        change = float((torch.linalg.vector_norm(candidate - estimate)
                        / torch.linalg.vector_norm(estimate).clamp_min(1e-30)).item())
        estimate = candidate
        history.append(change)
        if change < tolerance:
            break
    return estimate.cpu().numpy(), {"rank": rank, "iterations": iteration + 1,
                                   "relative_updates": history, "converged": change < tolerance}


def fit_lowrank(measured, acquired, central, pattern, seed, lift):
    started = time.perf_counter()
    heldout = synthetic.validation_mask(acquired, central, pattern, seed)
    reduced = measured.copy()
    reduced[heldout] = complex(np.nan, np.nan)
    trials = []
    for rank in RANKS:
        prediction, diagnostics = structured_lowrank(reduced, lift, rank)
        trials.append({**diagnostics, "validation_rmse": float(np.sqrt(np.mean(
            np.abs(prediction[heldout] - measured[heldout]) ** 2)))})
        print(f"  rank {rank}: validation RMSE {trials[-1]['validation_rmse']:.6g}", flush=True)
    selected = min(trials, key=lambda record: record["validation_rmse"])
    prediction, diagnostics = structured_lowrank(measured, lift, selected["rank"])
    return prediction, heldout, {"trials": trials, "selected_rank": selected["rank"], "final": diagnostics,
                                "seconds_including_selection": time.perf_counter() - started}


def score_reconstruction(truth, prediction, acquired):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid reconstruction")
    data_error = float(np.max(np.abs(prediction[acquired] - truth[acquired])))
    if data_error != 0:
        raise ValueError("Reconstruction changed measured k-space")
    reference, reconstructed = rss_image(truth), rss_image(prediction)
    rss_error = reconstructed - reference
    error = prediction - truth
    hidden = error[~acquired]
    data_range = float(np.ptp(reference))
    return {"rss_image_nrmse": float(np.linalg.norm(rss_error) / np.linalg.norm(reference)),
            "complex_image_nrmse": float(np.linalg.norm(coil_images(error)) / np.linalg.norm(coil_images(truth))),
            "rss_ssim": float(structural_similarity(reference, reconstructed, data_range=data_range)),
            "rss_psnr_db": float(10 * np.log10(data_range ** 2 / max(float(np.mean(rss_error ** 2)), 1e-300))),
            "hidden_kspace_rmse": float(np.sqrt(np.mean(np.abs(hidden) ** 2))),
            "hidden_kspace_relative_l2": float(np.linalg.norm(hidden) / np.linalg.norm(truth[~acquired])),
            "max_observed_error": data_error}


def summarize(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_case.csv", index=False)
    summary = frame.groupby(["pattern", "rate", "method"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/root/phantom_data.mat"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require unique seeds")
    truth = load_phantom(args.input)
    lift = MultiCoilSLift(truth.shape[:2], truth.shape[2])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir()
    shutil.copy2(args.input, args.output / "phantom_data.mat")
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sources = [Path(__file__).resolve(), Path(shared.__file__).resolve(), Path(synthetic.__file__).resolve(),
               ROOT / "PHYSICS_MRI_REAL.md", ROOT / "scripts/report_real_mri_completion.py",
               ROOT / "src/tabimpute/model/model_new_stable.py", ROOT / "src/tabimpute/model/bar_distribution.py",
               ROOT / "src/tabimpute/data/borders.pt", shared.SINGLE, shared.SINGLE.with_name("config.json")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_source": str(args.input),
                "public_source": "https://mr.usc.edu/download/data/", "data_sha256": DATA_SHA256,
                "shape": list(truth.shape), "s_shape": list(lift.shape), "radius": 2, "calibration": 8,
                "patterns": ["points", "lines"], "rates": [.5, .75], "seeds": args.seeds,
                "methods": METHODS, "rank_grid": RANKS, "maximum_chunk": 1024,
                "chunk_seeds": [20261027, 20261028], "checkpoint": str(shared.SINGLE),
                "file_sha256": {str(path): shared.sha256(path) for path in sources},
                "packages": {name: metadata.version(name) for name in ["numpy", "scipy", "torch", "scikit-image"]},
                "gpu": torch.cuda.get_device_name()}
    if manifest["file_sha256"][str(shared.SINGLE)] != shared.SINGLE_SHA256:
        raise ValueError("Unexpected checkpoint")
    shared.write_json(args.output / "manifest.json", manifest)
    np.save(args.output / "reference_rss.npy", rss_image(truth))
    total, rows, completed, failures = len(args.seeds) * 4, [], 0, []
    shared.write_json(args.output / "status.json", {"status": "running", "completed": 0, "total": total})
    models, distribution = shared.load_models({"single": shared.SINGLE})
    torch_lift = TorchSLift(lift)
    for pattern in manifest["patterns"]:
        for seed in args.seeds:
            for rate in manifest["rates"]:
                case_id = f"{pattern}_seed{seed}_missing{rate:g}"
                print(f"Running {case_id}", flush=True)
                shared.write_json(args.output / "status.json", {"status": "running", "case": case_id,
                                  "completed": completed, "failed": len(failures), "total": total})
                try:
                    acquired, central = acquisition_mask(truth.shape[:2], rate, pattern, seed)
                    measured = np.where(acquired[..., None], truth, complex(np.nan, np.nan))
                    predictions = {"Zero-fill": np.nan_to_num(measured)}
                    lowrank, validation, tuning = fit_lowrank(measured, acquired, central, pattern, seed, torch_lift)
                    predictions["Structured low-rank S"] = lowrank
                    raw_input, lifted_input = observed_inputs(measured, acquired, lift)
                    raw, raw_details = neural_complete(raw_input, models["single"], distribution, False)
                    width = raw.shape[1] // 2
                    raw_kspace = (raw[:, :width] + 1j * raw[:, width:]).reshape(truth.shape)
                    predictions["MatrixFormer raw"] = synthetic.preserve_measurements(raw_kspace, measured)
                    lifted, lifted_details = neural_complete(lifted_input, models["single"], distribution, True)
                    predictions["MatrixFormer S"] = synthetic.preserve_measurements(lift.inverse(lifted), measured)
                    metrics = {method: score_reconstruction(truth, prediction, acquired)
                               for method, prediction in predictions.items()}
                    artifact = args.output / "cases" / f"{case_id}.npz"
                    np.savez_compressed(artifact, acquired=acquired, central=central, validation=validation,
                                        **{f"{KEYS[method]}_kspace": prediction for method, prediction in predictions.items()})
                    record = {"case_id": case_id, "pattern": pattern, "seed": seed, "rate": rate,
                              "metrics": metrics, "tuning": tuning, "raw_inference": raw_details,
                              "lifted_inference": lifted_details, "arrays_sha256": shared.sha256(artifact),
                              "actual_kspace_missing_fraction": float((~acquired).mean()),
                              "actual_s_missing_fraction": float(np.isnan(lifted_input).mean()),
                              "uncovered_missing_spatial_frequencies": int(((lift.counts.reshape(acquired.shape) == 0)
                                                                          & ~acquired).sum())}
                    shared.write_json(artifact.with_suffix(".json"), record)
                    for method, scores in metrics.items():
                        rows.append({"case_id": case_id, "pattern": pattern, "seed": seed, "rate": rate,
                                     "method": method, **scores})
                    summarize(rows, args.output)
                    completed += 1
                    print({method: round(values["rss_image_nrmse"], 6) for method, values in metrics.items()}, flush=True)
                    del raw_input, lifted_input, raw, lifted
                except Exception:
                    failure = {"case_id": case_id, "error": traceback.format_exc()}
                    failures.append(failure)
                    shared.write_json(args.output / "cases" / f"{case_id}.failure.json", failure)
                    print(failure["error"], flush=True)
                    torch.cuda.empty_cache()
    shared.write_json(args.output / "status.json", {"status": "complete" if not failures else "finished_with_failures",
                      "completed": completed, "failed": len(failures), "total": total})
    if failures:
        raise RuntimeError(f"{len(failures)} cases failed; preserved on disk")
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
