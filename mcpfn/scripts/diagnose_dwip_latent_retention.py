"""Fresh generic family/shape/support comparisons, never used for checkpoint selection."""
import argparse
from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from general_spectral_geometry import GeometrySpectrumGenerator
from rowcol_fixed_validation import apply_support_mask, preserved_rng
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256, write_json
from run_rowcol_controlled_experiments import REPO, WORK
from spectral_support_normalization import normalize_remaining_support
from tabimpute.interface import ImputePFN
from tabimpute.prior.training_set_generation import LatentFactorPrior

QUEUES = {
    "spectral24k": "dwip_spectral_20260915",
    "spectral48k_seed1": "dwip_spectral_horizon48k_20260916",
    "spectral48k_seed2": "dwip_spectral_seed2_48k_20260916",
    "point2_24k": "dwip_spectral_point2_20260916",
    "point2_48k_seed1": "dwip_spectral_point2_horizon48k_20260916",
    "point2_48k_seed2": "dwip_spectral_point2_seed2_48k_20260916",
    "continuous24k": "dwip_spectral_continuous_20260916",
    "continuous48k": "dwip_spectral_continuous_horizon48k_20260916",
}
SHAPES = {"anchor_range": ((10, 50), (5, 50)), "expanded_range": ((80, 320), (35, 100))}
LATENT_CONFIG = {
    "latent_rank_low": 1, "latent_rank_high": 11, "apply_feature_warping_prob": 0.,
    "apply_quantization_prob": 0., "latent_gaussian_sigma": 1., "latent_laplace_b": 1.,
    "latent_t_df_low": 2, "latent_t_df_high": 11, "latent_spike_p": .5,
    "latent_slab_sigma": 3., "latent_dirichlet_alpha": 1.,
}


def masked_case(observation, target, keep, mode, seed):
    source = observation.unsqueeze(0).masked_fill(~keep.unsqueeze(0), float("nan"))
    full_target = target.unsqueeze(0)
    if mode == "mcar":
        score_mask, stats = ~keep, {"support_episode": 0.}
    elif mode == "broad_support":
        masked, _, score_mask, stats = apply_support_mask(source, full_target, mode, seed)
        source = source.masked_fill(~torch.isfinite(masked), float("nan"))
        score_mask = score_mask[0]
    else:
        raise ValueError("Unknown support mode")
    _, _, _, scale = normalize_remaining_support(source, full_target, torch.isfinite(source), variance_df=1.)
    if not score_mask.any() or (score_mask & torch.isfinite(source[0])).any():
        raise ValueError("Scored cells must be missing, with at least one target")
    return {"source": source[0], "target": target.clone(), "mask": score_mask,
            "score_scale": scale[0], "support_stats": stats}


def prepare_cases(prior, matrices_per_cell, seed):
    if matrices_per_cell < 1:
        raise ValueError("Require positive matrices_per_cell")
    components = prior["mixed_priors"]
    probabilities = np.array([component["weight"] for component in components], dtype=float)
    probabilities /= probabilities.sum()
    cases = []
    for shape_index, (shape_name, (row_range, column_range)) in enumerate(SHAPES.items()):
        for matrix_index in range(matrices_per_cell):
            shape_rng = np.random.default_rng(np.random.SeedSequence([seed, shape_index, matrix_index]))
            rows = int(shape_rng.integers(row_range[0], row_range[1] + 1))
            columns = int(shape_rng.integers(column_range[0], column_range[1] + 1))
            keep = torch.from_numpy(shape_rng.random((rows, columns)) >= .4)
            dimensions = {"num_rows_low": rows, "num_rows_high": rows,
                          "num_cols_low": columns, "num_cols_high": columns}
            for family_index, family in enumerate(("latent", "spectral")):
                generation_seed = seed + 100000 * shape_index + 1000 * matrix_index + family_index
                with preserved_rng(generation_seed):
                    if family == "latent":
                        values = LatentFactorPrior({**LATENT_CONFIG, **dimensions}).generate_complete_matrix()
                        observation = torch.tensor(values.to_numpy(), dtype=torch.float32)
                        target, component_name = observation.clone(), "current_latent_defaults"
                    else:
                        component = components[int(np.random.choice(len(components), p=probabilities))]
                        generator = GeometrySpectrumGenerator({**component["config"], **dimensions})
                        observation, target = generator.generate_matrix_pair()
                        component_name = component["name"]
                if not torch.isfinite(observation).all() or not torch.isfinite(target).all():
                    raise FloatingPointError("Nonfinite generated matrix")
                for mode in ("mcar", "broad_support"):
                    case = masked_case(observation, target, keep, mode, seed + 2000000 + 1000 * shape_index + matrix_index)
                    group = f"{family}/{shape_name}/{mode}"
                    cases.append({**case, "id": f"{group}/{matrix_index}", "group": group,
                                  "family": family, "shape_range": shape_name, "mode": mode,
                                  "matrix_index": matrix_index, "component": component_name})
    return cases


def score_prediction(case, prediction):
    target, mask, scale = (case[name].numpy() for name in ("target", "mask", "score_scale"))
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or nonfinite values")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    normalized = error / scale.astype(np.float64)
    return {"id": case["id"], "group": case["group"], "n": int(mask.sum()),
            "sse": float((normalized[mask] ** 2).sum()), "raw_sse": float((error[mask] ** 2).sum())}


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    return {name: {"case_count": len(rows), "n": sum(row["n"] for row in rows),
                   "macro_case_rmse": float(np.mean([np.sqrt(row["sse"] / row["n"]) for row in rows])),
                   "pooled_rmse": float(np.sqrt(sum(row["sse"] for row in rows) / sum(row["n"] for row in rows))),
                   "raw_pooled_rmse": float(np.sqrt(sum(row["raw_sse"] for row in rows) / sum(row["n"] for row in rows)))}
            for name, rows in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/information_flow_diagnostics/dwip_latent_retention_20260916")
    parser.add_argument("--matrices-per-cell", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "diagnostic.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    spec = dwip_spec()
    checkpoints = {"original": Path(spec["checkpoint_path"]), **{
        label: WORK / f"tabimpute-{queue}-dwip1e5/best_validation_checkpoint.pth" for label, queue in QUEUES.items()}}
    origin = REPO / "runs/objective_queues/dwip_spectral_20260915/manifest.json"
    prior = json.loads(origin.read_text())["prior"]
    anchor = Path("/root/tabimpute_anchor3_named/anchor3_init_eval_manifest_20260420.json")
    sources = [Path(__file__), REPO / "scripts/general_spectral_geometry.py",
               REPO / "scripts/rowcol_fixed_validation.py", REPO / "scripts/spectral_support_normalization.py",
               REPO / "src/tabimpute/prior/training_set_generation.py",
               REPO / "src/tabimpute/prior/spectrum_training_set_generation.py",
               REPO / "src/tabimpute/interface.py", REPO / "src/tabimpute/model/model_new_stable.py",
               REPO / "src/tabimpute/model/bar_distribution.py"]
    manifest = {"seed": args.seed, "matrices_per_family_shape": args.matrices_per_cell,
        "shapes": {name: list(map(list, ranges)) for name, ranges in SHAPES.items()}, "latent_config": LATENT_CONFIG,
        "spectral_prior": prior, "prior_manifest_sha256": sha256(origin), "anchor_manifest_sha256": sha256(anchor),
        "checkpoint_sha256": {label: sha256(path) for label, path in checkpoints.items()},
        "source_sha256": {str(path): sha256(path) for path in sources}, "model_config": spec["model_config"],
        "protocol": "Raw public impute, FP32/BF16/default SDPA, mean, no windowing; MCAR0.4 shared across families, plus existing broad_support; score scales use only remaining observations with df1 pooled shrinkage, no clipping",
        "scope": "Fresh generic diagnostic, not a selector or benchmark-specific prior. Original-family provenance does not prove exact historical generator defaults. Spectral components retain their full noisy-observation/clean-target semantics.",
        "paired_structure": "Same shape and initial mask across families; each matrix shared across two support regimes and all models. Summaries are descriptive, not independent-observation significance tests."}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Diagnostic configuration changed; use fresh artifacts")
    write_json(manifest_path, manifest)
    cache, integrity = args.output / "cases.pth", args.output / "cases_sha256.json"
    if cache.exists():
        if not integrity.exists() or sha256(cache) != json.loads(integrity.read_text())["sha256"]:
            raise ValueError("Diagnostic case cache integrity mismatch")
        cases = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        cases = prepare_cases(prior, args.matrices_per_cell, args.seed)
        torch.save(cases, cache)
        write_json(integrity, {"sha256": sha256(cache)})
    if args.prepare_only:
        write_json(args.output / "status.json", {"state": "prepared", "case_count": len(cases)})
        return
    if (args.output / "results.json").exists():
        raise ValueError("Results already exist; inspect rather than overwrite")
    if gpu_processes():
        raise RuntimeError("GPU busy; no diagnostic started")
    results = {}
    try:
        for label, checkpoint in checkpoints.items():
            started = time.monotonic()
            imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                               json_config={"config": spec["model_config"]}, model_family="new_stable",
                               precision_mode="amp_bfloat16", prediction_stat="mean", max_num_rows=None, max_num_cols=None)
            if any(parameter.dtype != torch.float32 for parameter in imputer.model.parameters()):
                raise ValueError("Expected FP32 model weights")
            torch.cuda.reset_peak_memory_stats()
            records, predictions = [], {}
            for case in cases:
                write_json(args.output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                prediction = imputer.impute(case["source"].numpy().copy())
                records.append(score_prediction(case, prediction))
                predictions[case["id"]] = torch.from_numpy(prediction[case["mask"].numpy()].copy())
            torch.cuda.synchronize()
            results[label] = {"records": records, "groups": summarize(records),
                              "seconds": time.monotonic() - started,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            torch.save(predictions, args.output / f"{label}_predictions.pth")
            write_json(args.output / "results.json", results)
            if sha256(checkpoint) != manifest["checkpoint_sha256"][label]:
                raise RuntimeError("Checkpoint changed during evaluation")
            print(label, results[label]["seconds"], "seconds", flush=True)
            del imputer
            gc.collect()
            torch.cuda.empty_cache()
        if sha256(cache) != json.loads(integrity.read_text())["sha256"] or any(
                sha256(path) != expected for path, expected in manifest["source_sha256"].items()):
            raise RuntimeError("Source or case cache changed during evaluation")
        write_json(args.output / "status.json", {"state": "complete", "model_count": len(results), "case_count": len(cases)})
    except Exception as error:
        write_json(args.output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
