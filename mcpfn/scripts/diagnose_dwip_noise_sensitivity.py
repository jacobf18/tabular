"""Paired observation-noise sensitivity on generic independent synthetic cases."""
import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_dwip_fold_regression import QUEUES
from diagnose_dwip_parameter_gradients import select_cases
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from tabimpute.interface import ImputePFN

SCENARIOS = {"clean": (0., 1.), "dense_rms025": (.25, 1.), "dense_rms1": (1., 1.),
             "sparse_rms025": (.25, .01), "sparse_rms1": (1., .01)}


def perturb_observed(source, rms, probability, seed):
    if source.ndim != 2 or not np.isfinite(rms) or rms < 0 or not 0 < probability <= 1:
        raise ValueError("Invalid noise configuration")
    observed = np.isfinite(source)
    counts = observed.sum(0)
    mean = np.where(observed, source, 0.).sum(0) / counts.clip(1)
    centered = np.where(observed, source - mean, 0.)
    scale = np.sqrt((centered ** 2).sum(0) / counts.clip(1))
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal(source.shape)
    active = rng.random(source.shape) < probability
    standardized_noise = rms * gaussian * active / np.sqrt(probability)
    delta = np.where(observed, standardized_noise * scale, 0.)
    perturbed = source.copy() + delta
    eligible = observed & (scale > 0)
    stats = {"changed_observed_fraction": float((delta[observed] != 0).mean()),
             "eligible_observed_fraction": float(eligible.sum() / observed.sum()),
             "realized_noise_rms_in_original_observed_std": float(np.sqrt(np.mean(standardized_noise[eligible] ** 2))) if eligible.any() else 0.,
             "max_noise_in_original_observed_std": float(np.abs(standardized_noise[eligible]).max()) if eligible.any() else 0.}
    return perturbed, stats


def prepare_cases(caches, per_group, seed):
    prepared = []
    for population, cache in caches.items():
        cases = select_cases(torch.load(cache, map_location="cpu", weights_only=True)["cases"], per_group)
        for case in cases:
            if case["source"].shape[0] != 1:
                raise ValueError("Expected one matrix per cached case")
            identity = f"{population}/{case['id']}"
            case_seed = seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
            source = case["source"][0].double().numpy()
            item = {"id": identity, "population": population, "group": case["group"],
                    "target": case["target"][0].clone(), "mask": case["mask"][0].clone(),
                    "source": case["source"][0].clone(), "variants": {}}
            if np.isfinite(source)[item["mask"].numpy()].any():
                raise ValueError("Scoring mask includes observed values")
            for name, (rms, probability) in SCENARIOS.items():
                corrupted, stats = perturb_observed(source, rms, probability, case_seed)
                item["variants"][name] = {"source": torch.from_numpy(corrupted), "stats": stats}
            prepared.append(item)
    return prepared


def summarize(records):
    groups = {}
    for record in records:
        key = f"{record['population']}/{record['group']}"
        group = groups.setdefault(key, {})
        for name, values in record["scenarios"].items():
            bucket = group.setdefault(name, {"n": 0, "sse": 0., "prediction_change_sse": 0.})
            for field in bucket:
                bucket[field] += values[field]
    for group in groups.values():
        for values in group.values():
            values["rmse"] = (values["sse"] / values["n"]) ** .5
            values["prediction_change_rms"] = (values["prediction_change_sse"] / values["n"]) ** .5
        for values in group.values():
            values["rmse_relative_to_clean"] = values["rmse"] / group["clean"]["rmse"] if group["clean"]["rmse"] else None
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/information_flow_diagnostics/dwip_noise_sensitivity_20260916")
    parser.add_argument("--cases-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "diagnostic.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    spec = dwip_spec()
    checkpoints = {"original": Path(spec["checkpoint_path"]), **{
        label: WORK / f"tabimpute-{queue}-dwip1e5/best_validation_checkpoint.pth" for label, queue in QUEUES.items()}}
    caches = {"spectral": REPO / "runs/objective_queues/dwip_spectral_20260915/holdout.pth",
              "general": REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"}
    sources = [Path(__file__), REPO / "scripts/diagnose_dwip_parameter_gradients.py",
               REPO / "src/tabimpute/interface.py", REPO / "src/tabimpute/model/model_new_stable.py",
               REPO / "src/tabimpute/model/bar_distribution.py", REPO / "src/tabimpute/model/prediction_distribution.py",
               REPO / "src/tabimpute/data/borders.pt"]
    manifest = {"checkpoint_sha256": {label: sha256(path) for label, path in checkpoints.items()},
                "cache_sha256": {label: sha256(path) for label, path in caches.items()},
                "source_sha256": {str(path): sha256(path) for path in sources}, "model_config": spec["model_config"],
                "scenarios": {name: {"expected_rms": values[0], "probability": values[1]} for name, values in SCENARIOS.items()},
                "seed": args.seed, "cases_per_group": args.cases_per_group,
                "perturbation": "Observed-only original column population SD; zero-SD columns unchanged; shared Gaussian and Bernoulli draws across severities; dense and1%-sparse noise have matched expected RMS",
                "protocol": "Public unchanged ImputePFN mean, FP32 weights/BF16 autocast and default SDPA; re-estimates observed normalization after perturbation; fixed clean targets and scoring masks; no score clipping",
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled()},
                "purpose": "Report-only generic stress diagnostic; no benchmark data, training, selector alteration or checkpoint selection",
                "cautions": ["Fresh public-interface clean scores are not the cached direct-model validation scores.",
                             "Stress scales are generic relative-noise regimes, not estimates of benchmark errors.",
                             "Masks share matrices; descriptive differences are not independent significance tests."]}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Diagnostic configuration changed; use a new output directory")
    write_json(manifest_path, manifest)
    prepared_path, hash_path = args.output / "cases.pth", args.output / "cases_sha256.json"
    if prepared_path.exists():
        if not hash_path.exists() or json.loads(hash_path.read_text())["sha256"] != sha256(prepared_path):
            raise ValueError("Prepared inputs changed")
        prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
    else:
        prepared = prepare_cases(caches, args.cases_per_group, args.seed)
        torch.save(prepared, prepared_path)
        write_json(hash_path, {"sha256": sha256(prepared_path), "case_ids": [case["id"] for case in prepared]})
    if args.prepare_only:
        write_json(args.output / "status.json", {"state": "prepared", "case_count": len(prepared)})
        return
    if gpu_processes():
        raise RuntimeError("GPU is busy; no diagnostic started")
    results = {}
    try:
        for label, checkpoint in checkpoints.items():
            started = time.monotonic()
            imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                               json_config={"config": spec["model_config"]}, model_family="new_stable",
                               precision_mode="amp_bfloat16", prediction_stat="mean")
            if any(parameter.dtype != torch.float32 for parameter in imputer.model.parameters()):
                raise ValueError("Expected FP32 weights")
            torch.cuda.reset_peak_memory_stats()
            records, predictions = [], {}
            for case in prepared:
                write_json(args.output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                mask, target = case["mask"].numpy(), case["target"].double().numpy()
                record = {"id": case["id"], "population": case["population"], "group": case["group"], "scenarios": {}}
                predictions[case["id"]] = {}
                clean_prediction = None
                for name, variant in case["variants"].items():
                    prediction = imputer.impute(variant["source"].numpy().copy())
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError(f"Nonfinite predictions: {label}/{case['id']}/{name}")
                    if name == "clean":
                        clean_prediction = prediction[mask].copy()
                    difference = prediction[mask] - clean_prediction
                    record["scenarios"][name] = {"n": int(mask.sum()), "sse": float(((prediction[mask] - target[mask]) ** 2).sum()),
                                                "prediction_change_sse": float((difference ** 2).sum())}
                    predictions[case["id"]][name] = torch.from_numpy(prediction[mask].copy())
                records.append(record)
            torch.cuda.synchronize()
            results[label] = {"records": records, "groups": summarize(records), "seconds": time.monotonic() - started,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            torch.save(predictions, args.output / f"{label}_predictions.pth")
            write_json(args.output / "results.json", results)
            if sha256(checkpoint) != manifest["checkpoint_sha256"][label]:
                raise RuntimeError("Checkpoint changed during diagnosis")
            print(label, "complete", results[label]["seconds"], "seconds", flush=True)
            del imputer
            gc.collect()
            torch.cuda.empty_cache()
        write_json(args.output / "status.json", {"state": "complete", "model_count": len(results), "case_count": len(prepared)})
    except Exception as error:
        write_json(args.output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
