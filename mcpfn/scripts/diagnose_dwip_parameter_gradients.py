"""Report loss alignment on generic cached cases without updating DWIP weights."""
import argparse
from collections import Counter
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from rowcol_fixed_validation import preserved_rng
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


def select_cases(cases, per_group=2):
    if per_group < 1:
        raise ValueError("At least one case per group is required")
    counts, selected = Counter(), []
    for case in sorted(cases, key=lambda case: case["id"]):
        if not case["monitor_only"] and counts[case["group"]] < per_group:
            selected.append(case)
            counts[case["group"]] += 1
    if not selected:
        raise ValueError("No eligible synthetic cases")
    return selected


def parameter_group(name):
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "transformer_blocks" else parts[0]


def gradient_alignment(names, nll_gradients, point_gradients):
    if not (len(names) == len(nll_gradients) == len(point_gradients)):
        raise ValueError("Gradient lists must match parameter names")
    totals = {}
    unused = {"nll": 0, "point": 0}
    for name, nll_gradient, point_gradient in zip(names, nll_gradients, point_gradients):
        for label, gradient in (("nll", nll_gradient), ("point", point_gradient)):
            unused[label] += int(gradient is None)
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite {label} parameter gradient: {name}")
        nll_squared = float(nll_gradient.double().square().sum()) if nll_gradient is not None else 0.
        point_squared = float(point_gradient.double().square().sum()) if point_gradient is not None else 0.
        dot = float((nll_gradient.double() * point_gradient.double()).sum()) if (
            nll_gradient is not None and point_gradient is not None) else 0.
        for group in ("all", parameter_group(name)):
            total = totals.setdefault(group, [0., 0., 0.])
            for index, value in enumerate((nll_squared, point_squared, dot)):
                total[index] += value
    groups = {}
    for group, (nll_squared, point_squared, dot) in totals.items():
        nll_norm, point_norm = math.sqrt(nll_squared), math.sqrt(point_squared)
        groups[group] = {"nll_norm": nll_norm, "weighted_point_norm": point_norm,
                         "point_to_nll_ratio": point_norm / nll_norm if nll_norm else None,
                         "cosine": dot / (nll_norm * point_norm) if nll_norm * point_norm else None}
    return {"groups": groups, "unused_parameters": unused}


def summarize(records):
    summary = {}
    for group in records[0]["alignment"]["groups"]:
        entry = {}
        for key in ("point_to_nll_ratio", "cosine", "nll_norm", "weighted_point_norm"):
            values = [record["alignment"]["groups"][group][key] for record in records]
            finite = [value for value in values if value is not None and math.isfinite(value)]
            entry[key] = dict(zip(("min", "q25", "median", "q75", "max"),
                                 np.quantile(finite, [0., .25, .5, .75, 1.]).tolist())) if finite else None
        summary[group] = entry
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="dwip_parameter_gradients_20260916")
    parser.add_argument("--cases-per-group", type=int, default=2)
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.experiment
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if gpu_processes():
        raise RuntimeError("GPU is busy; no gradient diagnostic started")
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    spec = dwip_spec()
    checkpoints = {"original": Path(spec["checkpoint_path"]), "horizon48k_selected":
        WORK / "tabimpute-dwip_spectral_horizon48k_20260916-dwip1e5/best_validation_checkpoint.pth"}
    cache = REPO / "runs/objective_queues/dwip_spectral_20260915/validation.pth"
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    cases = select_cases(torch.load(cache, map_location="cpu", weights_only=True)["cases"], args.cases_per_group)
    sources = [Path(__file__), REPO / "src/tabimpute/model/model_new_stable.py",
               REPO / "src/tabimpute/model/bar_distribution.py", REPO / "scripts/rowcol_fixed_validation.py"]
    manifest = {"checkpoint_sha256": {name: sha256(path) for name, path in checkpoints.items()},
        "cache_sha256": sha256(cache), "borders_sha256": sha256(borders),
        "source_sha256": {str(path): sha256(path) for path in sources}, "model_config": spec["model_config"],
        "case_ids": [case["id"] for case in cases], "weights": {"nll": 1., "point_mse": .2},
        "protocol": "FP32 weights; BF16 autocast; eval mode; no optimizer, clipping or updates; first sorted cases per non-monitor synthetic group",
        "limitations": "Fixed cached validation masks, not the training distribution. CDF and Adam preconditioning are not measured. Ratios justify only a controlled hypothesis, not a predicted improvement.",
        "selection": "No benchmark data or scores used; descriptive diagnostic only"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Diagnostic inputs changed; use a new experiment")
    write_json(manifest_path, manifest)
    criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
    try:
        results = {}
        for label, checkpoint in checkpoints.items():
            started = time.monotonic()
            model = TabImputeModelStable(**spec["model_config"]).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError("Expected FP32 model parameters")
            names, parameters = zip(*model.named_parameters())
            torch.cuda.reset_peak_memory_stats()
            records = []
            for case in cases:
                write_json(root / "status.json", {"state": "diagnosing", "model": label, "case": case["id"]})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask].float()
                with preserved_rng(20260916), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask)
                nll = criterion(logits.float(), target).mean()
                point = (criterion.mean(logits.float()) - target).square().mean()
                nll_gradients = torch.autograd.grad(nll, parameters, retain_graph=True, allow_unused=True)
                point_gradients = torch.autograd.grad(.2 * point, parameters, allow_unused=True)
                record = {"id": case["id"], "group": case["group"], "shape": list(source.shape),
                          "target_count": target.numel(), "nll": float(nll.detach()), "mse": float(point.detach()),
                          "alignment": gradient_alignment(names, nll_gradients, point_gradients)}
                if not math.isfinite(record["nll"]) or not math.isfinite(record["mse"]):
                    raise FloatingPointError("Nonfinite diagnostic losses")
                records.append(record)
                write_json(root / f"{label}_cases.json", records)
                print(label, case["id"], record["alignment"]["groups"]["all"], flush=True)
                del logits, nll, point, nll_gradients, point_gradients
            if sha256(checkpoint) != manifest["checkpoint_sha256"][label]:
                raise RuntimeError("Source checkpoint changed during diagnostic")
            results[label] = {"case_count": len(records), "summary": summarize(records),
                              "seconds": time.monotonic() - started,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
            write_json(root / "results.json", results)
            del model, parameters
            torch.cuda.empty_cache()
        write_json(root / "status.json", {"state": "complete", "models": list(results)})
    except Exception as error:
        write_json(root / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
