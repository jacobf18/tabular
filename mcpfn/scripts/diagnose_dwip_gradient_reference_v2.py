"""Frozen reference precision audit with explicit nondeterministic backward drift."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.resources
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from diagnose_dwip_gradient_pooling import clipped, flatten_gradients, frozen_adam_direction, norm, pooling_vectors, projection, read_json, summarize
from diagnose_dwip_huber_training_alignment import point_losses
from diagnose_dwip_matched_objective_gradients import verify_hashes
from diagnose_dwip_training_objective import clipping_coefficient
from rowcol_fixed_validation import preserved_rng
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, write_json
from train_direct_pure_rabel_curriculum import cdf_alignment_loss
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def reference_ids():
    return [*range(64001, 64065), *range(64129, 65089)]


class GradientMoments:
    def __init__(self):
        self.count = 0
        self.total = None
        self.squared_norm_sum = 0.

    def update(self, vector):
        squared_norm = norm(vector) ** 2
        if self.total is None:
            self.total = torch.zeros_like(vector)
        if vector.shape != self.total.shape or vector.dtype != self.total.dtype or vector.device != self.total.device:
            raise ValueError("Gradient shape, dtype or device changed")
        self.total.add_(vector.detach())
        self.count += 1
        self.squared_norm_sum += squared_norm

    def mean(self):
        if self.count == 0:
            raise ValueError("Empty gradient reference")
        return self.total / self.count

    def statistics(self):
        mean_norm = norm(self.mean())
        variance = max(0., (self.squared_norm_sum - self.count * mean_norm ** 2) / (self.count - 1)) if self.count > 1 else None
        standard_error = math.sqrt(variance / self.count) if variance is not None else None
        return {"count": self.count, "mean_gradient_norm": mean_norm,
                "mean_squared_gradient_norm": self.squared_norm_sum / self.count,
                "sample_variance_trace": variance, "estimated_standard_error_l2": standard_error,
                "standard_error_to_mean_norm": standard_error / mean_norm if standard_error is not None and mean_norm else None,
                "unbiased_squared_mean_norm_estimate": mean_norm ** 2 - variance / self.count if variance is not None else None}


def check_reproduction(record, previous):
    for key in ("id", "cohort", "family", "shape", "support"):
        if record[key] != previous[key]:
            raise ValueError(f"Frozen case identity changed: {record['id']}/{key}")
    for name, value in previous["losses"].items():
        if not np.isclose(record["losses"][name], value, atol=1e-6, rtol=1e-5):
            raise ValueError(f"Frozen loss does not reproduce: {record['id']}/{name}")
    if not math.isfinite(record["preclip_norm"]) or record["preclip_norm"] < 0:
        raise ValueError("Invalid current gradient norm")
    return {"previous_norm": previous["preclip_norm"], "current_norm": record["preclip_norm"],
            "relative_norm_delta": record["preclip_norm"] / previous["preclip_norm"] - 1 if previous["preclip_norm"] else None}


def storage_guard(output):
    allocated = sum(path.stat().st_size for path in output.rglob("*") if path.is_file() and not path.is_symlink())
    if allocated > 1024**3 or shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("Reference diagnostic exceeded1GiB artifacts or20GiB reserve; preserve files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in {"", ".", ".."}:
        raise ValueError("Require a fresh single-directory experiment")
    previous = REPO / "runs/objective_queues/dwip_gradient_pooling_20260917"
    failed = REPO / "runs/objective_queues/dwip_gradient_reference1024_20260917"
    if read_json(previous / "status.json")["state"] != "complete" or gpu_processes():
        raise RuntimeError("Require completed source audit and idle GPU")
    proposal_path = previous / "next_reference_precision_proposal.json"
    proposal, prior = read_json(proposal_path), read_json(previous / "registration.json")
    hashes = {**proposal["input_sha256"], **prior["input_sha256"]}
    hashes.update({str(REPO / path): expected for path, expected in read_json(previous / "completion_audit.json")["output_sha256"].items()})
    hashes[str(proposal_path)] = sha256(proposal_path)
    hashes[str(Path(__file__).resolve())] = sha256(__file__)
    if read_json(failed / "status.json")["state"] != "error":
        raise ValueError("Require the preserved terminal prototype, not a live or restarted job")
    verify_hashes(read_json(failed / "registration.json")["input_sha256"])
    for path in (failed / "registration.json", failed / "status.json", failed / "cases_manifest.json",
                 failed / "new_reference_cases.pth", failed / "backward_reproducibility_diagnostic.json"):
        hashes[str(path)] = sha256(path)
    if hashes[str(failed / "new_reference_cases.pth")] != read_json(failed / "cases_manifest.json")["sha256"]:
        raise ValueError("Previously frozen appended cases changed")
    verify_hashes(hashes)
    output = REPO / "runs/objective_queues" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "registration.json").exists():
        raise ValueError("Preserve prior diagnostic artifacts; no restart")
    if shutil.disk_usage(output).free < 21 * 1024**3:
        raise RuntimeError("Require20GiB reserve plus1GiB diagnostic allowance")
    registration = {"utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes, "proposal": proposal,
        "original_protocol": prior, "reference_ids": reference_ids(), "probe_ids": list(range(64065, 64129)),
        "prefix_counts": [64, 128, 256, 512, 1024], "reference_split_sizes": [256] * 4,
        "reproduction_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "correction": "Prototype incorrectly required backward norm/cosine reproduction at1e-5. Three same-seed repetitions of four frozen cases keep forward losses unchanged but show up to0.004918 relative backward-vector differences. Keep strict identity/forward-loss gates, but report all backward norm/reference/projection drift rather than assuming deterministic gradients. This does not change inference or training kernels, seeds, data, losses, moments or model weights. The failed source and outputs remain preserved.",
        "protocol": "Only reference sample precision changes scientifically. Reuse128 original cases and ALL960 already-frozen appended cases from the failed prototype; no new generation or seed. Same two terminal64k FP32 model/moment pairs, train-mode BF16 autocast, cuDNN SDPA disabled, seeds and losses. No optimizer updates, benchmark data, adaptive samples or new pooling sizes.",
        "statistics": "FP32 gradient sums and means; FP64 norms/dots. Sample covariance trace estimates iid gradient-mean L2 standard error for BOTH exact-MSE and total gradients. Four disjoint256-case splits and every prefix reported; shared references make pooling-block comparisons correlated. Negative unbiased squared mean estimates are retained, not clipped. No significance or population-optimality claim."}
    write_json(output / "registration.json", registration)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        write_json(output / "status.json", {"state": "loading_frozen_cases", "updated_unix": time.time()})
        old_cases = torch.load(previous / "cases.pth", weights_only=True, map_location="cpu")
        if [case["id"] for case in old_cases] != list(range(64001, 64129)):
            raise ValueError("Frozen original case order changed")
        cache = failed / "new_reference_cases.pth"
        new_cases = torch.load(cache, weights_only=True, map_location="cpu")
        references = [case for case in old_cases if case["cohort"] == "reference"] + new_cases
        probes = [case for case in old_cases if case["cohort"] == "probe"]
        if [case["id"] for case in references] != reference_ids() or [case["id"] for case in probes] != registration["probe_ids"]:
            raise ValueError("Reference/probe population differs from preregistration")
        cases = references + probes
        write_json(output / "cases_manifest.json", {"sha256": hashes[str(cache)], "reference_count": len(references), "probe_count": len(probes),
            "families": {"reference": dict(Counter(case["family"] for case in references)), "probe": dict(Counter(case["family"] for case in probes))},
            "cases": [{key: value for key, value in case.items() if key not in ("source", "target", "mask")} |
                      {"shape": list(case["source"].shape), "target_count": int(case["mask"].sum())} for case in cases]})
        storage_guard(output)
        criterion = FullSupportBarDistribution(torch.load(importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
        results = {}
        for label, checkpoint_path in prior["checkpoint_paths"].items():
            payload = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            if payload["epoch"] != 64000:
                raise ValueError("Require original terminal64k state")
            model = DwipTrainingAdapter(**prior["model_config"]).cuda().train()
            model.load_state_dict(payload["model"], strict=True)
            parameters = tuple(model.parameters())
            if any(parameter.dtype != torch.float32 for parameter in parameters):
                raise ValueError("Require FP32 parameters")
            optimizer = {"param_groups": payload["optimizer"]["param_groups"], "state": {
                index: {key: value.cuda() if key in {"exp_avg", "exp_avg_sq"} else value for key, value in state.items()}
                for index, state in payload["optimizer"]["state"].items()}}
            old_records = {record["id"]: record for record in read_json(previous / f"{label}_cases.json")}
            old_blocks = read_json(previous / f"{label}_blocks.json")
            accumulators = {name: GradientMoments() for name in ("reference_total", "reference_mse")}
            split_accumulators = {name: GradientMoments() for name in accumulators}
            prefix_statistics, split_statistics, split_means, old_means, old_reference_drift = {}, [], [], {}, {}
            records, blocks, pending, pending_active, pending_directions = [], [], [], [], []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for case in cases:
                storage_guard(output)
                write_json(output / "status.json", {"state": "diagnosing", "model": label, "case": case["id"],
                                                   "cohort": case["cohort"], "updated_unix": time.time()})
                source, mask = case["source"].cuda(), case["mask"].cuda()
                target = case["target"].cuda()[mask]
                with preserved_rng(prior["forward_seed"] + case["id"]), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(source, decode_mask=mask, prior_index=None)
                means = criterion.mean(logits.float())
                losses = {"nll": criterion(logits.float(), target.float()).mean(), **point_losses(means, target, logits.dtype),
                          "cdf": cdf_alignment_loss(logits, target, criterion, chunk_size=prior["chunk_size"])}
                losses["total"] = losses["nll"] + prior["point_coefficients"][label] * losses["huber"] + .05 * losses["cdf"]
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite reference diagnostic loss")
                is_reference = case["cohort"] == "reference"
                gradient, active = flatten_gradients(parameters, torch.autograd.grad(losses["total"], parameters, retain_graph=is_reference, allow_unused=True))
                gradient_norm = norm(gradient)
                record = {"id": case["id"], "cohort": case["cohort"], "family": case["family"], "shape": list(source.shape),
                    "support": case["support"], "losses": {name: float(value.detach()) for name, value in losses.items()},
                    "preclip_norm": gradient_norm, "clip_coefficient": clipping_coefficient(gradient_norm, .5)}
                if case["id"] in old_records:
                    record["backward_norm_drift"] = check_reproduction(record, old_records[case["id"]])
                    record["previous_losses_reproduced"] = True
                if is_reference:
                    mse_gradient, _ = flatten_gradients(parameters, torch.autograd.grad(losses["mse"], parameters, allow_unused=True))
                    record["mse_gradient_norm"] = norm(mse_gradient)
                    for name, vector in (("reference_total", gradient), ("reference_mse", mse_gradient)):
                        accumulators[name].update(vector)
                        split_accumulators[name].update(vector)
                    count = accumulators["reference_total"].count
                    if count in registration["prefix_counts"]:
                        prefix_statistics[str(count)] = {name: accumulator.statistics() for name, accumulator in accumulators.items()}
                        write_json(output / f"{label}_reference_prefixes.json", prefix_statistics)
                    if count == 64:
                        old_means = {name: accumulator.mean().clone() for name, accumulator in accumulators.items()}
                        old_norms = read_json(previous / "results.json")[label]["reference_norms"]
                        old_reference_drift = {name: {"previous_norm": old_norms[name], "current_norm": norm(value),
                            "relative_norm_delta": norm(value) / old_norms[name] - 1} for name, value in old_means.items()}
                    if count % 256 == 0:
                        split_means.append({name: accumulator.mean().clone() for name, accumulator in split_accumulators.items()})
                        split_statistics.append({name: accumulator.statistics() for name, accumulator in split_accumulators.items()})
                        split_accumulators = {name: GradientMoments() for name in accumulators}
                    del mse_gradient
                else:
                    complete_means = {name: accumulator.mean() for name, accumulator in accumulators.items()}
                    direction = frozen_adam_direction(parameters, clipped(gradient), active, optimizer)
                    record["adam"] = {name: projection(direction, reference) for name, reference in complete_means.items()}
                    old_projection = {name: projection(direction, reference) for name, reference in old_means.items()}
                    record["old64_projection_drift"] = {name: {
                        "previous_cosine": old_records[case["id"]]["adam"][name]["descent_cosine"],
                        "current_cosine": old_projection[name]["descent_cosine"],
                        "cosine_delta": old_projection[name]["descent_cosine"] - old_records[case["id"]]["adam"][name]["descent_cosine"]} for name in old_means}
                    pending.append(gradient)
                    pending_active.append(active)
                    pending_directions.append(direction)
                    if len(pending) == 4:
                        pooled = pooling_vectors(pending)
                        combined_active = tuple(any(values) for values in zip(*pending_active))
                        directions = {"mean_single_adam": torch.stack(pending_directions).mean(0),
                            "adam_clip_mean": frozen_adam_direction(parameters, pooled["clip_mean_gradient"], combined_active, optimizer),
                            "adam_mean_clipped": frozen_adam_direction(parameters, pooled["mean_clipped_gradient"], combined_active, optimizer)}
                        block = {"ids": list(range(case["id"] - 3, case["id"] + 1)),
                                 **{method: {name: projection(value, reference) for name, reference in complete_means.items()} for method, value in directions.items()},
                                 "reference_splits": [{method: {name: projection(value, reference) for name, reference in split.items()} for method, value in directions.items()} for split in split_means]}
                        previous_block = old_blocks[len(blocks)]
                        block["old64_projection_drift"] = {method: {name: {
                            "previous_cosine": previous_block[method][name]["descent_cosine"],
                            "current_cosine": projection(value, reference)["descent_cosine"],
                            "cosine_delta": projection(value, reference)["descent_cosine"] - previous_block[method][name]["descent_cosine"]}
                            for name, reference in old_means.items()} for method, value in directions.items()}
                        blocks.append(block)
                        write_json(output / f"{label}_blocks.json", blocks)
                        pending, pending_active, pending_directions = [], [], []
                        del pooled, directions
                    del direction, complete_means
                records.append(record)
                if len(records) % 16 == 0:
                    write_json(output / f"{label}_cases.json", records)
                del logits, means, losses, gradient
            if len(blocks) != 16 or len(split_means) != 4 or len(records) != 1088 or pending:
                raise ValueError("Incomplete reference audit")
            for name, value in model.state_dict().items():
                if not torch.equal(value.cpu(), payload["model"][name]):
                    raise ValueError("Read-only audit changed model state")
            for index, state in optimizer["state"].items():
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].dtype != torch.float32 or not torch.equal(state[key].cpu(), payload["optimizer"]["state"][index][key]):
                        raise ValueError("Read-only audit changed FP32 Adam moments")
            verify_hashes(hashes)
            complete_means = {name: accumulator.mean() for name, accumulator in accumulators.items()}
            results[label] = {"summary": summarize(blocks), "prefix_statistics": prefix_statistics, "split_statistics": split_statistics,
                "split_pairwise_cosines": {name: [[projection(first[name], second[name])["descent_cosine"] for second in split_means] for first in split_means] for name in accumulators},
                "total_mse_mean_alignment": projection(complete_means["reference_total"], complete_means["reference_mse"]),
                "reference_split_summaries": [summarize([block["reference_splits"][index] for block in blocks]) for index in range(4)],
                "seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "old128_case_identities_and_losses_reproduced": True, "old64_reference_norm_drift": old_reference_drift,
                "max_abs_backward_norm_relative_drift": max(abs(record["backward_norm_drift"]["relative_norm_delta"]) for record in records if "backward_norm_drift" in record),
                "max_abs_old64_block_cosine_drift": max(abs(value["cosine_delta"]) for block in blocks for method in block["old64_projection_drift"].values() for value in method.values()),
                "model_and_adam_moments_unchanged": True, "statistics_limits": registration["statistics"]}
            write_json(output / "results.json", results)
            print(label, json.dumps({"summary": results[label]["summary"], "reference1024": prefix_statistics["1024"], "seconds": results[label]["seconds"]}), flush=True)
            del model, parameters, payload, optimizer, accumulators, split_accumulators, split_means, old_means, complete_means
            torch.cuda.empty_cache()
        storage_guard(output)
        write_json(output / "status.json", {"state": "complete", "models": list(results), "cases_per_model": len(cases),
            "all_inputs_unchanged": True, "all_original_case_identities_and_forward_losses_reproduced": True,
            "backward_drift_reported_not_assumed_deterministic": True,
            "model_and_adam_moments_unchanged": True, "updated_unix": time.time()})
    except Exception as error:
        write_json(output / "status.json", {"state": "error", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
