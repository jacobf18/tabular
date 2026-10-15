"""CPU first-step AdamW counterfactuals on all audited saved domain-gradient pairs."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

import diagnose_single_model_missingness_coverage as coverage
from dwip_teacher_mean_training import verify_hashes
import probe_single_model_domain_gradients as gradients
from recsys_evaluation_common import sha256, write_json
from streamed_large_matrix_training import learning_rate


def clipped(vector, sizes, maximum):
    if vector.ndim != 1 or sum(sizes) != vector.numel() or not torch.isfinite(vector).all():
        raise ValueError("Require a finite complete original gradient inventory")
    norms = torch.stack([torch.linalg.vector_norm(part) for part in vector.split(sizes)])
    norm = float(torch.linalg.vector_norm(norms))
    return vector * min(1., maximum / (norm + 1e-6))


def method_gradients(imputation, rating, sizes):
    if imputation.shape != rating.shape:
        raise ValueError("Both domains must retain identical parameter coordinates")
    joint = clipped(.5 * imputation + .5 * rating, sizes, .5)
    separate = clipped(.5 * imputation, sizes, .25) + clipped(.5 * rating, sizes, .25)
    return {"joint_clip": joint, "domain_clip_then_mean": clipped(separate, sizes, .5)}


def first_adam(parameter, gradient, group):
    if parameter.shape != gradient.shape or not torch.isfinite(parameter).all() or not torch.isfinite(gradient).all():
        raise ValueError("Require finite aligned AdamW parameters and gradients")
    shadow = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.AdamW([shadow], **group, foreach=False, fused=False)
    shadow.grad = gradient.detach().clone()
    optimizer.step()
    if float(optimizer.state[shadow]["step"]) != 1:
        raise ValueError("Never carry optimizer moments across diagnostic pairs")
    realized = parameter.detach() - shadow.detach()
    analytic = group["lr"] * (gradient.double() / (gradient.double().abs() + group["eps"])
                              + group["weight_decay"] * parameter.double())
    if not torch.isfinite(realized).all() or not torch.isfinite(analytic).all():
        raise FloatingPointError("Nonfinite counterfactual AdamW direction")
    return realized, analytic


def directional_statistics(direction, imputation, rating):
    norm = math.sqrt(max(gradients.dot(direction, direction), 0.))
    result = {"direction_norm": norm}
    for domain, vector in (("imputation", imputation), ("recsys", rating)):
        product = gradients.dot(vector, direction)
        domain_norm = math.sqrt(max(gradients.dot(vector, vector), 0.))
        result[domain] = {"dot_with_descent_direction": product, "linearized_loss_delta": -product,
                          "cosine": product / (norm * domain_norm) if norm * domain_norm else None}
    return result


def summarize(records):
    summary = {}
    for group in ("all", "legacy", "structured", *gradients.mixture.FAMILIES):
        selected = [record for record in records if group == "all" or record["group"] == group or record.get("family") == group]
        if not selected:
            raise ValueError("Keep every original diagnostic family and group")
        summary[group] = {"cases": len(selected), "methods": {}}
        for method in ("joint_clip", "domain_clip_then_mean"):
            summary[group]["methods"][method] = {
                direction: {domain: {"adverse_direction_cases": sum(record["methods"][method][direction][domain]["dot_with_descent_direction"] < 0 for record in selected),
                                    "defined_cases": len(selected)} for domain in ("imputation", "recsys")}
                for direction in ("sgd", "adam_analytic", "adam_cpu_realized")}
    return summary


def markdown(summary, rate):
    lines = ["# First-step AdamW directions from saved synthetic gradients", "",
        "All64 previously audited pairs are retained at the fixed V8 initialization. No new forward/backward pass, candidate checkpoint, training decision or benchmark evaluation occurs.",
        f"Each shadow CPU AdamW replay starts with fresh zero moments, LR{rate:g}, betas(0.9,0.999), epsilon1e-8 and weight decay0.02. Shadow states are discarded.",
        "Both the analytic first step and the realized FP32 CPU step are reported. This does not replay the fused-CUDA implementation or the evolving V10 optimizer trajectory.",
        "Directions refer to the original scalar training losses, not validation NRMSE/RMSE/MAE or actual finite-step loss changes.", "",
        "| Group | Pairs | Rating adverse: SGD joint | Rating adverse: SGD domain | Rating adverse: Adam CPU joint | Rating adverse: Adam CPU domain | Imputation adverse: Adam CPU joint | Imputation adverse: Adam CPU domain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in summary.items():
        methods = group["methods"]
        values = [methods[method][direction][domain]["adverse_direction_cases"] for method,direction,domain in
                  (("joint_clip","sgd","recsys"),("domain_clip_then_mean","sgd","recsys"),
                   ("joint_clip","adam_cpu_realized","recsys"),("domain_clip_then_mean","adam_cpu_realized","recsys"),
                   ("joint_clip","adam_cpu_realized","imputation"),("domain_clip_then_mean","adam_cpu_realized","imputation"))]
        lines.append(f'| {name} | {group["cases"]} | ' + " | ".join(str(value) for value in values) + " |")
    lines += ["", "No admission rule, V10 training recipe, selection criterion or frozen benchmark is modified by this read-only diagnostic.", ""]
    return "\n".join(lines)


def run(source, source_audit, output):
    if output.exists() or any(output.is_relative_to(parent) or parent.is_relative_to(output) for parent in (source, source_audit)):
        raise ValueError("Use a fresh separate diagnostic output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CUDA_VISIBLE_DEVICES='' for CPU-only counterfactuals")
    torch.set_num_threads(4)
    base = gradients.verify_registration(source)
    audit = gradients.read(source_audit / "audit.json")
    if not audit["complete"] or audit["pairs"] != 64 or audit["gradient_vectors"] != 128 or not audit["original_artifacts_unchanged"]:
        raise ValueError("Require the complete original128-vector audit")
    verify_hashes(audit["original_artifact_sha256"])
    inventory = gradients.read(source / "parameter_inventory.json")
    sizes = [count for name, shape, count in inventory]
    checkpoint = Path(base["base"]["parent_checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    parameter = torch.cat([state[name].flatten() for name, shape, count in inventory])
    del state
    if parameter.numel() != audit["elements_per_vector"] or parameter.dtype != torch.float32:
        raise ValueError("Retain every original FP32 parameter coordinate")
    rate = learning_rate(1, base["base"]["steps"], base["base"]["peak_lr"], base["base"]["warmup_steps"])
    group = {"lr": rate, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .02}
    paths = [source / "config.json", source / "parameter_inventory.json", source_audit / "audit.json", checkpoint]
    config = {"cases": base["cases"], "optimizer": group, "input_sha256": {str(path): sha256(path) for path in paths},
              "source_sha256": coverage.source_hashes(), "registered_before_direction_evaluation": True,
              "candidate_training_performed": False, "scope": "One counterfactual first AdamW step per method/pair; all shadow states discarded",
              "cuda_first_step_not_replayed": True, "v10_trajectory_not_replayed": True, "selection_policy_unchanged": True}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    records = []
    for specification in config["cases"]:
        index = specification["index"]
        original_record = gradients.read(source / f"pair_{index:03d}.json")
        path = Path(original_record["raw_gradients"])
        verify_hashes({str(path): original_record["raw_sha256"]})
        payload = torch.load(path, map_location="cpu", weights_only=True)
        imputation, rating = payload["imputation"], payload["recsys"]
        record = {"index": index, "group": specification["group"],
                  "family": specification["imputation"].get("structured", {}).get("family"),
                  "raw_gradient_sha256": original_record["raw_sha256"], "methods": {}}
        for method, gradient in method_gradients(imputation, rating, sizes).items():
            realized, analytic = first_adam(parameter, gradient, group)
            record["methods"][method] = {"sgd": directional_statistics(gradient, imputation, rating),
                "adam_analytic": directional_statistics(analytic, imputation, rating),
                "adam_cpu_realized": directional_statistics(realized, imputation, rating),
                "maximum_realized_minus_analytic": float((realized.double() - analytic).abs().max())}
            del realized, analytic
        records.append(record)
        write_json(output / f"pair_{index:03d}.json", record)
        write_json(output / "status.json", {"state": "replaying_cpu_shadow_steps", "completed": len(records), "expected": 64,
                                           "pid": os.getpid(), "updated_unix": time.time()})
        del payload, imputation, rating, gradient
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    verify_hashes(audit["original_artifact_sha256"])
    summary = summarize(records)
    write_json(output / "summary.json", {"complete": True, "cases": 64, "shadow_cpu_optimizer_steps": 128,
        "candidate_training_performed": False, "checkpoints_written": False, "cuda_initialized": torch.cuda.is_initialized(),
        "groups": summary, "goal_success_not_assessed": True, "real_benchmark_inputs_used": False,
        "original_artifacts_unchanged": True, "config_sha256": sha256(output / "config.json")})
    (output / "RESULTS.md").write_text(markdown(summary, rate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source, source_audit = args.source.resolve(), args.source_audit.resolve()
    if output.exists():
        raise FileExistsError("Never replace a completed or failed diagnostic")
    if any(output.is_relative_to(parent) or parent.is_relative_to(output) for parent in (source, source_audit)):
        raise ValueError("Never write diagnostic outputs inside either original artifact tree")
    try:
        run(source, source_audit, output)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
