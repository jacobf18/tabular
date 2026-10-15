"""CPU balanced-batch directions from the complete previously audited gradient bank."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from dwip_teacher_mean_training import verify_hashes
import probe_single_model_adam_directions as adam
import probe_single_model_domain_gradients as gradients
from recsys_evaluation_common import sha256, write_csv, write_json
from streamed_large_matrix_training import learning_rate


METHODS = ("mean_of_clipped_examples", "clip_of_domain_means")
DIRECTIONS = ("sgd", "adam_analytic", "adam_cpu_realized")
POLICY = "All64 saved pairs at fixed selected V8 weights; imputation half legacy and half uniform over twelve structured families, ratings uniform over64 draws; equal domain weights. Compare averaging individually clipped domain gradients versus clipping accumulated domain means. No new data, forward/backward pass, trained checkpoint or model selection."
LIMITATIONS = "Exploratory reuse of a small stratified synthetic gradient bank, not a population estimate, validation-metric derivative, fused-CUDA replay, evolving-AdamW replay or finite-step quality measurement. Shadow AdamW tensors are discarded."


def read(path):
    return json.loads(Path(path).read_text())


def validate_records(records):
    if len(records) != 64 or [record["index"] for record in records] != list(range(64)):
        raise ValueError("Require all64 original ordered pairs")
    legacy = [record for record in records if record["group"] == "legacy"]
    structured = [record for record in records if record["group"] == "structured"]
    if len(legacy) != 16 or len(structured) != 48:
        raise ValueError("Keep the original16 legacy and48 structured episodes")
    for family in gradients.mixture.FAMILIES:
        selected = [record for record in structured if record["metadata"]["imputation"]["family"] == family]
        if len(selected) != 4 or {record["metadata"]["imputation"]["component"] for record in selected} != set(gradients.mixture.COMPONENTS):
            raise ValueError("Require all four generators for each of twelve families")
    for record in records:
        for domain in ("imputation", "recsys"):
            if record["metadata"][domain]["domain"] != domain or record["metadata"][domain]["gradient_data"] != "synthetic":
                raise ValueError("Only the original synthetic domain gradients may be combined")


def weights(record):
    if record["group"] not in ("legacy", "structured"):
        raise ValueError("Unknown diagnostic stratum")
    return {"imputation": .5 / (16 if record["group"] == "legacy" else 48), "recsys": 1 / 64}


def accumulate(records, load, sizes):
    validate_records(records)
    elements = sum(sizes)
    means = {domain: torch.zeros(elements, dtype=torch.float64) for domain in ("imputation", "recsys")}
    clipped_mean = torch.zeros(elements, dtype=torch.float64)
    for record in records:
        payload = load(record)
        proportions = weights(record)
        for domain in means:
            vector = payload[domain]
            if vector.shape != (elements,) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
                raise ValueError("Keep every finite original FP32 gradient coordinate")
            means[domain].add_(vector.double(), alpha=proportions[domain])
            clipped = adam.clipped(.5 * vector, sizes, .25)
            clipped_mean.add_(clipped.double(), alpha=proportions[domain])
        del payload
    return {"mean_of_clipped_examples": adam.clipped(clipped_mean.float(), sizes, .5),
            "clip_of_domain_means": adam.method_gradients(means["imputation"].float(), means["recsys"].float(), sizes)["domain_clip_then_mean"]}


def directions(parameter, methods, optimizer):
    result = {}
    for method in METHODS:
        realized, analytic = adam.first_adam(parameter, methods[method], optimizer)
        result[method] = {"sgd": methods[method], "adam_analytic": analytic, "adam_cpu_realized": realized}
    return result


def measure(records, load, methods):
    norms = {method: {kind: math.sqrt(max(gradients.dot(vector, vector), 0.)) for kind, vector in values.items()}
             for method, values in methods.items()}
    rows = []
    for record in records:
        payload = load(record)
        family = record["metadata"]["imputation"].get("family", "legacy") if record["group"] == "structured" else "legacy"
        for domain in ("imputation", "recsys"):
            vector = payload[domain]
            norm = math.sqrt(max(gradients.dot(vector, vector), 0.))
            group = family if domain == "imputation" else "recsys"
            for method in METHODS:
                for kind in DIRECTIONS:
                    product = gradients.dot(vector, methods[method][kind])
                    denominator = norm * norms[method][kind]
                    rows.append({"pair": record["index"], "group": group, "domain": domain, "method": method, "direction": kind,
                                 "dot_with_descent_direction": product, "linearized_loss_delta": -product,
                                 "cosine": product / denominator if denominator else None, "adverse": product < 0})
        del payload
    return rows


def summarize(rows):
    if len(rows) != 64 * 2 * len(METHODS) * len(DIRECTIONS):
        raise ValueError("Keep all768 directional measurements")
    keys = [(row["pair"], row["domain"], row["method"], row["direction"]) for row in rows]
    expected = {(index, domain, method, kind) for index in range(64) for domain in ("imputation", "recsys")
                for method in METHODS for kind in DIRECTIONS}
    if set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError("Directional identities must be complete and unique")
    summaries = []
    for group in ("legacy", *gradients.mixture.FAMILIES, "recsys"):
        for method in METHODS:
            for kind in DIRECTIONS:
                selected = [row for row in rows if row["group"] == group and row["method"] == method and row["direction"] == kind]
                expected_count = 16 if group == "legacy" else 64 if group == "recsys" else 4
                if len(selected) != expected_count:
                    raise ValueError("Every group must keep every original gradient")
                dots = [row["dot_with_descent_direction"] for row in selected]
                if any(not math.isfinite(value) for value in dots):
                    raise ValueError("Never omit a nonfinite dot product")
                summaries.append({"group": group, "method": method, "direction": kind, "cases": len(selected),
                                  "adverse_cases": sum(value < 0 for value in dots), "mean_dot": sum(dots) / len(dots),
                                  "mean_linearized_loss_delta": -sum(dots) / len(dots), "group_mean_adverse": sum(dots) < 0})
    return summaries


def markdown(summary):
    lines = ["# Balanced-batch directions from saved synthetic gradients", "", POLICY, "", LIMITATIONS, "",
             "All128 original gradient vectors contribute. No candidate checkpoint is produced or evaluated.",
             "Positive dot products indicate first-order descent of the original scalar training loss, not NRMSE/RMSE/MAE improvement.",
             "Negative group means remain visible alongside individual-example conflicts; no admission or promotion rule is defined here.", "",
             "| Group | Method | Direction | Cases | Adverse cases | Mean descent dot | Group mean adverse |",
             "|---|---|---|---:|---:|---:|:---:|"]
    for row in summary:
        lines.append(f'| {row["group"]} | {row["method"]} | {row["direction"]} | {row["cases"]} | {row["adverse_cases"]} | {row["mean_dot"]:.10g} | {row["group_mean_adverse"]} |')
    lines += ["", "This read-only counterfactual does not alter the active displacement grid, its guards, any confirmation reservation, or the frozen real benchmarks.", ""]
    return "\n".join(lines)


def run(source, source_audit, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CPU-only execution")
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output) for path in (source, source_audit)):
        raise ValueError("Use a new separate diagnostic output")
    torch.set_num_threads(4)
    base = gradients.verify_registration(source)
    audit = read(source_audit / "audit.json")
    if (not audit["complete"] or audit["pairs"] != 64 or audit["gradient_vectors"] != 128
            or not audit["all_original_coordinates_remeasured"] or not audit["original_artifacts_unchanged"] or audit["cuda_initialized"]):
        raise ValueError("Require the complete original CPU gradient audit")
    hashes = dict(audit["original_artifact_sha256"])
    verify_hashes(hashes)
    records = [read(source / f"pair_{index:03d}.json") for index in range(64)]
    validate_records(records)
    inventory = read(source / "parameter_inventory.json")
    sizes = [count for name, shape, count in inventory]
    if sum(sizes) != audit["elements_per_vector"]:
        raise ValueError("Original parameter inventory changed")
    checkpoint = Path(base["base"]["parent_checkpoint"])
    rate = learning_rate(1, base["base"]["steps"], base["base"]["peak_lr"], base["base"]["warmup_steps"])
    optimizer = {"lr": rate, "betas": (.9, .999), "eps": 1e-8, "weight_decay": base["base"]["weight_decay"]}
    for path in (checkpoint, source_audit / "audit.json", Path(__file__).resolve(), Path(adam.__file__).resolve(), Path(gradients.__file__).resolve()):
        hashes[str(path)] = sha256(path)
    output.mkdir(parents=True)
    write_json(output / "config.json", {"source": str(source), "audit": str(source_audit), "policy": POLICY, "limitations": LIMITATIONS,
               "methods": METHODS, "directions": DIRECTIONS, "optimizer": optimizer, "input_sha256": hashes,
               "training_performed": False, "new_forward_backward_passes": 0, "checkpoint_selection_performed": False})
    write_json(output / "registration.json", {"registered_unix": time.time(), "config_sha256": sha256(output / "config.json"),
               "counterfactual_directions_computed": False})
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    try:
        def load(record):
            path = record["raw_gradients"]
            if hashes.get(path) != record["raw_sha256"]:
                raise ValueError("Every raw vector must be bound by the original audit")
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

        write_json(output / "status.json", {"state": "accumulating_original_gradients", "pid": os.getpid(), "updated_unix": time.time()})
        methods = accumulate(records, load, sizes)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        parameter = torch.cat([state[name].flatten() for name, shape, count in inventory])
        del state
        if parameter.dtype != torch.float32:
            raise ValueError("Keep original FP32 parameter arithmetic")
        methods = directions(parameter, methods, optimizer)
        del parameter
        write_json(output / "status.json", {"state": "measuring_all_original_vectors", "pid": os.getpid(), "updated_unix": time.time()})
        rows = measure(records, load, methods)
        summaries = summarize(rows)
        verify_hashes(hashes)
        write_csv(output / "directions.csv", rows)
        write_csv(output / "groups.csv", summaries)
        (output / "RESULTS.md").write_text(markdown(summaries))
        write_json(output / "summary.json", {"complete": True, "groups": summaries, "pairs": 64, "gradient_vectors": 128,
                   "directional_measurements": len(rows), "original_artifacts_unchanged": True,
                   "cuda_initialized": torch.cuda.is_initialized(), "training_performed": False, "checkpoint_selection_performed": False,
                   "new_forward_backward_passes": 0, "goal_success_not_assessed": True, "limitations": LIMITATIONS})
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.audit.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
