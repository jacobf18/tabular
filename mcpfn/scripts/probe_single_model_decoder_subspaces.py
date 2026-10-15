"""CPU-only protected decoder displacement counterfactual on audited V8 gradients."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_single_model_point_alignment as alignment
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_parameter_subspace as subspace
from single_model_point_proposal_grid import direction_products, displacement_decision
from streamed_large_matrix_training import atomic_torch_save


SOURCE = alignment.ROOT / "point_alignment_v1"
OUTPUT = alignment.ROOT / "decoder_subspace_probe_v1"
INVENTORY = alignment.ROOT / "domain_gradient_probe_v1/parameter_inventory.json"
SCOPES = ("decoder", "output_layer")
MARGIN = .1
POLICY = ("Two prospectively fixed coordinate subsets of the same audited V8 training-gradient state: entire "
          "decoder and final output layer. Restrict the original full-model Adam displacement, then project in "
          "that coordinate subspace against every original fourteen composite and twenty-eight point gradient "
          "with margin0.1 and the unchanged2x actual-norm bound. All other model coordinates remain exactly "
          "unchanged. This is not a decoder-only optimizer replay, a neural finite-step test, a trained checkpoint "
          "selection, or a real benchmark. No scope/margin search or change to ongoing V13 training.")


def read(path):
    return json.loads(Path(path).read_text())


def status(output, state, **values):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **values})


def verify(output):
    config = read(output / "config.json")
    if (sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["scopes"] != list(SCOPES) or config["margin"] != MARGIN):
        raise ValueError("Never alter the registered coordinate counterfactual")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    alignment.verify_registration(SOURCE)
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use the fresh declared diagnostic directory")
    alignment.verify_registration(SOURCE)
    audit, evidence = read(SOURCE / "audit.json"), read(SOURCE / "v8/result.json")
    if not audit["complete"] or not audit["saved_full_gradient_geometry_recomputed"] or "v8" not in audit["states"]:
        raise ValueError("Require completed independent full-gradient evidence")
    inputs = {str(SOURCE / name): sha256(SOURCE / name) for name in ("config.json", "registration.json", "audit.json", "v8/result.json")}
    inputs.update({task["artifact"]: task["sha256"] for task in evidence["tasks"]})
    inputs[evidence["proposal"]] = evidence["proposal_sha256"]
    inputs[str(INVENTORY)] = sha256(INVENTORY)
    for path, digest in inputs.items():
        if path in audit["input_sha256"] and audit["input_sha256"][path] != digest:
            raise ValueError("Source evidence no longer matches its independent audit")
    verify_hashes(inputs)
    sources = {Path(__file__).resolve(), Path(subspace.__file__).resolve(), Path(alignment.projection.__file__).resolve(),
               Path(alignment.preflight.__file__).resolve(), REPO / "scripts/single_model_point_proposal_grid.py",
               REPO / "tests/test_single_model_parameter_subspace.py"}
    output.mkdir(parents=True)
    write_json(output / "config.json", {"policy": POLICY, "scopes": list(SCOPES), "margin": MARGIN,
               "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources}})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "projection_performed": False, "neural_evaluation_performed": False})
    status(output, "registered")


def load_scope(scope):
    evidence = read(SOURCE / "v8/result.json")
    ranges, total = subspace.inventory_ranges(read(INVENTORY), scope)
    elements = sum(row["stop"] - row["start"] for row in ranges)
    gradients = torch.empty((42, elements), dtype=torch.float32)
    seen = set()
    for task in evidence["tasks"]:
        saved = torch.load(task["artifact"], map_location="cpu", weights_only=False, mmap=True)
        for vector, entry in zip(saved["gradients"], saved["entries"]):
            index = entry["index"]
            if entry != evidence["entries"][index] or index in seen:
                raise ValueError("Preserve every original gradient identity")
            seen.add(index)
            gradients[index].copy_(subspace.extract(vector, ranges, total))
    if seen != set(range(42)) or len(evidence["entries"]) != 42:
        raise ValueError("Never drop a task or point gradient")
    snapshot = torch.load(evidence["proposal"], map_location="cpu", weights_only=False, mmap=True)
    before = subspace.extract(snapshot["before"], ranges, total)
    tentative = subspace.extract(snapshot["tentative"], ranges, total)
    return ranges, total, gradients, before, tentative


def geometry(gradients, before, tentative):
    gram, products, norm = alignment.projection.gram_and_products(gradients, before.double() - tentative.double())
    record = {"gram": gram.tolist(), "restricted_adam_products": products.tolist(), "restricted_adam_norm": norm,
              "adverse_before": int(np.sum(products < 0)), "nonzero_gradients": int(np.sum(np.diag(gram) > 0))}
    if norm == 0:
        return {**record, "feasible_projection": False, "error": "zero_restricted_adam_proposal"}, None
    try:
        solution = alignment.projection.solve_projection(gram, products, norm, margin_ratio=MARGIN)
        alignment.preflight.verify_certificate(gram, products, norm, solution, MARGIN)
    except (ValueError, AssertionError) as error:
        return {**record, "feasible_projection": False, "error": repr(error)}, None
    candidate = alignment.projection.projected_parameters(tentative, gradients, solution["raw_gradient_coefficients"])
    actual, actual_norm = direction_products(gradients, before.double() - candidate.double())
    decision = displacement_decision(actual, actual_norm, norm)
    return {**record, "feasible_projection": True, "projection": solution, "actual_products": actual.tolist(),
            "actual_norm": actual_norm, "decision": decision}, candidate


def run(output):
    verify(output)
    if read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started diagnostic")
    results = {}
    for scope in SCOPES:
        status(output, "projecting", scope=scope)
        ranges, total, gradients, before, tentative = load_scope(scope)
        record, candidate = geometry(gradients, before, tentative)
        path = output / f"{scope}.pt"
        atomic_torch_save({"before": before, "tentative": tentative, "candidate": candidate, "ranges": ranges}, path)
        results[scope] = {**record, "parameters": before.numel(), "total_parameters": total, "artifact": str(path), "sha256": sha256(path)}
        del gradients, before, tentative, candidate
    verify(output)
    write_json(output / "result.json", {"scopes": results, "neural_evaluation_performed": False, "optimizer_replayed": False,
               "candidate_selection_performed": False, "goal_success_not_assessed": True})
    status(output, "awaiting_independent_audit")


def audit(output):
    verify(output)
    if read(output / "status.json")["state"] != "awaiting_independent_audit":
        raise ValueError("Require a fresh complete unaudited diagnostic")
    result = read(output / "result.json")
    hashes = {str(output / "result.json"): sha256(output / "result.json")}
    for scope in SCOPES:
        record = result["scopes"][scope]
        ranges, total, gradients, before, tentative = load_scope(scope)
        hashes[record["artifact"]] = record["sha256"]
        saved = torch.load(record["artifact"], map_location="cpu", weights_only=True)
        if (saved["ranges"] != ranges or record["parameters"] != before.numel() or record["total_parameters"] != total
                or not torch.equal(saved["before"], before) or not torch.equal(saved["tentative"], tentative)):
            raise ValueError("Restricted parameters must exactly bind the original full-vector state")
        expected, candidate = geometry(gradients, before, tentative)
        for key, value in expected.items():
            alignment.compare_metric_records(record[key], value)
        if (candidate is None) != (saved["candidate"] is None) or candidate is not None and not torch.equal(candidate, saved["candidate"]):
            raise ValueError("The exact CPU rounded subset projection does not reproduce")
        del gradients, before, tentative, candidate
    verify_hashes(hashes)
    verify(output)
    write_json(output / "audit.json", {"complete": True, "source_gradient_vectors": 42, "scopes": list(SCOPES),
               "artifact_sha256": hashes, "cuda_initialized": False, "optimizer_replayed": False,
               "finite_step_accuracy_assessed": False, "goal_success_not_assessed": True})
    lines = ["# Decoder-coordinate protection feasibility", "", POLICY, "",
             "| Scope | Parameters | Adverse before /42 | Feasible projection | Actual norm ratio | Admissible |",
             "|---|---:|---:|---|---:|---|"]
    for scope, record in result["scopes"].items():
        decision = record.get("decision", {})
        lines.append(f'| {scope} | {record["parameters"]} | {record["adverse_before"]} | {record["feasible_projection"]} | '
                     f'{decision.get("norm_ratio", "undefined")} | {decision.get("admissible", False)} |')
    (output / "RESULTS.md").write_text("\n".join([*lines, "", "A positive directional certificate does not guarantee finite-step or held-out improvement.", ""]))
    status(output, "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Use CPU only while the independent V13 trainer occupies the GPU")
    output = arguments.output.resolve()
    if arguments.stage == "register":
        register(output)
    else:
        with (output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage](output)
            except BaseException as error:
                status(output, "failed", error=repr(error))
                raise
