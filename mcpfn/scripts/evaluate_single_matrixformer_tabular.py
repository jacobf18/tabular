"""Frozen single-checkpoint MissBench/UCI evaluation against the observed blend."""

import argparse
import ast
from collections import Counter
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from audit_dwip_missbench_uci import load_original
from compare_imputation_checkpoint import aggregate
from evaluate_dwip_missbench_uci import score_predictions
from evaluate_two_expert_router import GENERAL, REPO, ROOT, TabularExpert, make_model
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from train_streamed_large_matrices import wait_for_gpu


ROUTER = ROOT / "two_expert_router_v2_20261004"
GOAL = REPO / "runs/single_model_goal_20261005"
SCORERS = {"missbench": REPO / "benchmark/plot_options.py", "uci": REPO / "benchmark/uci/plot_uci_negative_rmse.py"}


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(hashes):
    for filename, digest in hashes.items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Frozen artifact changed: {filename}")


def read_blend_rows(path):
    with Path(path).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == "matrixformer_plus"]
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate blended reference cases")
    return {row["case_id"]: row for row in rows}


def non_cli_ast(path):
    module = ast.parse(Path(path).read_text())
    module.body = [node for node in module.body if not isinstance(node, ast.FunctionDef) or node.name != "main"]
    return ast.dump(module, include_attributes=False)


def prediction_grid(missing, path, key):
    with np.load(path, allow_pickle=False) as stored:
        values = stored[key]
        if tuple(stored["shape"]) != missing.shape or values.shape != (int(np.isnan(missing).sum()),):
            raise ValueError("Saved prediction shape/query coverage mismatch")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite saved predictions")
        result = missing.astype(values.dtype)
        result[np.isnan(missing)] = values
    return result


def checked_score(truth, prediction, missing, suite, original):
    result = score_predictions(truth, prediction, missing, suite)
    score = result["nrmse_original"]
    independent = float(original(truth, prediction, np.isnan(missing)))
    if not np.isfinite(score):
        raise ValueError("Nonfinite tabular NRMSE")
    np.testing.assert_allclose(score, independent, rtol=1e-5, atol=1e-5)
    return score, {key: result[key] for key in ("missing_cells", "nrmse_scored_columns", "nrmse_skipped_constant_columns")}


def check_reserve(output):
    if shutil.disk_usage(output).free < 15 * 1024**3:
        raise OSError("Preserve the registered 15 GiB disk reserve")


def prepare(output):
    archived = read_json(ROUTER / "manifest.json")
    checkpoint_inputs = {filename: digest for filename, digest in archived["input_sha256"].items()
                         if filename.endswith(".pth") or filename.endswith("/config.json")}
    verify_hashes(checkpoint_inputs)
    current_sources = {}
    compatibility = []
    for filename, digest in archived["source_sha256"].items():
        source = Path(filename)
        actual = sha256(source)
        if actual != digest:
            snapshot = ROUTER / "source_snapshot" / source.relative_to(REPO.parent)
            if source.name != "evaluate_two_expert_router.py" or sha256(snapshot) != digest or non_cli_ast(source) != non_cli_ast(snapshot):
                raise ValueError(f"Archived inference/scoring dependency changed: {source}")
            compatibility.append({"path": filename, "difference": "CLI main only; all other AST nodes identical"})
        current_sources[filename] = actual
    current_sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    for filename in ("audit_dwip_missbench_uci.py", "compare_imputation_checkpoint.py", "train_streamed_large_matrices.py"):
        source = REPO / "scripts" / filename
        current_sources[str(source)] = sha256(source)
    csv_path = GOAL / "reference/imputation_nrmse_per_case.csv"
    frozen_paper = read_blend_rows(csv_path)
    if set(frozen_paper) != {case["case_id"] for case in archived["cases"]}:
        raise ValueError("Frozen paper and router case inventories differ")
    originals = {suite: load_original(path, "compute_normalized_rmse_columnwise") for suite, path in SCORERS.items()}
    cases = []
    for specification in archived["cases"]:
        case_id = specification["case_id"]
        source = Path(specification["source_directory"])
        record_path = ROUTER / "tabular/cases" / (case_id + ".json")
        prediction_path = ROUTER / "tabular/predictions" / (case_id + ".npz")
        previous = read_json(record_path)
        if any(previous[key] != value for key, value in specification.items()):
            raise ValueError("Reference case metadata mismatch")
        hashes = {str(source / "missing.npy"): previous["missing_sha256"], str(source / "true.npy"): previous["true_sha256"],
                  str(prediction_path): previous["prediction_sha256"]}
        verify_hashes(hashes)
        missing, truth = (np.load(source / name, allow_pickle=False) for name in ("missing.npy", "true.npy"))
        prediction = prediction_grid(missing, prediction_path, "observed_blend")
        score, diagnostics = checked_score(truth, prediction, missing, specification["suite"], originals[specification["suite"]])
        np.testing.assert_allclose(score, previous["nrmse"]["observed_blend"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(score, float(frozen_paper[case_id]["nrmse"]), rtol=1e-6, atol=1e-6)
        if frozen_paper[case_id]["status"] != "ok":
            raise ValueError("Incomplete paper baseline")
        cases.append({**specification, "shape": list(missing.shape), "nrmse_original": score,
                      "general_nrmse": previous["nrmse"]["general"], "diagnostics": diagnostics,
                      "reference_predictions": str(prediction_path),
                      "input_sha256": {**hashes, str(record_path): sha256(record_path)}})
    if Counter(case["suite"] for case in cases) != {"missbench": 462, "uci": 1800}:
        raise ValueError("Incomplete frozen benchmark inventory")
    complete = {case["case_id"]: {"new_nrmse": case["nrmse_original"]} for case in cases}
    summaries = aggregate(cases, complete, ("suite",))
    paper_overall = GOAL / "reference/imputation_nrmse_overall.csv"
    with paper_overall.open() as stream:
        paper = {row["suite"]: row for row in csv.DictReader(stream) if row["method"] == "matrixformer_plus"}
    for row in summaries:
        np.testing.assert_allclose(row["new_nrmse"], float(paper[row["suite"]]["nrmse"]), rtol=1e-7, atol=1e-7)
    protocol = {"version": 1, "cases": cases, "source_sha256": current_sources,
                "reference_sha256": {str(csv_path): sha256(csv_path), str(paper_overall): sha256(paper_overall),
                                     str(ROUTER / "manifest.json"): sha256(ROUTER / "manifest.json"), **checkpoint_inputs},
                "historical_source_compatibility": compatibility, "baseline_summaries": summaries,
                "inference": "Single unchanged network; FP32 weights/BF16 autocast, predictive mean; cuDNN SDPA off",
                "context": "MissBench full matrix; UCI observed-column scaling globally before 2000x1000 overlapping-tail windows",
                "metric": "Original masked-target column NRMSE and constant-column rules; cases within datasets, equal dataset weights",
                "selection": "Synthetic-only selected checkpoint; these previously studied real benchmarks are report-only"}
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol_hash.json", {"sha256": sha256(output / "protocol.json")})
    for source in current_sources:
        destination = output / "source_snapshot" / Path(source).relative_to(REPO.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_csv(output / "reference_overall.csv", summaries)
    write_csv(output / "reference_per_pattern.csv", aggregate(cases, complete, ("suite", "pattern")))
    write_json(output / "status.json", {"state": "prepared", "cases": len(cases), "baseline_rescored": True})
    print(json.dumps({"prepared": len(cases), "reference": summaries}), flush=True)


def load_protocol(path):
    if sha256(path / "protocol.json") != read_json(path / "protocol_hash.json")["sha256"]:
        raise ValueError("Protocol hash mismatch")
    protocol = read_json(path / "protocol.json")
    verify_hashes(protocol["source_sha256"])
    verify_hashes(protocol["reference_sha256"])
    return protocol


def write_summaries(output, protocol, completed):
    for name, dimensions in (("overall", ("suite",)), ("per_pattern", ("suite", "pattern")),
                             ("per_dataset", ("suite", "dataset")), ("per_pattern_rate", ("suite", "pattern", "missingness"))):
        write_csv(output / f"{name}.csv", aggregate(protocol["cases"], completed, dimensions))
    if completed:
        write_csv(output / "per_case.csv", [{key: row[key] for key in
                  ("case_id", "suite", "dataset", "pattern", "missingness", "repeat", "new_nrmse", "blend_nrmse", "delta_nrmse")}
                  for row in completed.values()])


def load_case(case):
    verify_hashes(case["input_sha256"])
    return tuple(np.load(Path(case["source_directory"]) / name, allow_pickle=False) for name in ("missing.npy", "true.npy"))


def audit(output, protocol):
    manifest = read_json(output / "manifest.json")
    bound_protocol = Path(manifest["protocol"]) / "protocol.json"
    if sha256(bound_protocol) != manifest["protocol_sha256"] or read_json(bound_protocol) != protocol:
        raise ValueError("Audit protocol differs from the registered candidate evaluation")
    verify_hashes(manifest["candidate_sha256"])
    parity = read_json(output / "parent_parity.json")
    if not parity["passed"] or not parity["cases"] or parity["parent_sha256"] != sha256(GENERAL):
        raise ValueError("Require a valid parent parity check")
    originals = {suite: load_original(path, "compute_normalized_rmse_columnwise") for suite, path in SCORERS.items()}
    completed = {}
    for case in protocol["cases"]:
        record = read_json(output / "cases" / (case["case_id"] + ".json"))
        if any(record[key] != case[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")):
            raise ValueError("Saved result metadata does not match the frozen case")
        if record["manifest_sha256"] != sha256(output / "manifest.json"):
            raise ValueError("Saved result belongs to another evaluation")
        path = output / "predictions" / (case["case_id"] + ".npz")
        if sha256(path) != record["prediction_sha256"]:
            raise ValueError("Saved prediction changed")
        missing, truth = load_case(case)
        prediction = prediction_grid(missing, path, "prediction")
        score, diagnostics = checked_score(truth, prediction, missing, case["suite"], originals[case["suite"]])
        np.testing.assert_allclose(score, record["new_nrmse"], rtol=1e-7, atol=1e-7)
        if diagnostics != record["diagnostics"]:
            raise ValueError("Constant-column/query coverage changed")
        completed[case["case_id"]] = record
    write_summaries(output, protocol, completed)
    write_json(output / "audit.json", {"complete": True, "cases": len(completed), "single_checkpoint": True,
                                       "manifest_sha256": sha256(output / "manifest.json"), "training_performed": False,
                                       "parity_sha256": sha256(output / "parent_parity.json")})


def evaluate(output, protocol_path, checkpoint, confirmation):
    import torch

    protocol = load_protocol(protocol_path)
    gate = read_json(confirmation / "gate.json")
    if not gate["passed"] or gate["checkpoint_sha256"] != sha256(checkpoint) or gate["selected_step"] == 0:
        raise ValueError("Require the frozen nonzero checkpoint passing independent synthetic confirmation")
    manifest = {"candidate_sha256": {str(checkpoint): sha256(checkpoint), str(checkpoint.parent / "config.json"): sha256(checkpoint.parent / "config.json")},
                "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path / "protocol.json"),
                "confirmation": str(confirmation), "confirmation_sha256": sha256(confirmation / "gate.json"),
                "selected_step": gate["selected_step"], "single_checkpoint": True}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Refusing to resume a changed candidate, gate or protocol")
    else:
        write_json(manifest_path, manifest)
    wait_for_gpu(output, False, False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.9)
    parity_path = output / "parent_parity.json"
    if not parity_path.exists():
        parent = TabularExpert(make_model(GENERAL))
        selected = {}
        for case in protocol["cases"]:
            selected.setdefault((case["suite"], case["pattern"]), case)
        for suite in ("missbench", "uci"):
            selected[suite, "largest"] = max((case for case in protocol["cases"] if case["suite"] == suite), key=lambda case: np.prod(case["shape"]))
        parity = []
        for case in {case["case_id"]: case for case in selected.values()}.values():
            missing, _ = load_case(case)
            parent.suite = case["suite"]
            prediction = parent.impute(missing.copy(order="K"))
            previous = prediction_grid(missing, Path(case["reference_predictions"]), "general")
            mask = np.isnan(missing)
            np.testing.assert_allclose(prediction[mask], previous[mask], rtol=1e-6, atol=1e-6)
            parity.append({"case_id": case["case_id"], "max_abs_difference": float(np.max(np.abs(prediction[mask] - previous[mask])))})
        write_json(parity_path, {"passed": True, "cases": parity, "parent_sha256": sha256(GENERAL)})
        del parent
        gc.collect()
        torch.cuda.empty_cache()
    model = TabularExpert(make_model(checkpoint))
    originals = {suite: load_original(path, "compute_normalized_rmse_columnwise") for suite, path in SCORERS.items()}
    completed = {}
    fingerprint = sha256(manifest_path)
    for case in protocol["cases"]:
        check_reserve(output)
        missing, truth = load_case(case)
        destination = output / "predictions" / (case["case_id"] + ".npz")
        record_path = output / "cases" / (case["case_id"] + ".json")
        if record_path.exists():
            record = read_json(record_path)
            if record["manifest_sha256"] != fingerprint or sha256(destination) != record["prediction_sha256"]:
                raise ValueError("Cached result changed")
        else:
            model.suite = case["suite"]
            started = time.monotonic()
            prediction = model.impute(missing.copy(order="K"))
            score, diagnostics = checked_score(truth, prediction, missing, case["suite"], originals[case["suite"]])
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_npz(destination, shape=np.array(missing.shape), prediction=prediction[np.isnan(missing)])
            record = {key: case[key] for key in ("case_id", "suite", "dataset", "pattern", "missingness", "repeat")}
            record.update(new_nrmse=score, blend_nrmse=case["nrmse_original"], delta_nrmse=score - case["nrmse_original"],
                          diagnostics=diagnostics, manifest_sha256=fingerprint, prediction_sha256=sha256(destination),
                          seconds=time.monotonic() - started)
            write_json(record_path, record)
        completed[case["case_id"]] = record
        write_json(output / "status.json", {"state": "evaluating", "completed": len(completed), "expected": len(protocol["cases"]),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        if len(completed) % 25 == 0:
            write_summaries(output, protocol, completed)
            print(f"Single checkpoint: {len(completed)}/{len(protocol['cases'])}", flush=True)
    audit(output, protocol)
    write_json(output / "status.json", {"state": "complete", "completed": len(completed), "expected": len(protocol["cases"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--confirmation", type=Path)
    arguments = parser.parse_args()
    if arguments.stage != "prepare" and arguments.protocol is None:
        parser.error("Evaluation/audit requires --protocol")
    if arguments.stage == "evaluate" and (arguments.checkpoint is None or arguments.confirmation is None):
        parser.error("Evaluation requires --checkpoint and --confirmation")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=arguments.stage != "prepare")
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if arguments.stage == "prepare":
            prepare(output)
        elif arguments.stage == "audit":
            audit(output, load_protocol(arguments.protocol.resolve()))
        else:
            evaluate(output, arguments.protocol.resolve(), arguments.checkpoint.resolve(), arguments.confirmation.resolve())


if __name__ == "__main__":
    main()
