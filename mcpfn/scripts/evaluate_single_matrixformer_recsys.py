"""One-checkpoint ratings evaluation with a matched-preprocessing parent control."""

import argparse
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from compare_recsys_checkpoint import audit_scores, reference_paths
from dwip_recsys_experiment import inference_components, inference_row_scales
from evaluate_matrixformer_plus_large_recsys import collect_predictions
from evaluate_single_matrixformer_tabular import GOAL, REPO, ROOT, check_reserve, make_model, read_json, verify_hashes
from evaluate_two_expert_router import CHECKPOINT as SPECIALIST, GENERAL
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from tabimpute.streamed_inference import overlapping_spans
from train_streamed_large_matrices import wait_for_gpu


SMALL = ROOT / "two_expert_router_v2_20261004"
LARGE = ROOT / "matrixformer_plus_large_recsys_20261004"
DATASETS = ("100K", "netflix", "1M", "10M")
ROLES = ("student", "parent_same_recipe")


def load_split(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def observed_matrix(split):
    shape = tuple(split["shape"])
    observed = np.full(shape, np.nan, dtype=np.float64)
    for role in ("train", "test"):
        users, items, ratings = (split[f"{role}_{name}"] for name in ("users", "items", "ratings"))
        if users.ndim != 1 or items.shape != users.shape or ratings.shape != users.shape or not len(users):
            raise ValueError("Invalid rating split vectors")
        if not np.isfinite(ratings).all() or (users < 0).any() or (users >= shape[0]).any() or (items < 0).any() or (items >= shape[1]).any():
            raise ValueError("Invalid rating values or indices")
    observed[split["train_users"], split["train_items"]] = split["train_ratings"]
    if np.count_nonzero(np.isfinite(observed)) != len(split["train_ratings"]):
        raise ValueError("Duplicate training cells")
    if np.isfinite(observed[split["test_users"], split["test_items"]]).any():
        raise ValueError("Held-out rating leakage")
    return observed


def normalization(observed):
    means, scales, offsets = inference_components(observed, "two_way_user_scale")
    row_scales = inference_row_scales(observed, "two_way_user_scale", means, scales, offsets)
    return {"means": means, "scales": scales, "offsets": offsets, "row_scales": row_scales}


def expected_coverage(shape, users, items):
    row_counts, column_counts = np.zeros(len(users), dtype=np.uint8), np.zeros(len(items), dtype=np.uint8)
    for start, stop in overlapping_spans(shape[0], 2000):
        row_counts += (users >= start) & (users < stop)
    for start, stop in overlapping_spans(shape[1], 1000):
        column_counts += (items >= start) & (items < stop)
    counts = row_counts * column_counts
    if (counts == 0).any():
        raise ValueError("Uncovered rating query")
    return counts


def predict_queries(model, observed, users, items, normalizer, rating_range, progress=None):
    model.max_num_rows, model.max_num_cols = 2000, 1000
    prediction, counts = collect_predictions(model, observed, users, items, **normalizer, progress=progress)
    np.testing.assert_array_equal(counts, expected_coverage(observed.shape, users, items))
    return np.clip(prediction, *rating_range), counts


def reference_case(dataset):
    paths = reference_paths(dataset, "global")
    if dataset in ("100K", "netflix"):
        folder = "movielens_100k" if dataset == "100K" else "netflix_small"
        prediction_path = SMALL / "recsys" / f"{folder}.npz"
        record_path = prediction_path.with_suffix(".json")
        record = read_json(record_path)
        manifest = read_json(SMALL / "manifest.json")
        split_hash = manifest["input_sha256"][str(paths["split"])]
        prediction_hash = record["prediction_sha256"]
    else:
        prediction_path = LARGE / dataset / "query_predictions.npz"
        record_path = LARGE / dataset / "audit.json"
        record = read_json(record_path)
        split_hash = read_json(LARGE / "manifest.json")["split_sha256"][dataset]
        prediction_hash = record["query_predictions_sha256"]
    hashes = {str(paths["split"]): split_hash, str(prediction_path): prediction_hash, str(record_path): sha256(record_path)}
    verify_hashes(hashes)
    return {"dataset": dataset, "label": paths["label"], "split": str(paths["split"]),
            "reference_prediction": str(prediction_path), "reference_sha256": hashes}


def prepare(output):
    import torch

    torch.set_num_threads(4)
    sources = dict(read_json(LARGE / "manifest.json")["source_sha256"])
    verify_hashes(sources)
    for filename in ("evaluate_single_matrixformer_recsys.py", "evaluate_single_matrixformer_tabular.py",
                     "compare_recsys_checkpoint.py", "evaluate_large_recsys.py", "evaluate_matrixformer_recsys_ratings.py",
                     "recsys_evaluation_common.py", "train_streamed_large_matrices.py"):
        source = REPO / "scripts" / filename
        sources[str(source)] = sha256(source)
    borders = REPO / "src/tabimpute/data/borders.pt"
    sources[str(borders)] = sha256(borders)
    inputs = dict(read_json(LARGE / "manifest.json")["checkpoint_sha256"])
    verify_hashes(inputs)
    inputs.update({str(path.parent / "config.json"): sha256(path.parent / "config.json") for path in (GENERAL, SPECIALIST)})
    paper_path = GOAL / "reference/recsys/ratings.csv"
    with paper_path.open() as stream:
        paper = {row["dataset"]: row for row in csv.DictReader(stream) if row["method"] == "MatrixFormer+"}
    cases, summaries = [], []
    for dataset in DATASETS:
        check_reserve(output)
        case = reference_case(dataset)
        split = load_split(case["split"])
        with np.load(case["reference_prediction"], allow_pickle=False) as saved:
            if dataset in ("100K", "netflix"):
                for stored, key in (("users", "test_users"), ("items", "test_items"), ("truth", "test_ratings")):
                    np.testing.assert_array_equal(saved[stored], split[key])
            blend = audit_scores(split, saved["observed_blend"])
            specialist = audit_scores(split, saved["specialist"])
            general = audit_scores(split, saved["general"])
        np.testing.assert_allclose([blend["RMSE"], blend["MAE"]],
                                   [float(paper[case["label"]]["RMSE"]), float(paper[case["label"]]["MAE"])], rtol=1e-12, atol=1e-12)
        observed = observed_matrix(split)
        normalizer = normalization(observed)
        normalization_path = output / "normalization" / f"{dataset}.npz"
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(normalization_path, **normalizer)
        if dataset in ("1M", "10M"):
            old_path = LARGE / dataset / "specialist_normalization.npz"
            if sha256(old_path) != read_json(LARGE / dataset / "specialist.json")["normalization_sha256"]:
                raise ValueError("Archived normalization changed")
            with np.load(old_path, allow_pickle=False) as previous:
                for name, values in normalizer.items():
                    np.testing.assert_allclose(values, previous[name], rtol=1e-12, atol=1e-12)
            case["reference_sha256"][str(old_path)] = sha256(old_path)
        case.update(shape=list(observed.shape), train_ratings=len(split["train_ratings"]), test_ratings=len(split["test_ratings"]),
                    rating_range=[float(split["train_ratings"].min()), float(split["train_ratings"].max())],
                    normalization=str(normalization_path.resolve()), normalization_sha256=sha256(normalization_path),
                    blend=blend, specialist=specialist, general_original_recipe=general)
        cases.append(case)
        summaries.append({"dataset": case["label"], **blend})
        print(json.dumps({"prepared": dataset, "blend": blend}), flush=True)
        del observed, normalizer, split
        gc.collect()
    references = {str(paper_path): sha256(paper_path), **inputs}
    references.update({str(root / "manifest.json"): sha256(root / "manifest.json") for root in (SMALL, LARGE)})
    protocol = {"version": 1, "cases": cases, "source_sha256": sources, "reference_sha256": references,
                "normalization": "Observed-only global two-way user/item biases, residual scale and shrunk user scales; fit before windowing",
                "context": "Original dense BF16 inference; 2000x1000 overlapping-tail windows; restore rating units before averaging; training-range clip afterward",
                "metrics": "Frozen seed123 splits; mean user RMSE and MAE, not cell pooled",
                "roles": {"student": "One synthetic-confirmed checkpoint, no ensemble", "parent_same_recipe": "Unmodified general parent with identical normalization and context"},
                "selection": "Report-only, synthetic-selected candidate; no benchmark labels in training or candidate selection"}
    write_json(output / "protocol.json", protocol)
    write_json(output / "protocol_hash.json", {"sha256": sha256(output / "protocol.json")})
    write_csv(output / "reference.csv", summaries)
    for source in sources:
        destination = output / "source_snapshot" / Path(source).relative_to(REPO.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "status.json", {"state": "prepared", "datasets": 4, "reference_scores_reproduced": True})


def load_protocol(path):
    if sha256(path / "protocol.json") != read_json(path / "protocol_hash.json")["sha256"]:
        raise ValueError("Ratings protocol changed")
    protocol = read_json(path / "protocol.json")
    verify_hashes(protocol["source_sha256"])
    verify_hashes(protocol["reference_sha256"])
    for case in protocol["cases"]:
        verify_hashes(case["reference_sha256"])
        verify_hashes({case["normalization"]: case["normalization_sha256"]})
    return protocol


def write_summary(output, protocol):
    rows = []
    for case in protocol["cases"]:
        for role in ROLES:
            path = output / case["dataset"] / f"{role}.json"
            if not path.exists():
                continue
            metrics = read_json(path)["metrics"]
            row = {"dataset": case["label"], "role": role, **metrics}
            for metric in ("RMSE", "MAE"):
                row[f"blend_{metric}"] = case["blend"][metric]
                row[f"delta_{metric}"] = metrics[metric] - case["blend"][metric]
                row[f"general_original_recipe_{metric}"] = case["general_original_recipe"][metric]
            rows.append(row)
    if rows:
        write_csv(output / "comparison.csv", rows)


def audit(output, protocol):
    manifest = read_json(output / "manifest.json")
    verify_hashes(manifest["candidate_sha256"])
    bound = Path(manifest["protocol"]) / "protocol.json"
    if sha256(bound) != manifest["protocol_sha256"] or read_json(bound) != protocol:
        raise ValueError("Evaluation protocol mismatch")
    completed = []
    for case in protocol["cases"]:
        parity = read_json(output / case["dataset"] / "specialist_parity.json")
        if not parity["passed"] or parity["checkpoint_sha256"] != sha256(SPECIALIST):
            raise ValueError("Specialist recipe parity required")
        split = load_split(case["split"])
        for role in ROLES:
            path = output / case["dataset"] / f"{role}.npz"
            record = read_json(path.with_suffix(".json"))
            if record["dataset"] != case["dataset"] or record["role"] != role:
                raise ValueError("Saved rating metadata changed")
            if record["manifest_sha256"] != sha256(output / "manifest.json") or sha256(path) != record["prediction_sha256"]:
                raise ValueError("Saved rating prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["counts"], expected_coverage(case["shape"], split["test_users"], split["test_items"]))
                if (saved["prediction"] < case["rating_range"][0]).any() or (saved["prediction"] > case["rating_range"][1]).any():
                    raise ValueError("Predictions violate frozen clipping range")
                metrics = audit_scores(split, saved["prediction"], record["metrics"])
            completed.append({"dataset": case["dataset"], "role": role, "queries": len(split["test_ratings"]), **metrics})
    if len(completed) != 8:
        raise ValueError("All four datasets and both separate model roles are required")
    write_summary(output, protocol)
    write_json(output / "audit.json", {"complete": True, "cases": completed, "single_checkpoint_per_prediction": True,
                                       "no_ensemble": True, "training_performed": False,
                                       "manifest_sha256": sha256(output / "manifest.json")})


def evaluate(output, protocol_path, checkpoint, confirmation):
    import torch

    protocol = load_protocol(protocol_path)
    gate = read_json(confirmation / "gate.json")
    if not gate["passed"] or gate["checkpoint_sha256"] != sha256(checkpoint) or gate["selected_step"] == 0:
        raise ValueError("Require the synthetic-confirmed frozen nonzero checkpoint")
    manifest = {"candidate_sha256": {str(checkpoint): sha256(checkpoint), str(checkpoint.parent / "config.json"): sha256(checkpoint.parent / "config.json")},
                "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path / "protocol.json"),
                "confirmation": str(confirmation), "confirmation_sha256": sha256(confirmation / "gate.json"),
                "selected_step": gate["selected_step"], "student_has_one_checkpoint": True}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Refusing changed resume manifest")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    wait_for_gpu(output, False, False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.9)
    for case in protocol["cases"]:
        check_reserve(output)
        destination = output / case["dataset"]
        destination.mkdir(exist_ok=True)
        split = load_split(case["split"])
        observed = observed_matrix(split)
        with np.load(case["normalization"], allow_pickle=False) as saved:
            normalizer = dict(saved)
        parity_path = destination / "specialist_parity.json"
        if not parity_path.exists():
            indices = np.arange(len(split["test_ratings"])) if case["dataset"] in ("100K", "netflix") else np.unique(np.concatenate([
                np.linspace(0, len(split["test_ratings"]) - 1, 12, dtype=np.int64),
                [np.argmax(split["test_users"]), np.argmax(split["test_items"])]]))
            specialist = make_model(SPECIALIST)
            estimate, _ = predict_queries(specialist, observed, split["test_users"][indices], split["test_items"][indices],
                                          normalizer, case["rating_range"])
            with np.load(case["reference_prediction"], allow_pickle=False) as previous:
                np.testing.assert_allclose(estimate, previous["specialist"][indices], rtol=0, atol=1e-7)
                difference = float(np.max(np.abs(estimate - previous["specialist"][indices])))
            write_json(parity_path, {"passed": True, "queries": len(indices), "all_queries": len(indices) == len(split["test_ratings"]),
                                     "sampling": "All small-suite queries; deterministic spread plus maximum user/item on large suites",
                                     "max_abs_difference": difference, "checkpoint_sha256": sha256(SPECIALIST)})
            del specialist
            gc.collect()
            torch.cuda.empty_cache()
        for role, model_path in (("student", checkpoint), ("parent_same_recipe", GENERAL)):
            path = destination / f"{role}.npz"
            if path.with_suffix(".json").exists():
                record = read_json(path.with_suffix(".json"))
                if record["manifest_sha256"] != sha256(manifest_path) or sha256(path) != record["prediction_sha256"]:
                    raise ValueError("Cached rating result changed")
                continue
            model = make_model(model_path)
            model.model.eval().requires_grad_(False)

            def progress(row, rows, column, columns):
                write_json(output / "status.json", {"state": "evaluating", "dataset": case["dataset"], "role": role,
                                                    "row_block": row, "row_blocks": rows, "column_block": column,
                                                    "column_blocks": columns, "pid": os.getpid(), "updated_unix": time.time()})

            started = time.monotonic()
            prediction, counts = predict_queries(model, observed, split["test_users"], split["test_items"], normalizer, case["rating_range"], progress)
            metrics = audit_scores(split, prediction)
            atomic_npz(path, prediction=prediction, counts=counts)
            write_json(path.with_suffix(".json"), {"dataset": case["dataset"], "role": role, "metrics": metrics,
                                                  "manifest_sha256": sha256(manifest_path), "prediction_sha256": sha256(path),
                                                  "checkpoint_sha256": sha256(model_path), "seconds": time.monotonic() - started})
            write_summary(output, protocol)
            print(json.dumps({"dataset": case["dataset"], "role": role, "metrics": metrics}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del observed, normalizer, split
        gc.collect()
    audit(output, protocol)
    write_json(output / "status.json", {"state": "complete", "datasets": 4, "separate_roles": list(ROLES)})


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
        try:
            if arguments.stage == "prepare":
                prepare(output)
            elif arguments.stage == "audit":
                audit(output, load_protocol(arguments.protocol.resolve()))
            else:
                evaluate(output, arguments.protocol.resolve(), arguments.checkpoint.resolve(), arguments.confirmation.resolve())
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
