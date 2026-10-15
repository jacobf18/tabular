"""Measure fresh small-context prior coverage without models or changed evaluation rules."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_csv, write_json
from single_model_area_prior import AreaMixedSampler
from train_single_matrixformer import storage_guard
from train_single_matrixformer_continuation import read, verify_run


SEED_START = 49100501
REPLICAS = 16


def column_statistics(source, target, mask):
    if (source.ndim != 2 or source.shape != target.shape or mask.shape != target.shape
            or mask.dtype != np.bool_ or not mask.any() or not np.isfinite(target).all()
            or not np.isnan(source[mask]).all()):
        raise ValueError("Require finite complete truth and an aligned nonempty hidden-query mask")
    result = []
    for column in range(target.shape[1]):
        queries = mask[:, column]
        if not queries.any():
            continue
        observed = np.isfinite(source[:, column])
        query_std = float(np.std(target[queries, column]))
        count = int(queries.sum())
        result.append({"column": column, "queries": count, "observations": int(observed.sum()),
                       "query_std": query_std, "full_target_std": float(np.std(target[:, column])),
                       "observed_input_std": float(np.std(source[observed, column])) if observed.any() else None,
                       "uci_only_nonconstant": 1e-4 <= query_std < 1e-3,
                       "common_nonconstant": query_std >= 1e-3,
                       "query_band": "1" if count == 1 else "2" if count == 2 else "3-4" if count <= 4
                       else "5-15" if count <= 15 else "16-63" if count <= 63 else "64+"})
    return result


def summarize(rows, cases):
    if not rows or not cases or len({case["index"] for case in cases}) != len(cases):
        raise ValueError("Require complete uniquely indexed synthetic cases")
    rare = [row for row in rows if row["uci_only_nonconstant"]]
    return {"cases": len(cases), "queried_columns": len(rows), "uci_only_columns": len(rare),
            "cases_with_uci_only_columns": len({row["index"] for row in rare}),
            "uci_only_query_counts": dict(Counter(row["queries"] for row in rare)),
            "uci_only_full_target_std_at_least_point_one": sum(row["full_target_std"] >= .1 for row in rare),
            "query_bands": dict(Counter(row["query_band"] for row in rows)),
            "case_query_fraction_min": min(case["query_fraction"] for case in cases),
            "case_query_fraction_max": max(case["query_fraction"] for case in cases)}


def register(run, evidence, output):
    training = read(run / "config.json")
    verify_run(training, run)
    diagnosis = read(evidence / "saved_prediction_audit.json")
    if (not diagnosis["complete"] or diagnosis["raw_prediction_scores_reproduced"] != 720
            or diagnosis["gate_changed"] or diagnosis["worst_uci_column"]["queries"] != 2
            or read(evidence / "config.json")["run"] != str(run)):
        raise ValueError("Require the complete same-run saved-prediction diagnosis")
    specifications = [{"seed": SEED_START + 1000 * component + replica, "component": component,
                       "replica": replica, "transposed": transposed}
                      for component in range(8) for replica in range(REPLICAS) for transposed in (False, True)]
    prior_seeds = {case["seed"] for case in training["validation_cases"] + training["confirmation_cases"]}
    prior_seeds.update(case["seed"] for line in (run / "training.jsonl").read_text().splitlines()
                       for case in json.loads(line)["microsteps"])
    if prior_seeds.intersection(case["seed"] for case in specifications):
        raise ValueError("Fresh diagnostic draws may not reuse training or evaluation seeds")
    sources = {Path(__file__).resolve(), *(REPO / "src/tabimpute").rglob("*.py")}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    inputs = [run / "config.json", run / "training.jsonl", evidence / "config.json",
              evidence / "summary.json", evidence / "saved_prediction_audit.json"]
    config = {"run": str(run), "cases": specifications, "requested_shape": [240, 80], "seed_start": SEED_START,
              "replicas_per_component": REPLICAS, "regime": "legacy", "device": "cpu",
              "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "input_sha256": {str(path): sha256(path) for path in inputs},
              "question": "Do fresh small-context priors cover tiny held-out sample variance with ordinary full-column signal?",
              "design": "Eight original prior components, sixteen independent seeds each, both orientations per seed. Preserve original noise, masks, normalization, latent intrinsic shapes and clean targets; explicit legacy amplitude.",
              "training_performed": False, "model_inference_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False,
              "limitations": "Balanced diagnostic prior components, not the training mixture's population frequencies. Paired orientations share base draws but support augmentation may differ. No new training recipe or admission gate."}
    output.mkdir(parents=True, exist_ok=False)
    storage_guard(output)
    write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})
    print(json.dumps({"registered": str(output), "cases": len(specifications)}), flush=True)


def verify(config, output):
    if sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]:
        raise ValueError("Registered coverage probe changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])


def run_probe(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("Run the coverage probe with CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(4)
    config = read(output / "config.json")
    verify(config, output)
    sampler = AreaMixedSampler(read(Path(config["run"]) / "config.json"), device="cpu")
    rows, cases, bank = [], [], []
    for index, specification in enumerate(config["cases"]):
        episode = sampler.sample("imputation", 1, seed=specification["seed"], shape=config["requested_shape"],
                                 component=specification["component"], transpose=specification["transposed"], regime="legacy")
        assert episode.metadata["gradient_data"] == "synthetic" and not episode.metadata["area_prior"]["applied"]
        source, target, mask = [values[0].numpy() for values in (episode.source, episode.target, episode.mask)]
        columns = column_statistics(source, target, mask)
        case = {"index": index, **specification, **episode.metadata,
                "query_fraction": float(mask.mean()), "uci_only_columns": sum(row["uci_only_nonconstant"] for row in columns)}
        rows.extend({"index": index, "seed": specification["seed"], "component": specification["component"],
                     "transposed": specification["transposed"], **row} for row in columns)
        cases.append(case)
        bank.append({"source": episode.source, "target": episode.target, "mask": episode.mask, "metadata": case})
        if (index + 1) % 16 == 0:
            progress = {"state": "probing", "completed": index + 1, "expected": len(config["cases"]),
                        "pid": os.getpid(), "updated_unix": time.time()}
            write_json(output / "status.json", progress)
            print(json.dumps(progress), flush=True)
    assert len(cases) == 256 and not torch.cuda.is_initialized()
    write_csv(output / "columns.csv", rows)
    write_json(output / "cases.json", cases)
    bank_path = output / "synthetic_bank.pt"
    temporary = output / "synthetic_bank.pt.tmp"
    torch.save(bank, temporary)
    temporary.replace(bank_path)
    verify(config, output)
    summary = {"complete": True, **summarize(rows, cases), "config_sha256": sha256(output / "config.json"),
               "bank_sha256": sha256(bank_path), "columns_sha256": sha256(output / "columns.csv"),
               "cuda_initialized": False, "training_performed": False, "model_inference_performed": False,
               "checkpoint_selection_performed": False, "gate_changed": False,
               "real_benchmark_success": False, "limitations": config["limitations"]}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        if arguments.run is None or arguments.evidence is None:
            parser.error("Registration requires --run and --evidence")
        register(arguments.run.resolve(), arguments.evidence.resolve(), output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never overwrite a started or completed diagnostic")
        try:
            run_probe(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                               "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
