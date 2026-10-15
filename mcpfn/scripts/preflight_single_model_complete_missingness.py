"""Register and audit feasibility of all structured-prior families without model inference."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from audit_single_model_missingness_coverage import audit_normalization
import diagnose_single_model_missingness_coverage as coverage
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from single_model_complete_missingness_prior import CompleteMissingnessSampler, SPECIFICATION, preflight_cases


def register(output):
    if output.exists():
        raise FileExistsError("Never replace a prior preflight")
    cases = preflight_cases()
    sampler = CompleteMissingnessSampler()
    configurations = {f"{case['component']}/{case['shape']}": sampler.configuration(case["component"], case["shape"])
                      for case in cases}
    config = {"specification": SPECIFICATION, "cases": cases, "expected_cases": 192,
              "registered_unix": time.time(), "generator_configurations": configurations,
              "source_sha256": coverage.source_hashes(), "model_inference_performed": False,
              "training_performed": False, "checkpoint_selection_performed": False,
              "purpose": "CPU feasibility and clean-target coverage, not accuracy, promotion, or a selected training recipe"}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "registered", "config_sha256": sha256(output / "config.json")})


def run(output):
    config = json.loads((output / "config.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if status["state"] != "registered" or status["config_sha256"] != sha256(output / "config.json"):
        raise ValueError("Require untouched registration; never replace attempts")
    if config["cases"] != preflight_cases() or config["specification"] != SPECIFICATION:
        raise ValueError("Registered prior coverage changed")
    coverage.verify_hashes(config["source_sha256"])
    sampler = CompleteMissingnessSampler()
    records, files = [], {}
    directory = output / "arrays"
    directory.mkdir()
    write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "updated_unix": time.time()})
    try:
        for index, specification in enumerate(config["cases"]):
            expected = config["generator_configurations"][f"{specification['component']}/{specification['shape']}"]
            if sampler.configuration(specification["component"], specification["shape"]) != expected:
                raise ValueError("Registered generator configuration changed")
            started = time.monotonic()
            episode, arrays = sampler.draw(**specification)
            arrays = {name: array.numpy() for name, array in arrays.items()}
            destination = directory / f"case_{index:03d}.npz"
            atomic_npz(destination, **arrays)
            files[str(destination)] = sha256(destination)
            with np.load(destination, allow_pickle=False) as stored:
                audit_normalization(stored["raw_source"], stored["raw_target"], stored)
                if not np.array_equal(stored["query"], episode.mask[0].numpy()):
                    raise ValueError("Stored query coverage changed")
            record = {"index": index, **episode.metadata, "generation_and_io_seconds": time.monotonic() - started}
            records.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            write_json(output / "status.json", {"state": "running", "cases": len(records),
                                                "pid": os.getpid(), "updated_unix": time.time()})
        if len(records) != config["expected_cases"] or torch.cuda.is_initialized():
            raise ValueError("Incomplete CPU-only preflight")
        coverage.verify_hashes(config["source_sha256"])
        coverage.verify_hashes(files)
        write_json(output / "cases.json", records)
        write_csv(output / "cases.csv", records)
        family_counts = {family: sum(record["family"] == family for record in records) for family in coverage.FAMILIES}
        if set(family_counts.values()) != {16}:
            raise ValueError("Every family must retain its sixteen explicit cases")
        summary = {"complete": True, "cases": len(records), "family_counts": family_counts,
                   "usable_training_cases": sum(record["usable_for_training"] for record in records),
                   "large_spectral_cases": sum(record["rows"] >= 1024 and record["spectral"] for record in records),
                   "maximum_actual_rows": max(record["rows"] for record in records),
                   "maximum_actual_columns": max(record["columns"] for record in records),
                   "maximum_absolute_normalized_target": max(record["maximum_absolute_normalized_target"] for record in records),
                   "training_recipe_selected": False, "model_inference_performed": False,
                   "training_performed": False, "checkpoint_selection_performed": False,
                   "normalization_independently_recomputed": True, "config_sha256": sha256(output / "config.json"),
                   "array_sha256": files, "interpretation": config["purpose"]}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"state": "complete", "cases": len(records), "pid": os.getpid(),
                                            "config_sha256": sha256(output / "config.json"), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "cases": len(records), "error": repr(error),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    (register if args.stage == "register" else run)(args.output.resolve())


if __name__ == "__main__":
    main()
