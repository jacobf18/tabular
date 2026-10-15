"""Check point-metric helper parity on cached training teachers, without neural evaluation."""

import math
import os
from pathlib import Path
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import single_model_point_metric_gradients as metrics
import train_single_matrixformer_family_projection as training


THROUGH_STEP = 75
OUTPUT = training.ROOT / "point_metric_parity_v1"


def run(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Prediction-space metric checks must be CPU-only")
    config = training.verify_registration(training.PAIR)
    sources = {str(Path(__file__).resolve()): sha256(Path(__file__)),
               str(Path(metrics.__file__).resolve()): sha256(Path(metrics.__file__))}
    inputs = {str(training.PAIR / "config.json"): sha256(training.PAIR / "config.json")}
    tasks = []
    for batch in config["batches"][:THROUGH_STEP]:
        path = training.PAIR / "training_batches" / f'step_{batch["step"]:03d}' / "manifest.json"
        inputs[str(path)] = sha256(path)
        manifest = training.read(path)
        if manifest["step"] != batch["step"] or len(manifest["tasks"]) != 14:
            raise ValueError("Preserve all completed training tasks")
        for expected, logged in zip(batch["tasks"], manifest["tasks"]):
            if {key: logged[key] for key in expected} != expected:
                raise ValueError("Cached task differs from its registered synthetic input")
            inputs[logged["artifact"]] = logged["sha256"]
            tasks.append({"step": batch["step"], "expected": expected, "artifact": logged["artifact"]})
    if len(tasks) != 1050:
        raise ValueError("Check all fourteen tasks at every one of the seventy-five completed updates")
    write_json(output / "config.json", {"through_step": THROUGH_STEP, "tasks": tasks, "input_sha256": inputs,
        "source_sha256": sources, "metric_specification": metrics.SPECIFICATION,
        "absolute_metric_parity_tolerance": 1e-12, "relative_metric_parity_tolerance": 1e-12,
        "scope": "Only cached synthetic training teacher predictions; no validation/confirmation labels, neural forward/backward, parameter gradients, training or benchmark evaluation"})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    verify_hashes(inputs)
    verify_hashes(sources)
    rows = []
    maximum_error = 0.
    undefined = 0
    for item in tasks:
        payload = torch.load(item["artifact"], map_location="cpu", weights_only=True)
        if payload["task"] != item["expected"]:
            raise ValueError("Saved episode must bind the exact synthetic specification")
        episode = training.mixture.backend.restore_episode(payload["episode"], "cpu")
        expected_seed = item["expected"]["specification"].get("structured", item["expected"]["specification"])["seed"]
        if episode.metadata["seed"] != expected_seed or episode.metadata["gradient_data"] != "synthetic":
            raise ValueError("Only the original cached training episodes may enter this diagnostic")
        prediction = payload["teacher"]
        reference = training.mixture.defined.episode_metrics(prediction, episode)
        terms = metrics.point_metric_terms(prediction, episode)
        records = {}
        for name, record in terms.items():
            if record is None:
                if reference[name] is not None or reference["column_coverage"]["common_nonconstant_columns"] != 0:
                    raise ValueError("Never fabricate an undefined common metric")
                undefined += 1
                records[name] = None
                continue
            error = abs(record["value"] - reference[name])
            maximum_error = max(maximum_error, error)
            if not math.isclose(record["value"], reference[name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Point metric differs from the frozen original definition")
            if name == "common_nrmse" and record["defined_units"] != reference["column_coverage"]["common_nonconstant_columns"]:
                raise ValueError("Truth-defined scored-column coverage changed")
            coefficients, offset = metrics.linearization(prediction, episode.mask, record)
            torch.testing.assert_close(coefficients / record["queries"], record["gradient"],
                                       rtol=4 * torch.finfo(torch.float64).eps, atol=0.)
            anchor = float((prediction[episode.mask].double() * coefficients[episode.mask]).mean()) + offset
            magnitude = float((prediction[episode.mask].double() * record["gradient"][episode.mask]).abs().sum())
            operations = 2 * record["queries"] + 4
            factor = operations * torch.finfo(torch.float64).eps
            bound = factor / (1 - factor) * (magnitude + abs(offset) + abs(record["value"]))
            if abs(anchor - record["value"]) > bound:
                raise ValueError("Linearization anchor exceeds the FP64 accumulation bound")
            records[name] = {"value": record["value"], "reference": reference[name], "absolute_metric_error": error,
                "prediction_gradient_norm": float(torch.linalg.vector_norm(record["gradient"])),
                "queries": record["queries"], "defined_units": record["defined_units"],
                "linearization_anchor_error": abs(anchor - record["value"]), "fp64_accumulation_bound": bound}
        rows.append({"step": item["step"], "task": item["expected"]["name"], "seed": expected_seed, "metrics": records})
        if len(rows) % 70 == 0:
            write_json(output / "status.json", {"state": "checking", "completed_tasks": len(rows), "expected_tasks": len(tasks),
                "pid": os.getpid(), "updated_unix": time.time()})
            print(f"Checked {len(rows)}/{len(tasks)} training-teacher grids", flush=True)
    verify_hashes(inputs)
    verify_hashes(sources)
    training.verify_registration(training.PAIR)
    write_json(output / "audit.json", {"complete": True, "training_tasks": len(rows), "metric_records": 2 * len(rows),
        "undefined_common_metrics_preserved": undefined, "maximum_absolute_metric_error": maximum_error, "cases": rows,
        "input_sha256": inputs, "source_sha256": sources, "cuda_initialized": torch.cuda.is_initialized(),
        "metric_values_and_linearization_checked": True, "prediction_space_gradient_unit_tests_separate": True,
        "parameter_gradients_computed": False, "neural_forward_or_backward_performed": False,
        "training_changed": False, "selection_changed": False, "goal_success_not_assessed": True})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        write_json(OUTPUT / "status.json", {"state": "starting", "pid": os.getpid(), "updated_unix": time.time()})
        run(OUTPUT)
        write_json(OUTPUT / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
