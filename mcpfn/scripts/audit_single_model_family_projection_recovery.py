"""Reaudit the unchanged preflight with FP64 CPU norm accumulation, not looser tolerances."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_family_projection as original
from recsys_evaluation_common import sha256, write_json


@contextmanager
def precise_cpu_norms():
    previous = torch.linalg.vector_norm
    calls = []

    def vector_norm(value, *args, **kwargs):
        if value.device.type == "cpu" and value.dtype == torch.float32 and kwargs.get("dtype") is None:
            if kwargs.get("out") is not None:
                raise ValueError("Precision recovery must not alter an explicit output tensor")
            calls.append({"elements": value.numel(), "input_dtype": str(value.dtype), "accumulation_dtype": "torch.float64"})
            return previous(value, *args, **{**kwargs, "dtype": torch.float64})
        return previous(value, *args, **kwargs)

    torch.linalg.vector_norm = vector_norm
    try:
        yield calls
    finally:
        torch.linalg.vector_norm = previous


def diagnose(source):
    result = original.read(source / "result.json")
    parameters = original.read(source / "parameter_inventory.json")
    state = torch.load(result["optimizer_step_artifact"], map_location="cpu", weights_only=True, mmap=True)
    parts = state["combined_gradient"].split([record[2] for record in parameters])
    low = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(part) for part in parts])))
    high = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(part.double()) for part in parts])))
    recorded = result["global_gradient_norm_before_clip"]
    np.testing.assert_allclose(high, recorded, rtol=1e-6, atol=1e-10)
    if max(low, high, recorded) >= .5 - 1e-6 or not torch.equal(state["combined_gradient"], state["applied_gradient"]):
        raise ValueError("This recovery is restricted to the verified inactive global clipping branch")
    return {"recorded_cuda_fp32": recorded, "original_cpu_fp32": low, "corrected_cpu_fp64": high,
            "cpu_fp32_relative_difference": abs(low / recorded - 1), "cpu_fp64_relative_difference": abs(high / recorded - 1),
            "unchanged_relative_tolerance": 1e-6, "unchanged_absolute_tolerance": 1e-10,
            "global_clip_active": False, "applied_gradient_exactly_equals_combined": True}


def recover(source, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Recovery must be CPU-only")
    if output.exists() or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Keep recovery separate and never replace existing evidence")
    original.verify_registration(source)
    status = original.read(source / "status.json")
    error = (source / "audit.log").read_text()
    if status["state"] != "failed" or "global_norm, result" not in error or "rtol=1e-06, atol=1e-10" not in error:
        raise ValueError("Require the preserved original CPU norm-audit failure")
    diagnosis = diagnose(source)
    names = ("config.json", "registration.json", "result.json", "tasks.json", "parameter_inventory.json")
    hashes = {str(source / name): sha256(source / name) for name in (*names, "status.json", "audit.log")}
    hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    hashes[str(Path(original.__file__).resolve())] = sha256(Path(original.__file__))
    output.mkdir(parents=True)
    view = output / "original_metadata"
    view.mkdir()
    for name in names:
        shutil.copy2(source / name, view / name)
        if sha256(view / name) != hashes[str(source / name)]:
            raise ValueError("Original preflight metadata must be copied byte-for-byte")
    write_json(output / "registration.json", {"registered_unix": time.time(), "source": str(source),
               "input_sha256": hashes, "diagnosis": diagnosis,
               "change": "Promote only implicit CPU FP32 vector-norm accumulation to FP64; preserve every original assertion and numerical tolerance",
               "training_restarted": False, "source_artifacts_modified": False, "selection_tolerance_changed": False})
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    try:
        write_json(output / "status.json", {"state": "auditing_unchanged_preflight", "pid": os.getpid(), "updated_unix": time.time()})
        with precise_cpu_norms() as calls:
            original.audit(view)
        sizes = [record[2] for record in original.read(source / "parameter_inventory.json")]
        if [call["elements"] for call in calls] != sizes:
            raise ValueError("The precision correction must apply only to the original per-parameter CPU norm reductions")
        verify_hashes(hashes)
        audit = original.read(view / "audit.json")
        write_json(output / "audit.json", {**audit, "recovery_complete": True, "norm_precision_diagnosis": diagnosis,
                   "promoted_cpu_norm_calls": calls, "original_preflight_status_preserved": True,
                   "original_failure_preserved": True, "audit_view": str(view), "audit_sha256": sha256(view / "audit.json"),
                   "recovery_input_sha256": hashes, "training_restarted": False, "selection_tolerance_changed": False})
        text = (view / "RESULTS.md").read_text()
        text += ("\nThe original audit failure remains untouched. This separate recovery accumulates CPU parameter norms in FP64, "
                 "preserves the original rtol1e-6/atol1e-10 assertions, and verifies that global clipping was inactive. "
                 "No training, checkpoint, gate, or evaluation is rerun or changed.\n")
        (output / "RESULTS.md").write_text(text)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
