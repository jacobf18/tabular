"""Check inference parity and user-context capacity without reading held-out ratings."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from evaluate_matrixformer_recsys_ratings import CHECKPOINT, CHECKPOINT_SHA256, FROZEN_SOURCE, REPO, normalization_parameters
from recsys_evaluation_common import sha256, write_json
from tabimpute.streamed_inference import DenseReferenceImputer, StreamedImputePFN


DEFAULT_OUTPUT = REPO / "runs/benchmark_evaluations/matrixformer_streaming_probe_20260930"


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def measure(imputer, observed, means, stds):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    predicted = imputer.impute(observed.copy(), means=means, stds=stds)
    torch.cuda.synchronize()
    measured = {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "finite": bool(np.isfinite(predicted).all()),
        "observed_unchanged": bool(np.array_equal(predicted[np.isfinite(observed)], observed[np.isfinite(observed)])),
    }
    if not measured["finite"] or not measured["observed_unchanged"]:
        raise ValueError("Invalid inference output")
    return predicted, measured


def run(output):
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Wrong checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    split_path = FROZEN_SOURCE / "netflix_small/split.npz"
    expected_hash = json.loads((FROZEN_SOURCE / "artifact_sha256.json").read_text())["netflix_small/split.npz"]
    if sha256(split_path) != expected_hash:
        raise ValueError("Frozen split changed")
    with np.load(split_path, allow_pickle=False) as split:
        observed = np.full(tuple(split["shape"]), np.nan)
        observed[split["train_users"], split["train_items"]] = split["train_ratings"]
    means, stds = normalization_parameters(observed, "global")
    configuration = json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    parameters = dict(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                      json_config={"config": configuration}, model_family="new_stable", prediction_stat="mean",
                      precision_mode="amp_bfloat16", max_num_rows=2000, max_num_cols=1000,
                      max_num_chunks=1, preprocessors=None, verbose=False)
    manifest = {
        "checkpoint_sha256": CHECKPOINT_SHA256, "split_sha256": expected_hash,
        "source_hashes": {str(path): sha256(path) for path in
                          (Path(__file__), REPO / "src/tabimpute/streamed_inference.py",
                           REPO / "src/tabimpute/model/model_new_stable.py")},
        "protocol": "Only observed training entries loaded; first user/item blocks; no held-out labels or scores",
        "parity_block": [2000, 1000], "capacity_rows": [4000, 8000, 9976], "item_columns": 1000,
        "attention_batch_size": 32, "cell_batch_size": 8192, "encoder_row_batch_size": 32,
        "parity_max_abs_tolerance": 0.02, "parity_rms_tolerance": 0.001,
        "precision": "amp_bfloat16", "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "numpy": np.__version__, "training": "None",
        "selection": "Largest memory-feasible user context; no test-error-based selection",
    }
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(Path(__file__), output / "probe_snapshot.py")
    shutil.copyfile(REPO / "src/tabimpute/streamed_inference.py", output / "streamed_inference_snapshot.py")
    write_json(output / "status.json", {"state": "running", "stage": "dense reference"})
    dense = DenseReferenceImputer(**parameters)
    initial_state = state_digest(dense.model)
    reference, dense_measurement = measure(dense, observed[:2000, :1000], means[:1000], stds[:1000])
    assert state_digest(dense.model) == initial_state
    del dense
    torch.cuda.empty_cache()
    streamed = StreamedImputePFN(**parameters)
    assert state_digest(streamed.model) == initial_state
    write_json(output / "status.json", {"state": "running", "stage": "streaming parity"})
    actual, stream_measurement = measure(streamed, observed[:2000, :1000], means[:1000], stds[:1000])
    error = actual - reference
    parity = {"dense": dense_measurement, "streamed": stream_measurement,
              "max_abs_difference": float(np.abs(error).max()), "rms_difference": float(np.sqrt(np.mean(error**2))),
              "weights_unchanged": state_digest(streamed.model) == initial_state,
              "gradients_absent": all(parameter.grad is None for parameter in streamed.model.parameters())}
    parity["passed"] = (parity["max_abs_difference"] <= manifest["parity_max_abs_tolerance"] and
                        parity["rms_difference"] <= manifest["parity_rms_tolerance"] and
                        parity["weights_unchanged"] and parity["gradients_absent"])
    write_json(output / "parity.json", parity)
    print(json.dumps({"parity": parity}), flush=True)
    if not parity["passed"]:
        write_json(output / "status.json", {"state": "failed", "stage": "parity"})
        raise ValueError("Streaming parity check failed")
    del reference, actual, error
    records = []
    for rows in manifest["capacity_rows"]:
        write_json(output / "status.json", {"state": "running", "stage": "capacity", "rows": rows})
        streamed.max_num_rows = rows
        try:
            prediction, measurement = measure(streamed, observed[:rows, :1000], means[:1000], stds[:1000])
            del prediction
            record = {"rows": rows, "columns": 1000, "status": "passed", **measurement}
        except torch.cuda.OutOfMemoryError as error:
            record = {"rows": rows, "columns": 1000, "status": "out_of_memory", "error": str(error)}
            torch.cuda.empty_cache()
        records.append(record)
        write_json(output / "capacity.json", records)
        print(json.dumps(record), flush=True)
        if record["status"] != "passed":
            break
    assert state_digest(streamed.model) == initial_state
    assert all(parameter.grad is None for parameter in streamed.model.parameters())
    write_json(output / "status.json", {
        "state": "complete", "max_verified_rows": max([2000] + [row["rows"] for row in records if row["status"] == "passed"]),
        "weights_unchanged": True, "gradients_absent": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
