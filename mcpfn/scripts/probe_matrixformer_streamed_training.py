"""Small synthetic full-model training smoke test; never saves updated weights."""

import argparse
from contextlib import nullcontext
import gc
from importlib import resources
import json
from pathlib import Path
import shutil
import time

import torch

from recsys_evaluation_common import CHECKPOINT, REPO, sha256, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model.model_new_stable_memory_efficient import TabImputeModelMemoryEfficient
from tabimpute.streamed_training import streamed_training_loss


def difference(actual, reference):
    squared_error = squared_reference = maximum = 0.0
    for name, value in actual.items():
        expected = reference[name]
        error = value.double() - expected.double()
        maximum = max(maximum, error.abs().max().item())
        squared_error += error.square().sum().item()
        squared_reference += expected.double().square().sum().item()
    return {"max_absolute": maximum, "relative_l2": (squared_error / max(squared_reference, 1e-30)) ** 0.5}


def measure(configuration, state, values, targets, mask, distribution, mode, precision, options):
    model_class = TabImputeModelMemoryEfficient if mode == "block_checkpoint" else TabImputeModelStable
    model = model_class(**configuration).cuda().train()
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, foreach=False)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with context:
        if mode == "streamed":
            loss = streamed_training_loss(model, values, targets, mask, distribution, **options)
        else:
            logits = model(values, decode_mask=mask)
            with torch.autocast("cuda", enabled=False):
                loss = distribution(logits.float(), targets[mask]).mean()
            del logits
    loss.backward()
    torch.cuda.synchronize()
    forward_backward_seconds = time.perf_counter() - started
    backward_peak = torch.cuda.max_memory_allocated()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing or nonfinite gradient: {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    optimizer_started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_started
    step_peak = torch.cuda.max_memory_allocated()
    updated = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    if not all(torch.isfinite(value).all() for value in updated.values()):
        raise ValueError("Nonfinite optimizer update")
    changes = sum(not torch.equal(value, state[name]) for name, value in updated.items())
    if not changes or not torch.isfinite(loss):
        raise ValueError("No optimizer update or nonfinite loss")
    measurement = {
        "mode": mode, "precision": precision, "loss": float(loss.detach()),
        "initial_allocated_gib": initial_allocated / 1024**3,
        "forward_backward_peak_gib": backward_peak / 1024**3,
        "full_step_peak_gib": step_peak / 1024**3,
        "forward_backward_seconds": forward_backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "gradient_tensors": len(gradients), "updated_parameter_tensors": changes,
        "all_gradients_and_parameters_finite": True,
    }
    del model, optimizer, loss, parameter
    gc.collect()
    torch.cuda.empty_cache()
    return measurement, gradients, updated


def run(output, rows, columns):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(741)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.2)
    checkpoint_hash = sha256(CHECKPOINT)
    configuration = json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    if any(configuration[name] for name in ("attention_dropout", "ffn_dropout", "embedding_dropout", "drop_path_rate")):
        raise ValueError("Dense gradient parity requires this checkpoint's dropout-free configuration")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as handle:
        borders = torch.load(handle, map_location="cuda", weights_only=True)
    distribution = FullSupportBarDistribution(borders)
    targets = torch.randn(1, rows, columns).cuda()
    mask = (torch.rand(1, rows, columns) < 0.6).cuda()
    values = targets.masked_fill(mask, torch.nan)
    options = {"attention_batch_size": 8, "cell_batch_size": 256, "encoder_row_batch_size": 16}
    sources = [Path(__file__), REPO / "src/tabimpute/streamed_training.py",
               REPO / "src/tabimpute/model/model_new_stable.py",
               REPO / "src/tabimpute/model/model_new_stable_memory_efficient.py",
               REPO / "src/tabimpute/model/bar_distribution.py"]
    manifest = {
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash,
        "model": configuration, "shape": list(values.shape), "masked_cells": int(mask.sum()),
        "seed": 741, "options": options, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "objective": "Mean masked bar-distribution NLL on synthetic targets",
        "optimizer": {"kind": "AdamW", "lr": 1e-5, "weight_decay": 0.0, "foreach": False},
        "scope": "One independent training step per mode/precision, reset from the same checkpoint; not the full teacher-distillation recipe",
        "memory": "Per-process PyTorch allocated bytes, including gradients and optimizer state; first-step peak, not a capacity limit",
        "timing_caveat": "One warmup per mode/precision, but sharing the GPU with recommender evaluation; not a throughput benchmark",
        "gates": {"fp32_loss_absolute": 1e-5, "fp32_gradient_relative_l2": 1e-4,
                  "bf16_loss_absolute": 5e-3, "bf16_gradient_relative_l2": 0.03},
        "source_hashes": {str(path): sha256(path) for path in sources},
    }
    write_json(output / "manifest.json", manifest)
    for path in sources:
        shutil.copyfile(path, output / (path.stem + "_snapshot.py"))
    for precision in ("fp32", "bf16"):
        for mode in ("dense", "block_checkpoint", "streamed"):
            write_json(output / "status.json", {"state": "warming_up", "precision": precision, "mode": mode})
            measure(configuration, state, values, targets, mask, distribution, mode, precision, options)
    measurements = []
    for precision in ("fp32", "bf16"):
        reference_gradients = reference_updated = reference_loss = None
        for mode in ("dense", "block_checkpoint", "streamed"):
            write_json(output / "status.json", {"state": "running", "precision": precision, "mode": mode})
            result, gradients, updated = measure(
                configuration, state, values, targets, mask, distribution, mode, precision, options,
            )
            if mode == "dense":
                reference_gradients, reference_updated, reference_loss = gradients, updated, result["loss"]
            result["loss_absolute_difference"] = abs(result["loss"] - reference_loss)
            result["gradient_difference"] = difference(gradients, reference_gradients)
            result["updated_weights_difference"] = difference(updated, reference_updated)
            result["parity_passed"] = (
                result["loss_absolute_difference"] <= manifest["gates"][f"{precision}_loss_absolute"]
                and result["gradient_difference"]["relative_l2"] <= manifest["gates"][f"{precision}_gradient_relative_l2"]
            )
            measurements.append(result)
            write_json(output / "results.json", measurements)
            print(json.dumps(result), flush=True)
            del gradients, updated
        del reference_gradients, reference_updated
    checkpoint_unchanged = sha256(CHECKPOINT) == checkpoint_hash
    passed = checkpoint_unchanged and all(result["parity_passed"] for result in measurements)
    write_json(output / "status.json", {"state": "complete", "passed": passed, "checkpoint_unchanged": checkpoint_unchanged})
    lines = ["# Small-matrix streamed training smoke test", "",
             f"Shape: 1 × {rows} × {columns}; actual 8-layer, 768-dimensional paper checkpoint.",
             "Each case performs forward, backward, and one AdamW update on a disposable model copy.",
             "Synthetic masked NLL only; no checkpoint is overwritten and no benchmark labels are used.", "",
             "| Precision | Execution | Loss | Peak step GiB | Gradient relative L2 difference | Parity |",
             "|---|---|---:|---:|---:|---|"]
    for result in measurements:
        lines.append(f"| {result['precision']} | {result['mode']} | {result['loss']:.8f} | "
                     f"{result['full_step_peak_gib']:.4f} | {result['gradient_difference']['relative_l2']:.3g} | "
                     f"{result['parity_passed']} |")
    lines += ["", "Peak memory includes parameter gradients and the first allocation of AdamW state.",
              "FP32 checks numerical equivalence; BF16 allows the separately recorded mixed-precision tolerance.",
              "Dropout-free checkpoint; a separate unit test checks checkpoint RNG preservation with dropout enabled.",
              "The GPU is shared with an evaluation job, so recorded timings are not throughput measurements.",
              "This does not establish a maximum training matrix size or validate the complete distillation recipe.",
              f"Checkpoint unchanged: {checkpoint_unchanged}. All parity gates passed: {passed}."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("Training smoke-test parity failed; inspect saved results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--columns", type=int, default=32)
    arguments = parser.parse_args()
    run(arguments.output, arguments.rows, arguments.columns)
