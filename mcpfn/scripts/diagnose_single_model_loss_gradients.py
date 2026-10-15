"""CPU-only loss-gradient diagnosis on fixed crops of an existing synthetic bank."""

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn import functional

from dwip_teacher_mean_training import control_recipe, initialize_student, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from screen_single_matrixformer_initializers import restore_episode
from single_model_distillation import blend_predictions, distillation_loss, episode_metrics
from streamed_large_matrix_training import STREAM_OPTIONS, TRAIN_OPTIONS
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss
from train_single_matrixformer import GENERAL, SPECIALIST, INPUTS


ROOT = REPO / "runs/single_model_goal_20261005"
CASE_INDICES = tuple(range(8)) + tuple(range(20, 28))


def cropped_case(cached, rows=128, columns=64):
    result = deepcopy(cached)
    original = tuple(result["source"].shape)
    for name in ("source", "target", "mask"):
        result[name] = result[name][:, :rows, :columns].clone()
    if result["scaling"] is not None:
        for name in ("means", "stds"):
            result["scaling"][name] = result["scaling"][name][:columns].clone()
        for name in ("row_offsets", "row_scales"):
            result["scaling"][name] = result["scaling"][name][:rows].clone()
    result["metadata"].update(diagnostic_original_shape=list(original),
                              diagnostic_crop_shape=list(result["source"].shape),
                              diagnostic_normalization="Preserved full-parent synthetic normalization; no refitting on crop")
    if result["metadata"].get("gradient_data") != "synthetic" or not result["mask"].any():
        raise ValueError("Diagnostic requires synthetic crops with nonempty fixed queries")
    return restore_episode(result, "cpu")


def dot(left, right):
    if left.shape != right.shape:
        raise ValueError("Gradient inventories differ")
    return sum(float(torch.dot(first.double(), second.double()))
               for first, second in zip(left.split(1_000_000), right.split(1_000_000)))


def geometry(total, point):
    total_square, point_square, cross = dot(total, total), dot(point, point), dot(total, point)
    probability_square = max(total_square + point_square - 2 * cross, 0.)
    total_norm, point_norm, probability_norm = map(math.sqrt, (total_square, point_square, probability_square))
    denominator = point_norm * probability_norm
    return {"total_norm": total_norm, "point_norm": point_norm, "probability_norm": probability_norm,
            "point_probability_dot": cross - point_square,
            "point_probability_cosine": (cross - point_square) / denominator if denominator > 0 else None,
            "probability_to_point_norm": probability_norm / point_norm if point_norm > 0 else None}


def cross_geometry(first, second):
    first_square, second_square, cross = dot(first, first), dot(second, second), dot(first, second)
    denominator = math.sqrt(first_square * second_square)
    return {"first_norm": math.sqrt(first_square), "second_norm": math.sqrt(second_square),
            "dot": cross, "cosine": cross / denominator if denominator > 0 else None}


def gradient(model, criterion, episode, target, *, component, seed):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    nll_weight = .1 if episode.metadata["domain"] == "recsys" else 1.
    if component == "total":
        loss_function = lambda logits, truth, teacher: distillation_loss(criterion, logits, truth, teacher, nll_weight=nll_weight)
    elif component == "point":
        loss_function = lambda logits, truth, teacher: 2. * functional.huber_loss(criterion.mean(logits.float()), teacher.detach().float(), delta=1., reduction="none")
    else:
        raise ValueError("Unknown loss-gradient component")
    loss = streamed_training_loss(model, episode.source, episode.target, episode.mask, loss_function,
                                  auxiliary_targets=target, **TRAIN_OPTIONS)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite diagnostic loss")
    loss.backward()
    vector = torch.cat([(parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
                        for parameter in model.parameters()])
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Nonfinite diagnostic gradient")
    return float(loss.detach()), vector


def run(output):
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' to guarantee no GPU context")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    bank_path = ROOT / "initializer_screen_v1/synthetic_bank.pt"
    checkpoint = ROOT / "mixed_distill_v1/best_validation_checkpoint.pth"
    inputs = {**INPUTS, str(bank_path): "13b84fb32d70a90f7f30a2695c76aef7d6bf19f5ed0065a46a66a99ec4158fbe",
              str(checkpoint): "ef5b7b9f9e199de7545d06a830de654afbe3d5674ed1488f8c95b8cbe4a651ae"}
    verify_hashes(inputs)
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config = {"checkpoint": str(checkpoint), "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "case_indices": list(CASE_INDICES), "crop_shape": [128, 64], "gradient_seed_start": 18100501,
              "precision": "FP32 CPU; no CUDA and no optimizer updates", "threads": 4,
              "losses": "Existing complete loss and its 2*Huber teacher-mean term; probability gradient is their difference",
              "inference_context": "Fixed upper-left synthetic crops preserving parent scaling and original query masks",
              "limitations": "Small crops and FP32 CPU, not full-context BF16 training; reused development bank, not fresh confirmation or benchmark evidence",
              "real_data_used": False, "training_performed": False, "checkpoint_selection_performed": False}
    write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                             "cuda_available": False, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    arguments = control_recipe()
    arguments.init_model_checkpoint = str(checkpoint)
    model, optimizer, _, criterion = initialize_student(arguments, torch.device("cpu"))
    del optimizer
    teachers = []
    for path in (GENERAL, SPECIALIST):
        teacher = deepcopy(model)
        teacher.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
        teachers.append(teacher.eval().requires_grad_(False))
    groups, cases = {}, []
    for position, index in enumerate(CASE_INDICES):
        started = time.monotonic()
        episode = cropped_case(bank[index])
        with torch.no_grad():
            general = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
            specialist = streamed_point_predictions(teachers[1], episode.source, criterion, **STREAM_OPTIONS)
            if episode.metadata["domain"] == "imputation":
                target, routing = blend_predictions(episode.source, general, specialist)
            else:
                target, routing = specialist, {"specialist_weight": 1.}
            quality = {"general": episode_metrics(general, episode), "specialist": episode_metrics(specialist, episode),
                       "teacher": episode_metrics(target, episode)}
        seed = config["gradient_seed_start"] + position
        total_loss, total = gradient(model, criterion, episode, target, component="total", seed=seed)
        point_loss, point = gradient(model, criterion, episode, target, component="point", seed=seed)
        names = [episode.metadata["domain"]]
        if names[0] == "recsys":
            names.append(f"recsys_from_{episode.metadata['rows']}_rows")
        for name in names:
            if name not in groups:
                groups[name] = {"total": torch.zeros_like(total), "point": torch.zeros_like(point), "cases": 0}
            groups[name]["total"].add_(total)
            groups[name]["point"].add_(point)
            groups[name]["cases"] += 1
        report = {"case_index": index, **episode.metadata, "queries": int(episode.mask.sum()),
                  "routing": routing, "quality": quality, "total_loss": total_loss, "point_loss": point_loss,
                  "probability_loss": total_loss - point_loss, "gradient": geometry(total, point),
                  "seconds": time.monotonic() - started}
        write_json(output / "cases" / f"{index}.json", report)
        cases.append(report)
        write_json(output / "status.json", {"state": "diagnosing", "completed": len(cases), "expected": len(CASE_INDICES),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case_index": index, "domain": episode.metadata["domain"], "gradient": report["gradient"],
                          "seconds": report["seconds"]}), flush=True)
        del episode, general, specialist, target, total, point
    summaries = {}
    for name, group in groups.items():
        for component in ("total", "point"):
            group[component].div_(group["cases"])
        summaries[name] = {"cases": group["cases"], "mean_gradient": geometry(group["total"], group["point"])}
    cross = {component: cross_geometry(groups["imputation"][component], groups["recsys"][component])
             for component in ("total", "point")}
    if any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()):
        raise ValueError("Diagnostic must not change model state")
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    write_json(output / "summary.json", {"complete": True, "cases": len(cases), "groups": summaries,
                                         "imputation_vs_recsys_mean_gradient": cross,
                                         "model_state_unchanged": True, "cuda_initialized": torch.cuda.is_initialized(),
                                         "config_sha256": sha256(output / "config.json"), "limitations": config["limitations"]})
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(output / "status.json", {"state": "starting", "pid": os.getpid(), "updated_unix": time.time()})
            run(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
