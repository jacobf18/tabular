"""Verify synthetic loss-gradient balance at the actual BF16 GPU context sizes."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from diagnose_single_model_loss_gradients import ROOT, cross_geometry, geometry, gradient
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from screen_single_matrixformer_initializers import restore_episode
from single_model_distillation import blend_predictions, episode_metrics
from streamed_large_matrix_training import STREAM_OPTIONS
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import INPUTS, setup, storage_guard
from train_streamed_large_matrices import wait_for_gpu


CASE_INDICES = tuple(range(8)) + tuple(range(16, 28))


def read(path):
    return json.loads(Path(path).read_text())


def campaign_ready(campaign):
    status = read(campaign / "status.json")
    if status["state"] in ("complete", "not_promoted"):
        return True
    if status["state"] in ("failed", "interrupted"):
        raise RuntimeError("Resolve the stopped second-run campaign before the GPU diagnostic")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        if read(campaign / "status.json")["state"] in ("complete", "not_promoted"):
            return True
        raise RuntimeError("Campaign process is missing; do not trust a stale status or restart it") from None
    return False


def prepare(output):
    bank_path = ROOT / "initializer_screen_v1/synthetic_bank.pt"
    checkpoint = ROOT / "mixed_distill_v1/best_validation_checkpoint.pth"
    base_path = checkpoint.parent / "config.json"
    cpu_summary = ROOT / "loss_gradient_diagnostic_v1/summary.json"
    cpu = read(cpu_summary)
    if not cpu["complete"] or cpu["cases"] != 16 or not cpu["model_state_unchanged"] or cpu["cuda_initialized"]:
        raise ValueError("Require the completed read-only CPU diagnostic")
    inputs = {**INPUTS, str(bank_path): "13b84fb32d70a90f7f30a2695c76aef7d6bf19f5ed0065a46a66a99ec4158fbe",
              str(checkpoint): "ef5b7b9f9e199de7545d06a830de654afbe3d5674ed1488f8c95b8cbe4a651ae",
              str(base_path): sha256(base_path), str(cpu_summary): sha256(cpu_summary)}
    verify_hashes(inputs)
    base = deepcopy(read(base_path))
    verify_hashes(base["input_sha256"])
    verify_hashes(base["source_sha256"])
    verify_hashes(base["teacher_config_sha256"])
    base["parent_checkpoint"] = str(checkpoint)
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config = {"checkpoint": str(checkpoint), "bank": str(bank_path), "case_indices": list(CASE_INDICES),
              "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "base": base, "after_campaign": str(ROOT / "mixed_distill_v2_benchmarks"),
              "gradient_seed_start": 19100501, "precision": "FP32 weights, BF16 CUDA autocast; actual streamed training microbatches",
              "losses": "Unchanged complete loss and 2*Huber term; weighted NLL/CDF gradient is their difference",
              "training_performed": False, "real_benchmark_inputs_used": False, "checkpoint_selection_performed": False,
              "limitations": "Reused synthetic development bank, few cases per geometry, no optimizer updates; diagnostic only, not benchmark performance"}
    write_json(output / "config.json", config)
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return config


def run(config, output):
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    original = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)["model"]
    model, optimizer, criterion, teachers, sampler = setup(config["base"])
    del optimizer, sampler
    write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
    groups = {}
    for position, index in enumerate(config["case_indices"]):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        episode = restore_episode(bank[index], "cuda")
        if episode.metadata.get("gradient_data") != "synthetic":
            raise ValueError("Only cached synthetic episodes are allowed")
        with torch.no_grad(), compute_context(episode.source.device):
            general = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
            specialist = streamed_point_predictions(teachers[1], episode.source, criterion, **STREAM_OPTIONS)
            if episode.metadata["domain"] == "imputation":
                target, routing = blend_predictions(episode.source, general, specialist)
            else:
                target, routing = specialist, {"specialist_weight": 1.}
            quality = {"general": episode_metrics(general, episode), "specialist": episode_metrics(specialist, episode),
                       "teacher": episode_metrics(target, episode)}
        seed = config["gradient_seed_start"] + position
        with compute_context(episode.source.device):
            total_loss, vector = gradient(model, criterion, episode, target, component="total", seed=seed)
            total = vector.cpu()
            del vector
            point_loss, vector = gradient(model, criterion, episode, target, component="point", seed=seed)
            point = vector.cpu()
            del vector
        names = [episode.metadata["domain"]]
        if names[0] == "recsys":
            names.append(f"recsys_{episode.metadata['rows']}_rows")
        else:
            names.append("imputation_small" if index < 8 else "imputation_large")
        for name in names:
            if name not in groups:
                groups[name] = {"total": torch.zeros_like(total), "point": torch.zeros_like(point), "cases": 0}
            groups[name]["total"].add_(total)
            groups[name]["point"].add_(point)
            groups[name]["cases"] += 1
        report = {"case_index": index, **episode.metadata, "queries": int(episode.mask.sum()),
                  "routing": routing, "quality": quality, "total_loss": total_loss, "point_loss": point_loss,
                  "probability_loss": total_loss - point_loss, "gradient": geometry(total, point),
                  "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
        write_json(output / "cases" / f"{index}.json", report)
        write_json(output / "status.json", {"state": "diagnosing", "completed": position + 1,
                                            "expected": len(config["case_indices"]), "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case_index": index, "domain": episode.metadata["domain"], "gradient": report["gradient"],
                          "seconds": report["seconds"], "peak_gib": report["peak_gib"]}), flush=True)
        del episode, general, specialist, target, total, point
    summaries = {}
    for name, group in groups.items():
        for component in ("total", "point"):
            group[component].div_(group["cases"])
        summaries[name] = {"cases": group["cases"], "mean_gradient": geometry(group["total"], group["point"])}
    cross = {component: cross_geometry(groups["imputation"][component], groups["recsys"][component])
             for component in ("total", "point")}
    if any(not torch.equal(value.cpu(), original[name]) for name, value in model.state_dict().items()):
        raise ValueError("Diagnostic must not change model state")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(output / "summary.json", {"complete": True, "cases": len(config["case_indices"]), "groups": summaries,
                                         "imputation_vs_recsys_mean_gradient": cross, "model_state_unchanged": True,
                                         "config_sha256": sha256(output / "config.json"), "limitations": config["limitations"]})
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-campaign", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage_guard(output)
            config = prepare(output)
            while not campaign_ready(Path(config["after_campaign"])):
                if not arguments.wait_for_campaign:
                    raise RuntimeError("The existing second-run campaign is still live")
                write_json(output / "status.json", {"state": "waiting_for_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait_for_campaign, False)
            run(config, output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
