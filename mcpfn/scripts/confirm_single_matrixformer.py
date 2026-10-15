"""Confirm a frozen pilot selection on disjoint synthetic seeds, without training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import sha256, write_json
from single_model_distillation import episode_metrics, selection_score, teacher_target
from streamed_large_matrix_training import STREAM_OPTIONS
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer import setup
from train_streamed_large_matrices import append_json, wait_for_gpu


def confirmation_specs():
    specifications = [dict(domain="imputation", shape=(240, 80), component=component,
                           transpose=bool(repeat % 2)) for repeat in range(3) for component in range(8)]
    specifications.extend(dict(domain="imputation", shape=(2000, 1000), component=component,
                               transpose=bool(component % 2)) for component in range(6))
    specifications.extend(dict(domain="recsys", shape=(1000, 1000)) for _ in range(16))
    return specifications


def aggregate(cases):
    return {role: {domain: {metric: sum(case[role][metric] for case in cases if case["domain"] == domain)
                                      / sum(case["domain"] == domain for case in cases)
                           for metric in ("rmse", "mae")}
                   for domain in ("imputation", "recsys")}
            for role in ("student", "parent", "teacher")}


@torch.no_grad()
def confirm(run, output):
    config = json.loads((run / "config.json").read_text())
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    verify_hashes(config["teacher_config_sha256"])
    checkpoint = run / "best_validation_checkpoint.pth"
    fingerprint = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_sha256"] != sha256(run / "config.json"):
        raise ValueError("Checkpoint/config mismatch")
    specifications = confirmation_specs()
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint,
                "config_sha256": state["config_sha256"], "selected_step": state["step"],
                "seed_start": 10100501, "cases": specifications,
                "source_sha256": sha256(Path(__file__)), "training_performed": False,
                "purpose": "Independent synthetic retention screen; not a real-benchmark success claim"}
    write_json(output / "manifest.json", manifest)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    if state["step"] == 0:
        write_json(output / "gate.json", {"passed": False, "reason": "Pilot selected the unchanged initialization",
                                         "checkpoint_sha256": fingerprint, "selected_step": 0})
        return
    student, optimizer, criterion, teachers, sampler = setup(config)
    del optimizer
    student.load_state_dict(state["model"], strict=True)
    student.eval()
    del state
    cases = []
    for index, specification in enumerate(specifications):
        episode = sampler.sample(step=1, seed=manifest["seed_start"] + index, **specification)
        target, decision = teacher_target(teachers, criterion, episode)
        with compute_context(episode.source.device):
            prediction = streamed_point_predictions(student, episode.source, criterion, **STREAM_OPTIONS)
            parent = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
        case = {"index": index, **episode.metadata, "routing": decision,
                "student": episode_metrics(prediction, episode), "parent": episode_metrics(parent, episode),
                "teacher": episode_metrics(target, episode)}
        cases.append(case)
        append_json(output / "cases.jsonl", case)
        write_json(output / "status.json", {"state": "evaluating", "completed": len(cases), "expected": len(specifications),
                                            "pid": os.getpid(), "updated_unix": time.time()})
        print(json.dumps({"case": index, "domain": specification["domain"], "student": case["student"],
                          "parent": case["parent"]}), flush=True)
        del episode, target, prediction, parent
    metrics = aggregate(cases)
    score, eligible = selection_score(metrics["student"], metrics["parent"], metrics["teacher"])
    baseline_score, _ = selection_score(metrics["parent"], metrics["parent"], metrics["teacher"])
    gate = {"passed": eligible and score < baseline_score, "retention_eligible": eligible,
            "score": score, "parent_score": baseline_score, "metrics": metrics,
            "checkpoint_sha256": fingerprint, "selected_step": manifest["selected_step"],
            "real_benchmark_success": False, "cases": len(cases)}
    if sha256(checkpoint) != fingerprint:
        raise ValueError("Selected checkpoint changed during confirmation")
    write_json(output / "gate.json", gate)
    print(json.dumps({"gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-training", action="store_true")
    arguments = parser.parse_args()
    output, run = arguments.output.resolve(), arguments.run.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                status = json.loads((run / "status.json").read_text())
                if status["state"] == "training_complete":
                    break
                if not arguments.wait_for_training or status["state"] in {"failed", "interrupted"}:
                    raise RuntimeError(f"Training is not complete: {status}")
                os.kill(status["pid"], 0)
                write_json(output / "status.json", {"state": "waiting_for_training", "pid": os.getpid(),
                                                    "training_step": status.get("step"), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait_for_training, False)
            confirm(run, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error),
                                                "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
