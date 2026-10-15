"""Check actual synthetic point-metric changes for the predeclared diagnostic proposals."""

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import diagnose_single_model_point_alignment as alignment
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
from single_model_point_metric_gradients import point_metric_terms
from single_model_streamed_point_gradients import streamed_point_predictions
from train_streamed_large_matrices import append_json, wait_for_gpu


SOURCE = alignment.ROOT / "point_alignment_v1"
REPEATS = 2
ROLES = ("before", "adam", "composite_only", "composite_and_points")
POLICY = ("Evaluate every available predeclared point-alignment proposal on the same fourteen synthetic training tasks "
          "at both fixed states, twice per role with original task seeds. Include before and actual Adam, composite-only "
          "protection, and joint composite-plus-point protection; preserve unavailable projections explicitly. "
          "Use the exact point-gradient decoder chunking and original scorer. Measure repeat spread and capture/replay "
          "differences instead of assuming bitwise deterministic inference. Report every signed case/metric change; "
          "no finite-difference tuning, step-size search, trained checkpoint, selection, validation, confirmation, or benchmarks.")


def read(path):
    return json.loads(Path(path).read_text())


def signed_change(before, after):
    if not all(math.isfinite(value) and value >= 0 for value in (before, after)):
        raise ValueError("Require finite nonnegative point metrics")
    difference = after - before
    return {"before": before, "after": after, "difference": difference,
            "relative_difference": difference / before if before else None,
            "direction": "worse" if difference > 0 else "better" if difference < 0 else "equal"}


def verify_registration(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["source"] != str(SOURCE) or config["roles"] != list(ROLES)
            or config["repeats"] != REPEATS or config["policy"] != POLICY):
        raise ValueError("Never change registered finite-step diagnostics")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite a diagnostic")
    if shutil.disk_usage(alignment.ROOT).free < 24 * 1024**3:
        raise OSError("Keep at least18 GiB free after saving finite-step predictions")
    source_config = alignment.verify_registration(SOURCE)
    if read(SOURCE / "status.json")["state"] != "complete" or not read(SOURCE / "audit.json")["complete"]:
        raise ValueError("Require the complete independent gradient/projection audit")
    audit = read(SOURCE / "audit.json")
    inputs = {**source_config["input_sha256"], **audit["input_sha256"]}
    inputs.update({str(SOURCE / filename): sha256(SOURCE / filename)
                   for filename in ("config.json", "registration.json", "audit.json", "result.json")})
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_point_finite_steps.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    hashes = {**source_config["source_sha256"], **{str(path): sha256(path) for path in sources}}
    verify_hashes(inputs)
    verify_hashes(hashes)
    output.mkdir(parents=True)
    write_json(output / "config.json", {"source": str(SOURCE), "source_config": source_config, "roles": list(ROLES),
               "repeats": REPEATS, "policy": POLICY, "input_sha256": inputs, "source_sha256": hashes})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "finite_step_model_evaluated": False, "checkpoint_selection_performed": False})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def role_parameters(result, snapshot, role):
    if role == "before":
        return snapshot["before"], None
    if role == "adam":
        return snapshot["tentative"], result["adam_products"]
    selected = result["projections"][role]
    if selected["accepted_index"] is None:
        return None, None
    attempt = selected["attempts"][selected["accepted_index"]]
    saved = torch.load(attempt["artifact"], map_location="cpu", weights_only=True, mmap=True)
    if not saved["accepted"]:
        raise ValueError("Never evaluate a rejected projection as accepted")
    return saved["parameters"], attempt["all_products"]


def summarize_state(result, rows):
    summaries, changes = {}, []
    maximum_repeat_spread = 0.
    maximum_capture_replay_difference = 0.
    for role in ROLES:
        selected = [row for row in rows if row["role"] == role]
        if not selected:
            summaries[role] = {"available": False}
            continue
        if len(selected) != 14 * REPEATS:
            raise ValueError("Every task and repeat is mandatory for an available role")
        counts = {"better": 0, "worse": 0, "equal": 0, "undefined": 0}
        for task in result["tasks"]:
            actual = [row for row in selected if row["task"] == task["task"]]
            baseline = [row for row in rows if row["role"] == "before" and row["task"] == task["task"]]
            if (len(actual) != REPEATS or {row["repeat"] for row in actual} != set(range(REPEATS))
                    or len(baseline) != REPEATS):
                raise ValueError("Require exactly the original repeated case panel")
            for metric, captured in task["metrics"].items():
                current = [row["metrics"][metric] for row in actual]
                initial = [row["metrics"][metric] for row in baseline]
                if captured is None:
                    if any(record is not None for record in (*current, *initial)):
                        raise ValueError("Undefined common metric coverage changed")
                    counts["undefined"] += 1
                    continue
                values = [record["value"] for record in current]
                anchors = [record["value"] for record in initial]
                maximum_repeat_spread = max(maximum_repeat_spread, max(values) - min(values))
                if role == "before":
                    maximum_capture_replay_difference = max(maximum_capture_replay_difference,
                        max(abs(value - captured["value"]) for value in values))
                row = {"role": role, "task": task["task"], "metric": metric,
                       **signed_change(float(np.mean(anchors)), float(np.mean(values))),
                       "before_repeats": anchors, "after_repeats": values,
                       "captured_gradient_forward_value": captured["value"]}
                if role != "before":
                    index = next(entry["index"] for entry in result["entries"]
                                 if entry["task"] == task["task"] and entry["kind"] == metric)
                    row["first_order_predicted_change"] = -actual[0]["products"][index]
                changes.append(row)
                counts[row["direction"]] += 1
        summaries[role] = {"available": True, **counts}
    return {"roles": summaries, "changes": changes, "maximum_repeat_metric_spread": maximum_repeat_spread,
            "maximum_capture_replay_metric_difference": maximum_capture_replay_difference}


def execute(output):
    config = verify_registration(output)
    source_config = config["source_config"]
    wait_for_gpu(output, True, False)
    model, optimizer, criterion, teachers, sampler = alignment.mixture.original_setup(source_config["base"])
    del optimizer, teachers, sampler
    model.train()
    state_results = {}
    prediction_hashes = {}
    for name, specification in source_config["states"].items():
        directory = output / name
        directory.mkdir()
        result = read(SOURCE / name / "result.json")
        snapshot = torch.load(result["proposal"], map_location="cpu", weights_only=True, mmap=True)
        checkpoint = torch.load(specification["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        if not torch.equal(alignment.preflight.flatten(model).cpu(), snapshot["before"]):
            raise ValueError("Proposal must bind the unchanged original full parameter vector")
        rows = []
        for role in ROLES:
            parameters, products = role_parameters(result, snapshot, role)
            if parameters is None:
                continue
            alignment.preflight.assign(model, parameters)
            for task in source_config["cases"]:
                cached_path = SOURCE / "tasks" / f'task_{task["index"]:02d}.pt'
                cached = torch.load(cached_path, map_location="cpu", weights_only=False)
                episode = alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
                cpu_episode = alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
                for repeat in range(REPEATS):
                    write_json(output / "status.json", {"state": "checking_finite_steps", "checkpoint": name, "role": role,
                               "task": task["name"], "repeat": repeat, "pid": os.getpid(), "updated_unix": time.time()})
                    torch.manual_seed(task["gradient_seed"])
                    with torch.no_grad(), compute_context(episode.source.device):
                        prediction = streamed_point_predictions(model, criterion, episode.source, episode.mask,
                                                                  **source_config["streaming"])
                    records = point_metric_terms(prediction, episode)
                    metrics = alignment.metric_summary(prediction, episode, records)
                    cpu_prediction = prediction.detach().cpu()
                    cpu_records = point_metric_terms(cpu_prediction, cpu_episode)
                    cpu_metrics = alignment.metric_summary(cpu_prediction, cpu_episode, cpu_records)
                    alignment.compare_metric_records(metrics, cpu_metrics)
                    path = directory / f'{role}_task_{task["index"]:02d}_repeat_{repeat}.npz'
                    alignment.storage_guard(output)
                    atomic_npz(path, prediction=cpu_prediction.numpy())
                    prediction_hashes[str(path)] = sha256(path)
                    row = {"role": role, "task": task["name"], "repeat": repeat, "metrics": metrics, "products": products,
                           "prediction": str(path), "prediction_sha256": prediction_hashes[str(path)]}
                    rows.append(row)
                    append_json(directory / "cases.jsonl", row)
                    del prediction, cpu_prediction, records, cpu_records
                del cached, episode, cpu_episode
            print(json.dumps({"checkpoint": name, "completed_role": role, "forward_passes": len(rows)}), flush=True)
        alignment.preflight.assign(model, snapshot["before"])
        state_results[name] = summarize_state(result, rows)
        write_json(directory / "summary.json", state_results[name])
        del snapshot, parameters
        gc.collect()
        torch.cuda.empty_cache()
    verify_hashes(prediction_hashes)
    verify_registration(output)
    write_json(output / "result.json", {"complete": True, "states": state_results, "prediction_sha256": prediction_hashes,
               "all_finite_predictions_cpu_rescored": True, "training_restarted": False, "trained_checkpoint_created": False,
               "validation_or_confirmation_used": False, "real_benchmarks_used": False, "goal_achieved": False})
    lines = ["# Actual finite-step synthetic point-metric changes", "", POLICY, "",
             "Counts retain every defined task/metric slot, averaged over two repeated forwards; undefined slots stay explicit. Lower is better.", "",
             "| Fixed state | Proposal | Better | Worse | Equal |", "|---|---|---:|---:|---:|"]
    for name, state in state_results.items():
        for role, counts in state["roles"].items():
            if role != "before" and counts["available"]:
                lines.append(f'| {name} | {role} | {counts["better"]} | {counts["worse"]} | {counts["equal"]} |')
        lines.extend(["", f'{name}: maximum within-role repeat metric spread={state["maximum_repeat_metric_spread"]:.12g}; '
                      f'maximum original-capture/before-replay metric difference={state["maximum_capture_replay_metric_difference"]:.12g}.', ""])
    lines.extend(["No sign tolerance or checkpoint-selection criterion changes. Detailed case-level signed changes remain in each summary.json.",
                  "These are training-task diagnostics, not held-out or benchmark accuracy results. No single-model success is established.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never restart a started diagnostic")
        try:
            execute(output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
