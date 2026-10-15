"""Independently rescore completed V12 synthetic boundaries without changing training."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

import audit_single_model_family_projection_training as audit
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json


training = audit.training


def read_prefix(path, through_step):
    rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
    selected = [row for row in rows if row["step"] <= through_step]
    if [row["step"] for row in selected] != list(range(1, through_step + 1)):
        raise ValueError("Require every complete optimizer update in the requested prefix")
    return selected


def decisions_table(decisions):
    names = list(training.ARMS)
    control = decisions[names[0]]["comparisons"]
    other = decisions[names[1]]["comparisons"]
    if len(control) != 22 or len(other) != 22:
        raise ValueError("Never omit one of the original twenty-two retention guards")
    result = []
    for left, right in zip(control, other):
        if (left["group"], left["metric"], left["initialization"]) != (right["group"], right["metric"], right["initialization"]):
            raise ValueError("Compare identical metrics to the identical V8 initialization")
        row = {"group": left["group"], "metric": left["metric"], "initialization": left["initialization"]}
        for name, record in zip(names, (left, right)):
            if record["retained"] != (record["candidate"] <= record["initialization"]):
                raise ValueError("Retention must have zero tolerance")
            row[name] = {"value": record["candidate"], "retained": record["retained"],
                         "change_percent": 100 * (record["candidate"] / record["initialization"] - 1) if record["initialization"] else None}
        result.append(row)
    return result


def summarize(through_step, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Boundary rescoring must be independently CPU-only")
    if through_step not in range(training.EVERY, training.STEPS + 1, training.EVERY):
        raise ValueError("Only a registered completed validation boundary can be reported")
    config = training.verify_registration(training.PAIR)
    initial_path = training.fixed.PRODUCER / "validation_0.json"
    reference_path = training.fixed.PRODUCER / "reference.json"
    bank_path = training.fixed.PRODUCER / "synthetic_bank.pt"
    reference, original = training.read(reference_path), training.read(initial_path)
    baseline = original["student"]
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    expected = config["base"]["validation_cases"]
    if len(bank) != len(expected) or len(bank) != 152:
        raise ValueError("Preserve every original development case")
    verify_hashes({str(bank_path): reference["bank_sha256"]})
    for cached, specification in zip(bank, expected):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Original synthetic case order or identity changed")
    sources = {str(Path(module.__file__).resolve()): sha256(Path(module.__file__)) for module in (audit, training, training.fixed, training.mixture)}
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    paths = [bank_path, reference_path, initial_path, training.PAIR / "config.json"]
    paths += [arm / f"validation_{step}.json" for arm in training.ARMS.values() for step in range(0, through_step + 1, training.EVERY)]
    inputs = {str(path): sha256(path) for path in paths}
    prefixes = {name: read_prefix(arm / "training.jsonl", through_step) for name, arm in training.ARMS.items()}
    write_json(output / "training_prefix.json", prefixes)
    inputs[str(output / "training_prefix.json")] = sha256(output / "training_prefix.json")
    write_json(output / "config.json", {"through_step": through_step, "input_sha256": inputs, "source_sha256": sources,
        "scope": "CPU rescoring of full saved development prediction grids and logged update certificates; no new forward/backward, model selection, confirmation or benchmark run",
        "no_training_or_registered_source_changes": True})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    for report, roles in ((reference, ("general", "teacher")), (original, ("student",))):
        if len(report["cases"]) != len(bank):
            raise ValueError("Reference coverage changed")
        for role in roles:
            if training.mixture.aggregate(report["cases"], role) != report[role]:
                raise ValueError("Reference aggregates changed")
    if training.fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Original reference floors changed")
    fields = [(reference, ("general", "teacher")), (original, ("student",))]
    decisions, traces, directions = {}, {}, {}
    initial_score = training.fixed.selection_decision(baseline, baseline, reference["references"], config["base"])["initialization_score"]
    for name, arm in training.ARMS.items():
        best = {"step": 0, "score": initial_score}
        traces[name] = []
        for step in range(0, through_step + 1, training.EVERY):
            report = training.read(arm / f"validation_{step}.json")
            if report["step"] != step or len(report["cases"]) != 152:
                raise ValueError("Validation boundary or coverage changed")
            if training.mixture.aggregate(report["cases"], "student") != report["student"]:
                raise ValueError("Saved validation aggregation changed")
            if step == 0:
                audit.close_values(report["student"], baseline)
            decision = training.fixed.selection_decision(report["student"], baseline, reference["references"], config["base"])
            if report["decision"] != decision or len(decision["comparisons"]) != 22:
                raise ValueError("Original strict selection decision did not reproduce")
            best = training.selected_boundary(best, step, decision)
            if report["best_step"] != best["step"]:
                raise ValueError("Strict minimax selection or earlier-boundary tie rule changed")
            traces[name].append({"step": step, "decision": decision, "best": dict(best)})
            fields.append((report, ("student",)))
        decisions[name] = decision
        for row, batch in zip(prefixes[name], config["batches"]):
            if row["lr"] != training.learning_rate(batch["step"], training.STEPS, config["peak_lr"], config["warmup_steps"]):
                raise ValueError("Original learning-rate schedule changed")
            audit.check_record(row, [task["weight"] for task in batch["tasks"]], projected=name == "projected")
        directions[name] = {"logged_updates_checked": through_step,
            "updates_with_adverse_tasks_before": sum(row["adverse_tasks_before"] > 0 for row in prefixes[name]),
            "adverse_task_events_before": sum(row["adverse_tasks_before"] for row in prefixes[name]),
            "adverse_task_events_after": sum(row["adverse_tasks_after"] for row in prefixes[name]),
            "updates_with_nonzero_projection": sum(row["projection"] is not None and any(value != 0 for value in row["projection"]["raw_gradient_coefficients"]) for row in prefixes[name]),
            "scope": "Logged certificates only; this prefix report does not independently replay all raw task gradients"}
    for control, projected in zip(prefixes["control"], prefixes["projected"]):
        if (control["task_manifest"], control["task_manifest_sha256"]) != (projected["task_manifest"], projected["task_manifest_sha256"]):
            raise ValueError("Both arms must bind the same cached tasks")
        inputs[control["task_manifest"]] = control["task_manifest_sha256"]
    count, prediction_hashes = training.fixed.raw_audit(bank, fields)
    if count != 152 * sum(len(roles) for report, roles in fields):
        raise ValueError("Require all saved full prediction grids")
    inputs.update(prediction_hashes)
    if prefixes != {name: read_prefix(arm / "training.jsonl", through_step) for name, arm in training.ARMS.items()}:
        raise ValueError("Completed training prefix changed during audit")
    training.verify_registration(training.PAIR)
    verify_hashes(inputs)
    verify_hashes(sources)
    table = decisions_table(decisions)
    result = {"complete": True, "through_step": through_step, "synthetic_cases": 152, "prediction_grids_rescored": count,
        "selection_traces": traces, "decisions": decisions, "guards": table, "logged_directions": directions,
        "artifact_sha256": inputs, "source_sha256": sources, "research_selection_tolerance": 0,
        "cuda_initialized": torch.cuda.is_initialized(), "training_changed": False, "confirmation_consumed": False,
        "real_benchmarks_evaluated": False, "goal_success_not_assessed": True,
        "paired_caveat": "Same registered data and seeds, but backward vectors are not bitwise reproducible; between-arm differences alone do not isolate projection effects"}
    write_json(output / "audit.json", result)
    lines = [f"# V12 synthetic development boundary: update {through_step}", "",
        "These are synthetic development scores, not real MissBench, UCI, recommender or LLM benchmark results.",
        "All original zero-tolerance guards remain unchanged. The report does not select a different checkpoint or change the running trial.", "",
        "| Candidate | Extended minimax (lower is better) | Failed guards /22 | Eligible at this boundary |",
        "|---|---:|---:|:---:|", f"| Initial selected V8 | {initial_score:.10f} | 0 | Initialization only |"]
    for name, decision in decisions.items():
        lines.append(f'| {name} | {decision["score"]:.10f} | {sum(not row["retained"] for row in decision["comparisons"])} | {decision["passed"]} |')
    lines += ["", "## All retention guards", "", "| Group | Metric | Initial V8 | Control | Projected | Control change | Projected change |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        changes = {name: "undefined" if row[name]["change_percent"] is None else f'{row[name]["change_percent"]:+.6f}%' for name in training.ARMS}
        lines.append(f'| {row["group"]} | {row["metric"]} | {row["initialization"]:.10f} | {row["control"]["value"]:.10f} | {row["projected"]["value"]:.10f} | {changes["control"]} | {changes["projected"]} |')
    lines += ["", f"The independent CPU audit rescores all {count} full prediction grids and checks the logged certificates for {2 * through_step} updates.",
        "No additional autograd or fused-CUDA trajectory replay is claimed. Logged favorable first-order directions do not guarantee finite-step losses or held-out improvements.",
        result["paired_caveat"] + ".", "No real-benchmark parity or goal completion is established.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"through_step": through_step, "prediction_grids_rescored": count,
        "arms": {name: {"score": decision["score"], "passed": decision["passed"], "failed_guards": sum(not row["retained"] for row in decision["comparisons"])} for name, decision in decisions.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=range(training.EVERY, training.STEPS + 1, training.EVERY), required=True)
    args = parser.parse_args()
    output = training.ROOT / f"family_projection_v12_boundary_{args.step:03d}_audit"
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "status.json", {"state": "auditing", "pid": os.getpid(), "updated_unix": time.time()})
        summarize(args.step, output)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
