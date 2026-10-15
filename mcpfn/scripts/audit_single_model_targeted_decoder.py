"""CPU audit of V15 targeted decoder training, original V8 guards and the restricted endpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from audit_single_matrixformer_structured_training import close_values
from audit_single_model_finite_guarded_training import check_task, read_prefix
import single_model_decoder_training as decoder
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import single_model_finite_guarded_audit as checks
import train_single_matrixformer_targeted_decoder as training


def check_inventories(parameters, selected, expected):
    if parameters != expected or not selected or selected != [row for row in parameters if row[0].startswith("decoder.")]:
        raise ValueError("Require the original full model and exactly its six decoder tensors")
    if len(selected) != 6 or sum(row[2] for row in selected) != 11227016:
        raise ValueError("The decoder coordinate count changed")


def check_initialization(report, parent):
    close_values(report["student"], parent["student"])
    if len(report["cases"]) != 152 or len(parent["cases"]) != 152:
        raise ValueError("All152 V14 initialization grids are mandatory")
    for current, previous in zip(report["cases"], parent["cases"]):
        if {key: value for key, value in current.items() if key != "prediction_file"} != {key: value for key, value in previous.items() if key != "prediction_file"}:
            raise ValueError("The V15 initialization must reproduce the V14 final prediction grids")


def check_checkpoint(saved, parent, selected, expected_step, accepted, best, config, config_digest, bank_digest):
    frozen = decoder.frozen_state_equal(parent, saved["model"], {row[0] for row in selected})
    if (saved["step"] != expected_step or saved["epoch"] != expected_step or saved["attempted_batches"] != expected_step
            or saved["gradient_updates_performed"] != accepted or saved["parent_training_updates"] != 95
            or saved["config_sha256"] != config_digest or saved["bank_sha256"] != bank_digest
            or saved["independent_confirmation_passed"] or saved["preflight_only"] or not saved["decoder_only_training"]
            or saved["parent_checkpoint_sha256"] != config["input_sha256"][str(training.PARENT)]
            or saved["retention_baseline_sha256"] != config["input_sha256"][str(training.original.alignment.fixed.PARENT)]
            or saved["best_step"] != best["step"] or saved["best_score"] != best["score"]):
        raise ValueError("Decoder checkpoint provenance changed")
    return frozen


def audit(output, through_step):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Training audit must be CPU only")
    if output.exists() or through_step not in training.BOUNDARIES[1:]:
        raise ValueError("Require a new output and a complete scheduled boundary")
    run = training.OUTPUT
    config = training.verify_registration(run)
    if not (run / f"validation_{through_step}.json").exists():
        raise ValueError("The requested scheduled validation is not yet complete")
    terminal = training.read(run / "status.json")
    complete_trial = through_step == training.STEPS
    if complete_trial and (terminal["state"] != "training_complete" or terminal["step"] != training.STEPS):
        raise ValueError("Never replace the full hundred-attempt qualification audit with a partial trial")
    rows, raw = read_prefix(run / "training.jsonl", range(1, through_step + 1))
    boundaries = [step for step in training.BOUNDARIES if step <= through_step]
    trace, validation_raw = read_prefix(run / "validation.jsonl", boundaries)
    parameters = training.read(run / "parameter_inventory.json")
    selected = training.read(run / "decoder_inventory.json")
    check_inventories(parameters, selected, training.read(training.ROOT / "domain_gradient_probe_v1/parameter_inventory.json"))
    hashes = {str(run / name): sha256(run / name) for name in ("config.json", "registration.json", "parameter_inventory.json", "decoder_inventory.json")}
    accepted, previous_rate = 0, config["base"]["peak_lr"]
    accepted_by_step = {0: 0}
    rollback_rates = {}
    for batch, row in zip(config["batches"][:through_step], rows):
        step = batch["step"]
        manifest_path = run / "training_batches" / f"step_{step:03d}" / "manifest.json"
        if row["task_manifest"] != str(manifest_path):
            raise ValueError("Training task manifest location changed")
        hashes[str(manifest_path)] = row["task_manifest_sha256"]
        verify_hashes({str(manifest_path): row["task_manifest_sha256"]})
        manifest = training.read(manifest_path)
        if manifest["step"] != step or len(manifest["tasks"]) != 14:
            raise ValueError("Every registered fourteen-task batch is mandatory")
        point_before = {}
        entries = [{"task": task["name"], "kind": "composite", "index": task["index"]} for task in batch["tasks"]]
        for task, logged in zip(batch["tasks"], manifest["tasks"]):
            path = manifest_path.parent / f'task_{task["index"]:02d}.pt'
            if logged["artifact"] != str(path):
                raise ValueError("Cached task location changed")
            hashes[str(path)] = logged["sha256"]
            verify_hashes({str(path): logged["sha256"]})
            payload = torch.load(path, map_location="cpu", weights_only=False)
            point_before.update(check_task(payload, task, logged))
            for name, metric in logged["metrics_before"].items():
                if metric is not None:
                    entries.append({"task": task["name"], "kind": name, "index": len(entries)})
            del payload
        if entries != manifest["entries"]:
            raise ValueError("Composite and defined-point gradient order changed")
        checks.check_record(row, [task["weight"] for task in batch["tasks"]], point_before, len(entries))
        if row["rate"] != training.learning_rate(step, 100, 1e-5, 5):
            raise ValueError("The registered first-fifty original learning rates changed")
        rollback_rates[step] = previous_rate
        accepted += int(row["accepted"])
        if row["accepted_updates"] != accepted:
            raise ValueError("Attempted batches must never masquerade as accepted optimizer updates")
        accepted_by_step[step] = accepted
        if row["accepted"]:
            previous_rate = row["rate"]
    producer = training.original.alignment.fixed.PRODUCER
    reference, initial = training.read(producer / "reference.json"), training.read(producer / "validation_0.json")
    bank = torch.load(producer / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    if len(bank) != 152 or len(reference["cases"]) != 152 or len(initial["cases"]) != 152:
        raise ValueError("All152 original development cases and references are required")
    for cached, specification in zip(bank, config["base"]["validation_cases"]):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Original development case order or identity changed")
    for role in ("general", "teacher"):
        if training.original.alignment.mixture.aggregate(reference["cases"], role) != reference[role]:
            raise ValueError("Frozen reference aggregation changed")
    if training.original.alignment.fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Frozen minimax reference floors changed")
    initial_score = training.original.alignment.fixed.selection_decision(initial["student"], initial["student"], reference["references"], config["base"])["initialization_score"]
    best = {"step": 0, "score": initial_score}
    parent_report_path = training.INITIAL_RUN / "validation_50.json"
    parent_report = training.read(parent_report_path)
    hashes[str(parent_report_path)] = sha256(parent_report_path)
    fields, reports = [(reference, ("general", "teacher")), (initial, ("student",)), (parent_report, ("student",))], []
    for step, logged in zip(boundaries, trace):
        path = run / f"validation_{step}.json"
        report = training.read(path)
        hashes[str(path)] = sha256(path)
        if (report["step"] != step or report["accepted_updates"] != accepted_by_step[step] or len(report["cases"]) != 152
                or training.original.alignment.mixture.aggregate(report["cases"], "student") != report["student"]
                or logged != {key: value for key, value in report.items() if key != "cases"}):
            raise ValueError("Original validation coverage, aggregates or trace changed")
        if step == 0:
            check_initialization(report, parent_report)
        decision = training.original.alignment.fixed.selection_decision(report["student"], initial["student"], reference["references"], config["base"])
        if decision != report["decision"] or len(decision["comparisons"]) != 22 or decision["research_selection_tolerance"] != 0:
            raise ValueError("All22 zero-tolerance retention guards and strict minimax selection are mandatory")
        old_best = best
        best = training.selected_boundary(best, step, accepted_by_step[step], decision)
        frozen = report["frozen"]
        if (frozen["frozen_tensors"] != 248 or not frozen["backbone_and_buffers_bitwise_unchanged"]
                or not set(frozen["changed_decoder_tensors"]).issubset({row[0] for row in selected})
                or (step == 0 and frozen["changed_decoder_tensors"])):
            raise ValueError("Boundary must preserve every nondecoder parameter and buffer")
        if report["best_step"] != best["step"] or bool(report["selected_checkpoint_sha256"]) != (best != old_best):
            raise ValueError("Trained-only selection or earlier-boundary tie rule changed")
        reports.append(report)
        fields.append((report, ("student",)))
    count, prediction_hashes = training.original.alignment.fixed.raw_audit(bank, fields)
    if count != 152 * (4 + len(boundaries)):
        raise ValueError("Every full saved reference, initialization and scheduled prediction grid is mandatory")
    hashes.update(prediction_hashes)
    endpoint_result = None
    if complete_trial:
        if terminal["accepted_updates"] != accepted or terminal["best"] != best:
            raise ValueError("Terminal update counts or selection changed")
        if (run / "best_validation_checkpoint.pth").exists():
            raise ValueError("Never publish a best alias before independent qualification")
        parent = torch.load(training.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
        paths = [run / "latest_checkpoint.pth"]
        if best["step"]:
            paths.append(run / "selected_checkpoint.pth")
        elif (run / "selected_checkpoint.pth").exists():
            raise ValueError("No unqualified trained checkpoint may be selected")
        for path in paths:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            expected_step = best["step"] if path.name == "selected_checkpoint.pth" else training.STEPS
            details = check_checkpoint(saved, parent, selected, expected_step, accepted_by_step[expected_step], best,
                                       config, sha256(run / "config.json"), reference["bank_sha256"])
            expected_report = next(row for row in reports if row["step"] == expected_step)
            if details != expected_report["frozen"]:
                raise ValueError("Saved full model must reproduce the boundary frozen-state certificate")
            hashes[str(path)] = sha256(path)
            if path.name == "selected_checkpoint.pth":
                report = next(row for row in reports if row["step"] == best["step"])
                if hashes[str(path)] != report["selected_checkpoint_sha256"] or not details["changed_decoder_tensors"]:
                    raise ValueError("Selected weights must bind the actual audited trained boundary")
            else:
                latest = saved
        raw_path = run / "raw_final_update.pt"
        hashes[str(raw_path)] = rows[-1]["raw_endpoint_sha256"]
        verify_hashes({str(raw_path): hashes[str(raw_path)]})
        payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)
        if (payload["config_sha256"] != sha256(run / "config.json") or payload["batch"] != config["batches"][-1]
                or payload["entries"] != training.read(rows[-1]["task_manifest"])["entries"]
                or payload["task_manifest"] != rows[-1]["task_manifest"]):
            raise ValueError("Raw endpoint must bind the final registered attempted batch")
        endpoint_result = checks.check_endpoint(payload, rows[-1], [row[2] for row in selected],
            config["base"]["weight_decay"], rollback_rates[training.STEPS], latest["optimizer"])
        expected = torch.cat([latest["model"][name].flatten() for name, shape, size in selected])
        if not torch.equal(expected, payload["snapshot"]["after"]):
            raise ValueError("Latest model must equal the actual accepted or rolled-back endpoint")
        hashes.update({str(run / name): sha256(run / name) for name in ("training.jsonl", "validation.jsonl", "status.json")})
    verify_hashes(hashes)
    training.verify_registration(run)
    output.mkdir(parents=True)
    (output / "training_prefix.jsonl").write_text(raw)
    (output / "validation_prefix.jsonl").write_text(validation_raw)
    write_json(output / "audit.json", {"complete": True, "through_step": through_step, "complete_training_audit": complete_trial,
               "attempted_batches": through_step, "accepted_updates": accepted, "training_episodes": 14 * through_step,
               "decoder_only_training": True, "retention_baseline": config["retention_baseline"],
               "training_initialization": config["parent_checkpoint"], "best": best, "prediction_grids": count, "raw_endpoint": endpoint_result, "artifact_sha256": hashes,
               "training_prefix_sha256": hashlib.sha256(raw.encode()).hexdigest(),
               "validation_prefix_sha256": hashlib.sha256(validation_raw.encode()).hexdigest(),
               "cuda_initialized": False, "autograd_replayed": False, "full_optimizer_trajectory_replayed": False,
               "research_selection_tolerance": 0, "confirmation_evaluated": False, "goal_success_not_assessed": True,
               "source_sha256": {str(Path(__file__).resolve()): sha256(__file__), str(Path(checks.__file__).resolve()): sha256(checks.__file__)}})
    lines = ["# Decoder-only finite-guarded training audit", "", training.POLICY, "",
             f"Audited{through_step} attempted batches,{accepted} accepted updates, all{14 * through_step} cached tasks and{count} full prediction grids.",
             "", "| Boundary | Accepted updates | Extended minimax | Failed guards | Selected boundary |",
             "|---|---:|---:|---:|---:|"]
    for report in reports:
        lines.append(f'| {report["step"]} | {report["accepted_updates"]} | {report["decision"]["score"]:.10f} | '
                     f'{sum(not item["retained"] for item in report["decision"]["comparisons"])} | {report["best_step"]} |')
    lines.extend(["", "Earlier direction certificates are logged evidence, not full-gradient replays. The full audit remeasures only the declared final raw endpoint.",
                  "Neural endpoint replay, independent confirmation and the full frozen benchmark review remain mandatory before any success claim.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-step", type=int, default=training.STEPS)
    arguments = parser.parse_args()
    audit(arguments.output.resolve(), arguments.through_step)
