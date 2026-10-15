"""Reconstruct V22's cached tail once, preserving its failed original and scientific recipe."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

import probe_single_model_cached_continuation as probe
import single_model_composite_gradient_storage as storage
import single_model_shared_tensor_storage as shared_storage
import train_single_matrixformer_teacher_continuation as original_training
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from streamed_large_matrix_training import restore_rng


ROOT = original_training.ROOT
OUTPUT = ROOT / "teacher_continuation_v22_cached_recovery_v1"
AUDIT = ROOT / "teacher_continuation_v22_cached_recovery_v1_audit"
START, END = 50, 64
WRITE_BUDGET = 6 * 1024**3
POLICY_RECOVERY = ("One bounded reconstruction from the exact saved V22 boundary50 model,47-step optimizer and RNG. "
    "Replay cached51-64 without new prior generation or teacher inference. Require exact original cached "
    "manifest equality for all14 batches and exact scientific optimizer record equality for51-63, excluding "
    "only seconds and peak_gib. Batch64 had no original logged outcome: mark its result as reconstructed, "
    "never independently matched. Preserve all84 terminal protection gradients with the unchanged28-row "
    "composite proposal prefix and save the full endpoint. Correct serialization only. Reuse the original "
    "immutable0/10/25/50 validation evidence, then evaluate64 and apply the unchanged selection and original "
    "V8 retention baseline. Preserve the failed run, all original inputs and the unused original pipeline. "
    "No restart, automatic retry, equality relaxation, new draws, recipe changes, confirmation, promotion or "
    "real benchmarks in this reconstruction. Independent full training, teacher, endpoint, selected-boundary "
    "and fresh confirmation audits remain mandatory. Earlier full gradients were not saved; matching cached "
    "manifests and scalar records is not an independent full-gradient trajectory audit.")


def __getattr__(name):
    return getattr(original_training, name)


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def original_manifest(step):
    if type(step) is not int or not 1 <= step <= END:
        raise ValueError("Use a registered original batch")
    return original_training.OUTPUT / "training_batches" / f"step_{step:03d}" / "manifest.json"


def require_probe(report):
    required = ("complete", "exact_cached_manifest_replayed", "exact_scientific_record_replayed",
                "original_proposal_shape_and_bytes_preserved", "original_failure_preserved")
    forbidden = ("training_restarted", "checkpoint_promoted", "raw_endpoint_saved", "independent_full_gradient_audit",
                 "confirmation_consumed", "real_benchmark_evaluated")
    if (any(report.get(key) is not True for key in required) or any(report.get(key) is not False for key in forbidden)
            or report.get("composite_rows") != 28 or report.get("protected_rows") != 84):
        raise ValueError("Require the exact successful non-promotable one-step diagnostic")


def check_saved(saved, selected, config_digest, selected_digest, validation):
    if (saved["step"] != START or saved["gradient_updates_performed"] != 47 or saved["best_step"] != START
            or saved["config_sha256"] != config_digest or selected_digest != validation["selected_checkpoint_sha256"]
            or saved["model"].keys() != selected["model"].keys()
            or any(not storage.byte_equal(value, selected["model"][name]) for name, value in saved["model"].items())
            or not saved["optimizer"]["state"]
            or {int(value["step"].item()) for value in saved["optimizer"]["state"].values()} != {47}):
        raise ValueError("Require the actual saved boundary50 and47-update optimizer")


def compare_record(record, reference):
    if record["step"] != reference["step"] or not START < record["step"] < END:
        raise ValueError("Only original51-63 outcomes exist for exact record comparison")
    probe.require_same(probe.scientific_record(record), probe.scientific_record(reference),
                       f'complete scientific optimizer record{record["step"]}')


def register():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("Never overwrite or retry the bounded cached reconstruction")
    original_training.verify_registration(original_training.OUTPUT)
    probe.verify()
    require_probe(read(probe.OUTPUT / "result.json"))
    if read(probe.OUTPUT / "status.json")["state"] != "complete":
        raise ValueError("The feasibility probe must be finished")
    if shutil.disk_usage(ROOT).free - WRITE_BUDGET < shared_storage.MINIMUM_FREE:
        raise OSError("Reserve6 GiB for reconstruction while retaining the18 GiB write floor")
    paths = [original_training.OUTPUT / name for name in ("config.json", "registration.json", "status.json",
        "training.jsonl", "validation.jsonl", "latest_checkpoint.pth", "selected_checkpoint.pth",
        "parameter_inventory.json", "decoder_inventory.json")]
    paths.extend(original_training.OUTPUT / f"validation_{step}.json" for step in (0, 10, 25, 50))
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "result.json"))
    hashes = {str(path): sha256(path) for path in paths}
    for step in range(1, END + 1):
        path = original_manifest(step)
        hashes[str(path)] = sha256(path)
        hashes.update({row["artifact"]: row["sha256"] for row in read(path)["tasks"]})
    verify_hashes(hashes)
    sources = {Path(__file__).resolve(), REPO / "scripts/audit_single_model_cached_continuation.py",
               REPO / "tests/test_single_model_cached_continuation_recovery.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    OUTPUT.mkdir()
    write_json(OUTPUT / "recovery_config.json", {"policy": POLICY_RECOVERY, "original_run": str(original_training.OUTPUT),
        "start": START, "end": END, "exact_record_steps": list(range(START + 1, END)),
        "cached_manifest_steps": list(range(START + 1, END + 1)), "terminal_original_outcome_available": False,
        "write_budget_bytes": WRITE_BUDGET, "capture_source_sha256": probe.CAPTURE_SOURCE_SHA256,
        "storage_policy": storage.POLICY, "input_sha256": hashes,
        "source_sha256": {str(path): sha256(path) for path in sources}})
    write_json(OUTPUT / "recovery_registration.json", {"config_sha256": sha256(OUTPUT / "recovery_config.json"),
        "registered_unix": time.time(), "new_training_cases": 0, "confirmation_consumed": False})
    status("registered")


def verify_recovery():
    config = read(OUTPUT / "recovery_config.json")
    if (sha256(OUTPUT / "recovery_config.json") != read(OUTPUT / "recovery_registration.json")["config_sha256"]
            or config["policy"] != POLICY_RECOVERY or config["original_run"] != str(original_training.OUTPUT)
            or config["start"] != START or config["end"] != END
            or config["exact_record_steps"] != list(range(START + 1, END))
            or config["cached_manifest_steps"] != list(range(START + 1, END + 1))
            or config["terminal_original_outcome_available"] is not False
            or config["write_budget_bytes"] != WRITE_BUDGET or config["storage_policy"] != storage.POLICY
            or config["capture_source_sha256"] != probe.CAPTURE_SOURCE_SHA256):
        raise ValueError("Preserve the registered bounded reconstruction and exact comparison rules")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    probe.require_original_failure()
    original_training.verify_registration(original_training.OUTPUT)
    return config


def verify_registration(output):
    if output != OUTPUT:
        raise ValueError("Use only the registered recovery output")
    verify_recovery()
    if (sha256(output / "config.json") != sha256(original_training.OUTPUT / "config.json")
            or sha256(output / "registration.json") != sha256(original_training.OUTPUT / "registration.json")):
        raise ValueError("The scientific training registration must remain byte-identical")
    return original_training.verify_registration(output)


def seed_output():
    names = ["config.json", "registration.json", "parameter_inventory.json", "decoder_inventory.json",
             "validation.jsonl", "selected_checkpoint.pth"]
    names.extend(f"validation_{step}.json" for step in (0, 10, 25, 50))
    for name in names:
        destination = OUTPUT / name
        if destination.exists():
            raise FileExistsError("Never overwrite a started recovery")
        shutil.copyfile(original_training.OUTPUT / name, destination)
        if sha256(destination) != sha256(original_training.OUTPUT / name):
            raise ValueError("Preserve exact prefix bytes")
    raw = (original_training.OUTPUT / "training.jsonl").read_text().splitlines(keepends=True)
    if len(raw) != END - 1:
        raise ValueError("Require all63 original logged attempts")
    with (OUTPUT / "training.jsonl").open("x") as stream:
        stream.write("".join(raw[:START]))


def save_endpoint(gradients, snapshot, record, entries, batch, manifest):
    complete, shared = storage.share(gradients, snapshot, record["composite_tasks"])
    payload = {"snapshot": shared, "record": dict(record), "gradients": complete, "entries": entries,
               "batch": batch, "task_manifest": str(manifest), "config_sha256": sha256(OUTPUT / "config.json")}
    storage.check_alias(payload)
    return shared_storage.save(payload, OUTPUT / "raw_final_update.pt")


def verify_replayed_rows(rows, reference):
    if ([row["step"] for row in rows] != list(range(1, END + 1))
            or [row["step"] for row in reference] != list(range(1, END))):
        raise ValueError("Require64 reconstructed attempts and63 original logged outcomes")
    probe.require_same(rows[:START], reference[:START], "all50 untouched prefix records")
    for row, expected in zip(rows[START:END - 1], reference[START:]):
        compare_record(row, expected)
    return {"exact_prefix_records": START, "exact_replayed_records": END - START - 1,
            "reconstructed_terminal_step": END, "terminal_outcome_matched_to_original": False}


def audit_recovery():
    config = verify_recovery()
    rows = [json.loads(line) for line in (OUTPUT / "training.jsonl").read_text().splitlines()]
    reference = probe.require_original_failure()
    evidence = verify_replayed_rows(rows, reference)
    report = read(OUTPUT / "recovery_result.json")
    if (report != {"complete": True, **evidence, "exact_cached_manifests": END - START,
                   "original_failure_preserved": True, "full_gradient_trajectory_audit": False,
                   "confirmation_consumed": False, "real_benchmarks_evaluated": False,
                   "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json")}
            or read(OUTPUT / "status.json")["state"] != "training_complete"):
        raise ValueError("Require complete bounded reconstruction without promotion")
    hashes = {str(OUTPUT / name): sha256(OUTPUT / name)
              for name in ("recovery_config.json", "recovery_registration.json", "recovery_result.json")}
    return {**hashes, **config["input_sha256"]}


def run():
    verify_recovery()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart this one-shot reconstruction")
    original_training.wait_for_gpu(OUTPUT, True, False)
    status("loading_boundary50")
    seed_output()
    config = verify_registration(OUTPUT)
    saved = torch.load(original_training.OUTPUT / "latest_checkpoint.pth", map_location="cpu", weights_only=False, mmap=True)
    selected = torch.load(original_training.OUTPUT / "selected_checkpoint.pth", map_location="cpu", weights_only=False, mmap=True)
    check_saved(saved, selected, sha256(OUTPUT / "config.json"), sha256(OUTPUT / "selected_checkpoint.pth"),
                read(OUTPUT / "validation_50.json"))
    originals = probe.require_original_failure()
    reference = read(original_training.original.alignment.fixed.PRODUCER / "reference.json")
    baseline = read(original_training.original.alignment.fixed.PRODUCER / "validation_0.json")["student"]
    bank = torch.load(original_training.original.alignment.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    best, accepted = {"step": saved["best_step"], "score": saved["best_score"]}, 47
    with original_training.teacher_policy.runtime(config["base"], OUTPUT, config["teacher_identities"]):
        model, view, optimizer, criterion, teachers, sampler, parent_state = original_training.setup(config)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        gradients = torch.empty((84, sum(parameter.numel() for parameter in view.parameters())), device="cuda", dtype=torch.float32)
        for batch in config["batches"][START:END]:
            step, started = batch["step"], time.monotonic()
            manifest_path = original_manifest(step)
            manifest = read(manifest_path)
            torch.cuda.reset_peak_memory_stats()
            status("reconstructing_cached_update", active_step=step, accepted_updates=accepted)
            with original_training.decoder.decoder_capture_coordinates(model, view):
                entries, tasks, before = probe.capture_cached_batch(OUTPUT, manifest_path.parent, batch, model, optimizer,
                    criterion, teachers, sampler, gradients, manifest, config)
            original_training.paired.validate_layout(entries, batch["tasks"], before)
            record, snapshot = original_training.paired.protected_update(view, optimizer, gradients[:len(entries)],
                [task["weight"] for task in batch["tasks"]], original_training.constant_rate(step), before,
                lambda current: original_training.original.evaluate_training(model, criterion, tasks))
            accepted += int(record["accepted"])
            record.update(step=step, accepted_updates=accepted, task_manifest=str(manifest_path),
                          task_manifest_sha256=sha256(manifest_path), seconds=time.monotonic() - started,
                          peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
            if step < END:
                write_json(OUTPUT / f"actual_record_{step}.json", record)
                compare_record(record, originals[step - 1])
            else:
                record["raw_endpoint_sha256"] = save_endpoint(gradients[:len(entries)], snapshot, record, entries, batch, manifest_path)
            original_training.append_json(OUTPUT / "training.jsonl", record)
            print(json.dumps({"step": step, "accepted": record["accepted"], "accepted_updates": accepted,
                              "matched_original_record": step < END, "seconds": record["seconds"]}), flush=True)
            del snapshot
        del gradients, tasks
        status("validating_boundary64", accepted_updates=accepted)
        best = original_training.validate(OUTPUT, model, optimizer, END, accepted, best, criterion, bank, reference,
                                          baseline, parent_state, config)
        original_training.checkpoint(OUTPUT, model, optimizer, END, accepted, best, reference)
    verify_registration(OUTPUT)
    rows = [json.loads(line) for line in (OUTPUT / "training.jsonl").read_text().splitlines()]
    evidence = verify_replayed_rows(rows, originals)
    write_json(OUTPUT / "recovery_result.json", {"complete": True, **evidence, "exact_cached_manifests": END - START,
        "original_failure_preserved": True, "full_gradient_trajectory_audit": False, "confirmation_consumed": False,
        "real_benchmarks_evaluated": False, "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json")})
    status("training_complete", step=END, accepted_updates=accepted, best=best, qualification_pending=True)


def interrupted(received, frame):
    raise InterruptedError(f"Reconstruction interrupted by signal{received}; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, interrupted)
            try:
                run()
            except BaseException as error:
                status("failed", error=repr(error))
                raise
