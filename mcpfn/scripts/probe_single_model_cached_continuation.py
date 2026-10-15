"""Test one cached V22 optimizer step without restarting training or publishing a model."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

import single_model_composite_gradient_storage as storage
import train_single_matrixformer_teacher_continuation as training
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once
from streamed_large_matrix_training import restore_rng


OUTPUT = training.ROOT / "teacher_continuation_v22_cached_probe_v1"
ERROR = "ValueError('Share only the exact complete original proposal-gradient bytes')"
AUDIT_SHA256 = "252be986c33a9dfbe72551241cc5d7f3192ce20e0bfd528eea9834403c1dea29"
STEP = 51
POLICY = ("A non-promotable deterministic-replay feasibility probe after the original V22 terminal serialization "
          "failure. Load its exact latest boundary50 model, optimizer and RNG, then replay only cached batch51, "
          "with the original data, cached teachers, dropout/query seeds, gradients, constant2e-5 proposal and "
          "paired protection. Require exact JSON equality of every original task loss/metric/manifest and every "
          "scientific optimizer/projection/finite-acceptance record; exclude only wall time and peak memory. "
          "Test the corrected84-row protected-gradient/28-row proposal-prefix alias without changing proposal "
          "values. No new prior generation or teacher inference, validation, confirmation, selection, model "
          "promotion or training restart. Do not save a model or gradient endpoint from this diagnostic. "
          "Hashes and matching records are feasibility evidence, not an independent full-gradient audit or "
          "proof of the unrecorded final update. Any eventual recovery still requires its complete raw endpoint "
          "and unchanged qualification. Stop on any mismatch without a tolerance search or automatic retry.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"Exact cached replay differs: {label}")


def scientific_record(record):
    return {key: value for key, value in record.items() if key not in {"seconds", "peak_gib"}}


def cached_task(directory, task, logged, config):
    path = directory / f'task_{task["index"]:02d}.pt'
    if str(path) != logged["artifact"] or sha256(path) != logged["sha256"]:
        raise ValueError("Use the exact original cached artifact, not a regenerated draw")
    cached = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    training.preflight.task_audit.check_task(cached, task, logged)
    training.preflight.check_routing(cached, task, config)
    episode = training.original.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
    teacher = cached["teacher"].to(device="cuda")
    return episode, teacher, path, logged["sha256"]


CAPTURE_SOURCE = training.replicated.CAPTURE_SOURCE
CAPTURE_SOURCE = replace_once(CAPTURE_SOURCE,
    "def capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients):",
    "def capture_cached_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients, expected_manifest, config):")
start = CAPTURE_SOURCE.index("        episode = sampler.sample(")
end = CAPTURE_SOURCE.index("        optimizer.zero_grad(", start)
CAPTURE_SOURCE = CAPTURE_SOURCE[:start] + (
    '        episode, teacher, path, digest = cached_task(directory, task, expected_manifest["tasks"][task["index"]], config)\n'
) + CAPTURE_SOURCE[end:]
CAPTURE_SOURCE = replace_once(CAPTURE_SOURCE,
    '    write_json(directory / "manifest.json", {"step": batch["step"], "tasks": records, "entries": entries})',
    '    require_same({"step": batch["step"], "tasks": records, "entries": entries}, expected_manifest, "complete batch manifest")')
CAPTURE_SOURCE_SHA256 = hashlib.sha256(CAPTURE_SOURCE.encode()).hexdigest()
_capture_namespace = {**training.replicated._capture_namespace, "cached_task": cached_task, "require_same": require_same}
exec(compile(CAPTURE_SOURCE, __file__, "exec"), _capture_namespace)
capture_cached_batch = _capture_namespace["capture_cached_batch"]


def require_original_failure():
    failed = read(training.OUTPUT / "status.json")
    if (failed["state"] != "failed" or failed["error"] != ERROR
            or Path(f'/proc/{failed["pid"]}').exists()
            or read(training.ROOT / "teacher_continuation_v22_pipeline/status.json")["state"] != "registered"):
        raise ValueError("Require the stopped original terminal serialization failure and unused qualification")
    for name in ("raw_final_update.pt", "raw_final_update.write.json", "validation_64.json", "best_validation_checkpoint.pth"):
        if (training.OUTPUT / name).exists():
            raise ValueError("Never overwrite or infer a missing original terminal result")
    rows = [json.loads(line) for line in (training.OUTPUT / "training.jsonl").read_text().splitlines()]
    if ([row["step"] for row in rows] != list(range(1, 64)) or rows[-1]["accepted_updates"] != 58
            or [row["step"] for row in rows if not row["accepted"]] != [15, 27, 29, 56, 63]):
        raise ValueError("Preserve all63 recorded attempts and their original acceptance decisions")
    audit_path = training.ROOT / "teacher_continuation_v22_audit_050/audit.json"
    if sha256(audit_path) != AUDIT_SHA256:
        raise ValueError("Preserve the exact complete boundary50 prefix audit")
    return rows


def register():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite this one-step cached replay probe")
    config = training.verify_registration(training.OUTPUT)
    rows = require_original_failure()
    directory = training.OUTPUT / "training_batches/step_051"
    manifest = read(directory / "manifest.json")
    paths = [training.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "training.jsonl",
        "latest_checkpoint.pth", "selected_checkpoint.pth", "validation_50.json", "parameter_inventory.json", "decoder_inventory.json")]
    paths.extend([directory / "manifest.json", training.ROOT / "teacher_continuation_v22_audit_050/audit.json",
                  training.ROOT / "teacher_continuation_v22_pipeline/status.json"])
    hashes = {str(path): sha256(path) for path in paths}
    hashes.update({row["artifact"]: row["sha256"] for row in manifest["tasks"]})
    verify_hashes(hashes)
    sources = {Path(__file__).resolve(), Path(storage.__file__).resolve(),
               REPO / "tests/test_single_model_cached_continuation_probe.py",
               REPO / "tests/test_single_model_composite_gradient_storage.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "original_run": str(training.OUTPUT), "step": STEP,
        "batch": config["batches"][STEP - 1], "reference_record": rows[STEP - 1], "input_sha256": hashes,
        "source_sha256": {**config["source_sha256"], **{str(path): sha256(path) for path in sources}},
        "capture_source_sha256": CAPTURE_SOURCE_SHA256, "storage_policy": storage.POLICY,
        "raw_endpoint_preserved": False, "diagnostic_only": True})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
        "new_training_cases": 0, "model_selection_performed": False, "independent_confirmation_consumed": False})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (sha256(OUTPUT / "config.json") != read(OUTPUT / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["step"] != STEP
            or config["capture_source_sha256"] != CAPTURE_SOURCE_SHA256 or config["storage_policy"] != storage.POLICY):
        raise ValueError("Preserve the exact declared cached-replay diagnostic")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    require_original_failure()
    return config


def run():
    config = verify()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a cached replay probe")
    original_config = training.verify_registration(training.OUTPUT)
    training.wait_for_gpu(OUTPUT, True, False)
    status("loading_boundary50")
    saved = torch.load(training.OUTPUT / "latest_checkpoint.pth", map_location="cpu", weights_only=False, mmap=True)
    selected = torch.load(training.OUTPUT / "selected_checkpoint.pth", map_location="cpu", weights_only=False, mmap=True)
    if (saved["step"] != 50 or saved["gradient_updates_performed"] != 47 or saved["best_step"] != 50
            or saved["config_sha256"] != sha256(training.OUTPUT / "config.json")
            or sha256(training.OUTPUT / "selected_checkpoint.pth") != read(training.OUTPUT / "validation_50.json")["selected_checkpoint_sha256"]
            or saved["model"].keys() != selected["model"].keys()
            or any(not storage.byte_equal(tensor, selected["model"][name]) for name, tensor in saved["model"].items())
            or {int(state["step"].item()) for state in saved["optimizer"]["state"].values()} != {47}):
        raise ValueError("Require the actual saved boundary50 state and47-step optimizer")
    directory = training.OUTPUT / "training_batches/step_051"
    manifest = read(directory / "manifest.json")
    started = time.monotonic()
    with training.teacher_policy.runtime(original_config["base"], OUTPUT, original_config["teacher_identities"]):
        model, view, optimizer, criterion, teachers, sampler, parent_state = training.setup(original_config)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        gradients = torch.empty((84, sum(parameter.numel() for parameter in view.parameters())), device="cuda", dtype=torch.float32)
        with training.decoder.decoder_capture_coordinates(model, view):
            entries, tasks, before = capture_cached_batch(OUTPUT, directory, config["batch"], model, optimizer,
                criterion, teachers, sampler, gradients, manifest, original_config)
        training.paired.validate_layout(entries, config["batch"]["tasks"], before)
        status("replaying_original_step51")
        record, snapshot = training.paired.protected_update(view, optimizer, gradients[:len(entries)],
            [task["weight"] for task in config["batch"]["tasks"]], training.constant_rate(STEP), before,
            lambda current: training.original.evaluate_training(model, criterion, tasks))
        record.update(step=STEP, accepted_updates=47 + int(record["accepted"]), task_manifest=str(directory / "manifest.json"),
                      task_manifest_sha256=sha256(directory / "manifest.json"), seconds=time.monotonic() - started,
                      peak_gib=torch.cuda.max_memory_allocated() / 1024**3)
        write_json(OUTPUT / "actual_record.json", record)
        require_same(scientific_record(record), scientific_record(config["reference_record"]), "complete scientific optimizer record")
        complete, shared = storage.share(gradients[:len(entries)], snapshot, record["composite_tasks"])
        storage.check_alias({"gradients": complete, "snapshot": shared, "record": record})
        frozen = training.decoder.frozen_state_equal(saved["model"], model.state_dict(), {name for name, parameter in view.named_parameters()})
        result = {"complete": True, "exact_cached_manifest_replayed": True, "exact_scientific_record_replayed": True,
            "composite_rows": shared["task_gradients"].shape[0], "protected_rows": complete.shape[0],
            "raw_gradient_sha256": hashlib.sha256(memoryview(complete.numpy()).cast("B")).hexdigest(),
            "proposal_gradient_sha256": hashlib.sha256(memoryview(shared["task_gradients"].numpy()).cast("B")).hexdigest(),
            "original_proposal_shape_and_bytes_preserved": True, "frozen": frozen,
            "accepted": record["accepted"], "original_failure_preserved": True, "training_restarted": False,
            "checkpoint_promoted": False, "raw_endpoint_saved": False, "independent_full_gradient_audit": False,
            "confirmation_consumed": False, "real_benchmark_evaluated": False}
    verify()
    write_json(OUTPUT / "result.json", result)
    status("complete", accepted=record["accepted"], diagnostic_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except BaseException as error:
                status("failed", error=repr(error))
                raise
