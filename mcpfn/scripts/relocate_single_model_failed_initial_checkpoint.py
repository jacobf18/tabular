"""Preserve V34's unchanged step0 checkpoint without presenting its failed run as audited."""

import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_full_network_capacity as previous
from relocate_single_model_full_network_capacity import readers, sha256, verify_hashes, write_json, sync_directory, byte_equal
import probe_single_model_full_network_resident_memory as memory
from single_model_registered_adaptation import replace_once


ROOT, REPO = previous.ROOT, previous.REPO
OUTPUT = ROOT / "failed_initial_checkpoint_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/failed_initial_checkpoint_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
OWNERS = {"full_network_v34": {"kind": "failed_run_initial_state"}}
POLICY = ("Storage-only preservation of exactly the failed V34 run's zero-update step0 checkpoint. "
    "It is not a terminal trained optimizer tail, selected model or independently audited training result. "
    "Require an empty optimizer state and exact bitwise equality of every model tensor with the original "
    "V29 selected initializer, which stays a regular file on checkpoint storage. Preserve the original "
    "failed trainer/coordinator status and all eight recorded, unaudited updates without qualification. "
    "Bind both completed memory diagnostics and the consumed task capsule. Require absent workers/readers "
    "and UUID-bound scratch; copy/fsync/hash/mmap before atomic canonical linking, then independently "
    "recheck every serialized byte and parent tensor. No scientific retry, new seed, discarded evidence, "
    "selected-checkpoint move, source change or alteration of18/6/50 GiB reserves. Restore18 GiB plus2.25 "
    "GiB full-training admission. Scratch is machine-local, not provider-persistent storage or backup.")


def checkpoint_identity(payload, current, expected, config_digest):
    if (expected != OWNERS["full_network_v34"] or current["state"] != "failed"
            or "OutOfMemoryError" not in current["error"]
            or payload["step"] != 0 or payload["attempted_batches"] != 0
            or payload["gradient_updates_performed"] != 0 or payload["best_step"] != 0
            or payload["parent_training_updates"] != 98 or payload["config_sha256"] != config_digest
            or payload["parent_checkpoint_sha256"] != memory.training.PARENT_SHA256
            or payload["preflight_only"] is not False or payload["independent_confirmation_passed"] is not False
            or payload["full_network_training"] is not True or payload["optimizer"]["state"]):
        raise ValueError("Preserve only the failed run's unchanged zero-update initial checkpoint")
    parent_path = memory.training.PARENT
    if (parent_path.is_symlink() or not parent_path.is_file() or parent_path.stat().st_nlink != 1
            or parent_path.stat().st_dev != ROOT.stat().st_dev or sha256(parent_path) != memory.training.PARENT_SHA256):
        raise ValueError("Keep the original selected initializer physically on checkpoint storage")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False, mmap=True)["model"]
    if set(payload["model"]) != set(parent):
        raise ValueError("Preserve every original model tensor")
    for name, value in parent.items():
        if not torch.isfinite(value).all() or not byte_equal(value, payload["model"][name]):
            raise ValueError("Every zero-update model tensor must equal the retained initializer bitwise")


def require_idle():
    for name in ("full_network_v34", "full_network_v34_pipeline"):
        current = previous.copying.read(ROOT / name / "status.json")
        if (current["state"] != "failed" or type(current["pid"]) is not int or current["pid"] <= 0
                or Path(f'/proc/{current["pid"]}').exists()):
            raise ValueError("Preserve the stopped failed trainer and coordinator, not a fabricated completion")
    for directory in (memory.OUTPUT, memory.template.OUTPUT):
        readers.terminal(directory / "status.json")
    readers.require_idle([str(ROOT / "full_network_v34/latest_checkpoint.pth"), str(memory.training.PARENT)])


def inventory():
    previous.copying.require_mount()
    require_idle()
    directory = ROOT / "full_network_v34"
    path = directory / "latest_checkpoint.pth"
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_dev != ROOT.stat().st_dev:
        raise ValueError("Only the original regular zero-update duplicate may move")
    config = previous.copying.read(directory / "config.json")
    capsule_path = directory / "training_batches/step_009/task_00.pt"
    capsule = torch.load(capsule_path, map_location="cpu", weights_only=False)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    records = [memory.training.json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    current = previous.copying.read(directory / "status.json")
    memory.validate_failure(current, previous.copying.read(ROOT / "full_network_v34_pipeline/status.json"), records, capsule, payload)
    checkpoint_identity(payload, current, OWNERS["full_network_v34"], sha256(directory / "config.json"))
    del payload
    inputs = {**config["input_sha256"], **config["source_sha256"], str(previous.copying.UNIT): sha256(previous.copying.UNIT)}
    paths = [capsule_path, memory.training.PARENT]
    paths.extend(directory / name for name in ("config.json", "registration.json", "status.json", "training.jsonl", "train_and_audit.log"))
    paths.extend(ROOT / "full_network_v34_pipeline" / name for name in ("config.json", "registration.json", "status.json"))
    for probe in (memory.template.OUTPUT, memory.OUTPUT):
        result = previous.copying.read(probe / "result.json")
        probe_config = previous.copying.read(probe / "config.json")
        if (result["complete"] is not True or result["training_updates_performed"] != 0
                or result["historical_task_and_teacher_digest_equal"] is not True
                or result["config_sha256"] != sha256(probe / "config.json")
                or previous.copying.read(probe / "status.json")["result_sha256"] != sha256(probe / "result.json")):
            raise ValueError("Bind both completed nontraining diagnostics without claiming trained accuracy")
        inputs.update(probe_config["source_sha256"])
        paths.extend(probe / name for name in ("config.json", "registration.json", "status.json", "result.json"))
    for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth", "validation_10.json"):
        if (directory / name).exists():
            raise ValueError("The failed V34 run has no eligible trained model or trained boundary")
    inputs.update({str(source): sha256(source) for source in paths})
    verify_hashes(inputs)
    return [{"source": str(path), "destination": str(DESTINATION / "full_network_v34" / path.name),
             "sha256": sha256(path), "bytes": path.stat().st_size}], inputs


copy_and_verify = types.FunctionType(previous.copy_and_verify.__code__, globals(), "copy_and_verify")
copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template = previous.template
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'len(records) != 4': 'len(records) != 1',
    'Preserve exactly four completed artifacts': 'Preserve exactly one failed-run zero-update artifact',
    '"canonical_artifacts_checked": 4': '"canonical_artifacts_checked": 1',
    'REPO / "tests/test_single_model_full_network_capacity.py",':
        'Path(previous.__file__).resolve(), REPO / "tests/test_single_model_failed_initial_checkpoint.py",',
    'copying_completed_full_network_capacity': 'copying_failed_run_initial_checkpoint',
    '"selected_model_only_checkpoints_retained_on_original_volume": True':
        '"selected_model_only_checkpoints_retained_on_original_volume": True,\n'
        '                   "failed_training_run_qualified": False, "zero_update_model_equals_retained_parent": True',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
