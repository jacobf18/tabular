"""Recover V46's pre-inference CUDA audit entry without repeating training."""

import argparse
import builtins
from contextlib import ExitStack
import dis
import fcntl
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import torch

import audit_single_model_stronger_mean as auditor
import run_single_model_stronger_mean_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ENTRY = original.ROOT / "stronger_mean_distillation_v46_cuda_entry_recovery_v1"
OUTPUT = original.ROOT / "stronger_mean_distillation_v46_qualification_recovery_v1"
AUDIT = original.ROOT / "stronger_mean_distillation_v46_recovered_audit_v1"
FAILED_REPLAY = original.AUDIT.with_name(original.AUDIT.name + "_task_replay")
STAGES = (("replay", False), ("audit-replay", True), ("confirm", False))
SELECTED_SHA256 = "122234a88e81ab3a3b9c88129ee4bd33e60030de097c669ea599e7a26821462b"
CONFIG_SHA256 = "ac98d26801b73579928c510b6e76a7b8479cd4fa05c5690a5599e69127733c49"
PIPELINE_SHA256 = "7fe9090f789c90b3af6a30be91b72b78a39c9aedc7393f35e698a06222429bbc"
ERROR = "RuntimeError('No CUDA GPUs are available')"
POLICY = (
    "Additive operational recovery of the original V46 audit's CUDA-hidden setup failure. "
    "Preserve the original failed audit/coordinator, complete64-attempt/55-update training, "
    "and sole step10 selection. Require empty task evidence, untouched retention/component "
    "replay and pristine qualification reservations before registration. Run the unchanged "
    "full auditor exactly once with CUDA visible at a new output path. Qualify with unchanged "
    "scientific function code in isolated globals; change only the audit destination. Keep "
    "the original284 confirmation cases,32 comparisons, all frozen real suites/review, "
    "24 BLAS/4 Torch threads, capacity guards and original qualification reservations. "
    "No retraining, reselection, new seed, tolerance change, completed inference repetition, "
    "inference ensemble or benchmark-parity claim. Fail closed on every subsequent failure."
)


def task_replay(output):
    return output.with_name(output.name + "_task_replay")


def require_stopped(state):
    if Path(f'/proc/{state["pid"]}').exists():
        raise ValueError("The original training and failed owners must be stopped")


def require_original_failure():
    original.verify(original.OUTPUT)
    verify_hashes({str(original.training.OUTPUT / "config.json"): CONFIG_SHA256,
                   str(original.OUTPUT / "config.json"): PIPELINE_SHA256,
                   str(original.training.OUTPUT / "selected_checkpoint.pth"): SELECTED_SHA256})
    terminal = original.read(original.training.OUTPUT / "status.json")
    failed = original.read(FAILED_REPLAY / "status.json")
    coordinator = original.read(original.OUTPUT / "status.json")
    for state in (terminal, failed, coordinator):
        require_stopped(state)
    if (terminal["state"] != "training_complete" or terminal["step"] != 64
            or terminal["accepted_updates"] != 55 or terminal["best"]["step"] != 10
            or failed["state"] != "failed" or failed["error"] != ERROR
            or coordinator["state"] != "failed"
            or "The original one-shot training audit failed" not in coordinator["error"]
            or "No CUDA GPUs are available" not in coordinator["error"]):
        raise ValueError("Recover only the exact completed V46 and pre-inference CUDA failure")
    if original.AUDIT.exists() or set(path.name for path in FAILED_REPLAY.iterdir()) != {
            "config.json", "status.json", "tasks.jsonl"} or (FAILED_REPLAY / "tasks.jsonl").stat().st_size:
        raise ValueError("The original audit must contain no completed replay or audit output")
    trace = (original.training.OUTPUT / "train_and_audit.log").read_text().rsplit("Traceback", 1)[-1]
    if ('in setup' not in trace or 'torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])' not in trace
            or not trace.endswith("RuntimeError: No CUDA GPUs are available\n")):
        raise ValueError("Bind the initialization traceback before any task or model inference")
    failed_config = original.read(FAILED_REPLAY / "config.json")
    verify_hashes(failed_config["source_sha256"])
    if (failed_config["training_config_sha256"] != CONFIG_SHA256
            or failed_config["adapted_source_sha256"] != auditor.ADAPTED_SOURCE_SHA256
            or failed_config["training_performed"] is not False
            or failed_config["qualification_changed"] is not False):
        raise ValueError("Preserve the original full-audit registration")


def require_pristine():
    config = original.read(original.training.OUTPUT / "config.json")
    retention = Path(config["retention_storage"]["replay"])
    components = Path(config["all_imputation_blend_replay"]["directory"])
    if not retention.is_dir() or any(retention.iterdir()) or components.exists():
        raise ValueError("No retention or component replay may have begun")
    for directory in (original.REPLAY, original.CONFIRMATION):
        if (set(path.name for path in directory.iterdir()) != {"config.json", "registration.json", "status.json"}
                or original.read(directory / "status.json")["state"] != "registered"):
            raise ValueError("Require pristine original endpoint and confirmation reservations")
    if any(path.exists() for path in (AUDIT, task_replay(AUDIT), original.CAMPAIGN, original.REVIEW,
                                      original.training.OUTPUT / "best_validation_checkpoint.pth")):
        raise ValueError("Never repeat an audit, published selection or real campaign")


def owned_functions():
    return {name: value for name, value in vars(original).items()
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(original)}


def execution_namespace():
    namespace = {**vars(original), "AUDIT": AUDIT}
    for name, function in owned_functions().items():
        unwrapped = inspect.unwrap(function)
        if unwrapped.__closure__ is not None:
            raise ValueError("Do not retain closures into original pipeline globals")
        cloned = types.FunctionType(unwrapped.__code__, namespace, unwrapped.__name__, unwrapped.__defaults__)
        cloned.__kwdefaults__ = unwrapped.__kwdefaults__
        if function is not unwrapped:
            if name not in {"replay_actual", "confirm"}:
                raise ValueError("Only the original known no-grad wrappers are admitted")
            cloned = torch.no_grad()(cloned)
        namespace[name] = cloned
    return namespace


def unresolved_globals(function):
    function = inspect.unwrap(function)
    pending = [function.__code__]
    missing = set()
    while pending:
        code = pending.pop()
        pending.extend(value for value in code.co_consts if isinstance(value, types.CodeType))
        missing.update(entry.argval for entry in dis.get_instructions(code)
                       if entry.opname == "LOAD_GLOBAL" and entry.argval not in function.__globals__
                       and not hasattr(builtins, entry.argval))
    return sorted(missing)


def require_static_globals():
    functions = list(owned_functions().values())
    functions.extend(value for value in execution_namespace().values() if inspect.isfunction(value))
    for module in (auditor, auditor.retention_training, auditor.blend_replay):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                functions.append(value)
            elif inspect.isclass(value) and value.__module__ == module.__name__:
                functions.extend(member for member in vars(value).values() if inspect.isfunction(member))
    failures = {function.__qualname__: unresolved_globals(function) for function in functions
                if unresolved_globals(function)}
    if failures:
        raise ValueError(f"Unresolved scientific globals: {failures}")


def require_cuda():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not torch.cuda.is_available():
        raise ValueError("Full task regeneration requires a visible CUDA device")
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    if value.item() != 1:
        raise ValueError("CUDA arithmetic smoke check failed")
    return {"device": torch.cuda.get_device_name(), "devices": torch.cuda.device_count(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "model_inference": False}


def preserved_paths():
    paths = [original.training.OUTPUT / name for name in (
        "config.json", "registration.json", "status.json", "train_and_audit.log", "training.jsonl",
        "validation.jsonl", "endpoint.json", "selected_checkpoint.pth", "validation_64.json")]
    paths.extend(original.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "original_pipeline.log"))
    paths.extend(FAILED_REPLAY / name for name in ("config.json", "status.json", "tasks.jsonl"))
    paths.extend(directory / name for directory in (original.REPLAY, original.CONFIRMATION)
                 for name in ("config.json", "registration.json"))
    return paths


def register():
    if ENTRY.exists() or OUTPUT.exists():
        raise FileExistsError("Never replace either operational recovery registration")
    require_original_failure()
    require_pristine()
    require_static_globals()
    hardware = require_cuda()
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_stronger_mean_cuda_entry.py"]
    config = {"policy": POLICY, "original_pipeline": str(original.OUTPUT), "audit": str(AUDIT),
              "selected_step": 10, "selected_sha256": SELECTED_SHA256, "hardware_smoke": hardware,
              "source_sha256": {str(path): sha256(path) for path in sources},
              "input_sha256": {str(path): sha256(path) for path in preserved_paths()},
              "pristine_reservations": {str(path): sha256(path / "status.json")
                                        for path in (original.REPLAY, original.CONFIRMATION)},
              "qualification_stages": [list(stage) for stage in STAGES]}
    for directory, role in ((ENTRY, "audit_entry"), (OUTPUT, "qualification_coordinator")):
        directory.mkdir()
        write_json(directory / "config.json", {**config, "role": role})
        write_json(directory / "registration.json", {"config_sha256": sha256(directory / "config.json"),
                   "registered_unix": time.time(), "training_repeated": False, "scientific_stage_started": False})
        original.status(directory, "registered")


def verify(directory):
    config = original.read(directory / "config.json")
    if (original.read(directory / "registration.json")["config_sha256"] != sha256(directory / "config.json")
            or config["policy"] != POLICY or config["audit"] != str(AUDIT)
            or config["selected_sha256"] != SELECTED_SHA256 or config["selected_step"] != 10
            or config["qualification_stages"] != [list(stage) for stage in STAGES]):
        raise ValueError("Preserve both exact operational-only registrations")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    require_original_failure()
    require_static_globals()
    return config


def run_audit():
    verify(ENTRY)
    require_pristine()
    require_cuda()
    command = [sys.executable, "-u", str(Path(auditor.__file__).resolve()), "--output", str(AUDIT)]
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(TMPDIR="/run", PYTHONPATH=f"{REPO / 'scripts'}:{REPO / 'src'}",
                       OPENBLAS_NUM_THREADS="24", MKL_NUM_THREADS="24", OMP_NUM_THREADS="4")
    with (ENTRY / "audit.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO)
        original.status(ENTRY, "auditing", child_pid=process.pid, command=command)
        if process.wait():
            raise RuntimeError("Recovered audit failed; preserve all outputs without repeating inference")
    namespace = execution_namespace()
    namespace["prerequisite"](original.OUTPUT)
    verify(ENTRY)
    write_json(ENTRY / "result.json", {"complete": True, "training_repeated": False,
               "original_failures_preserved": True, "audit_sha256": sha256(AUDIT / "audit.json")})
    original.status(ENTRY, "complete")


def require_completed_audit():
    verify(ENTRY)
    result = original.read(ENTRY / "result.json")
    if (original.read(ENTRY / "status.json")["state"] != "complete" or result["complete"] is not True
            or result["audit_sha256"] != sha256(AUDIT / "audit.json")):
        raise ValueError("Qualification requires the independently completed recovered full audit")


def qualify():
    verify(OUTPUT)
    require_completed_audit()
    require_cuda()
    for stage, cpu in STAGES:
        original.run_child(OUTPUT, stage, Path(__file__).resolve(), ["--stage", stage], cpu=cpu)
    if original.read(original.CONFIRMATION / "status.json")["state"] == "awaiting_independent_audit":
        original.run_child(OUTPUT, "audit-confirmation", Path(__file__).resolve(), ["--stage", "audit-confirmation"], cpu=True)
    gate = original.read(original.CONFIRMATION / "gate.json")
    if gate["passed"]:
        original.run_child(OUTPUT, "benchmarks", REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
                           ["--run", str(original.training.OUTPUT), "--confirmation", str(original.CONFIRMATION),
                            "--source", str(original.PARENT_CACHE), "--output", str(original.CAMPAIGN)])
        original.run_child(OUTPUT, "review", REPO / "scripts/review_single_matrixformer_campaign.py",
                           ["--campaign", str(original.CAMPAIGN), "--output", str(original.REVIEW)], cpu=True)
    verify(OUTPUT)
    write_json(OUTPUT / "result.json", {"complete": True, "gate": gate,
               "audit_sha256": sha256(AUDIT / "audit.json"), "training_repeated": False,
               "original_failures_preserved": True, "scientific_stages_repeated": False,
               "campaign": str(original.CAMPAIGN) if gate["passed"] else None,
               "review": str(original.REVIEW) if gate["passed"] else None,
               "no_inference_ensemble": True, "goal_success_not_assessed": True})
    original.status(OUTPUT, "complete", confirmed=gate["passed"])


def scientific_stage(stage):
    verify(OUTPUT)
    require_completed_audit()
    namespace = execution_namespace()
    function = {"replay": "replay", "audit-replay": "audit_replay", "confirm": "confirm",
                "audit-confirmation": "audit_confirmation"}[stage]
    namespace[function](original.OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "audit", "qualify", "replay",
                                                         "audit-replay", "confirm", "audit-confirmation"))
    stage = parser.parse_args().stage
    with original.arithmetic_recovery.previous.original_arithmetic():
        if stage == "register":
            with ExitStack() as stack:
                for directory in (original.training.OUTPUT, original.OUTPUT):
                    lock = stack.enter_context((directory / "run.lock").open("r"))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                register()
            return
        directory = ENTRY if stage == "audit" else OUTPUT if stage == "qualify" else (
            original.REPLAY if "replay" in stage else original.CONFIRMATION)
        with (directory / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if stage in {"audit", "qualify"} and original.read(directory / "status.json")["state"] != "registered":
                raise ValueError("Never restart either registered recovery worker")
            try:
                if stage == "audit":
                    run_audit()
                elif stage == "qualify":
                    qualify()
                else:
                    scientific_stage(stage)
            except BaseException as error:
                original.status(directory, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
