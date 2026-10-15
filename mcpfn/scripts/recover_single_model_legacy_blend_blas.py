"""Recover V41 qualification under its original BLAS arithmetic, preserving failed evidence."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import marshal
import os
from pathlib import Path
import time
import types

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

import run_single_model_legacy_blend_teacher_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.ROOT / "legacy_blend_teacher_v41_blas_recovery_v1"
DIAGNOSIS = original.ROOT / "legacy_blend_teacher_v41_blas_diagnosis_v1"
REPLAY = original.ROOT / "legacy_blend_teacher_v41_recovered_endpoint_replay"
SCRATCH = original.training.SCRATCH / "qualification/recovered_endpoint_replay"
BLAS_THREADS = 24
STAGES = (("replay", False), ("audit-replay", True), ("confirm", False))
CONFIG_SHA256 = "64dee12d225b00d7c31f4b7dbb144e601b916cc99fc269347e7268443ddfc683"
SELECTED_SHA256 = "43d5a07c16812967a7ff0453581e5a0705e2d203ae68ba9403f3482f1f492c81"
AUDIT_SHA256 = "356672096ce19ee456858b88e42aecd204ba39b2fc9d97d79d62734c934f28e2"
ERROR = "Independently refitted observed weights must exactly reconstruct the cached target"
POLICY = ("Operational BLAS-environment recovery only. The original V41 training and complete GPU training audit "
    "used NumPy/SciPy OpenBLAS with24 threads; the qualification launcher imposed4, changing15 of128 archived "
    "routing weights in their least significant float64 bits, with zero changed FP32 target cells. Restore24 "
    "BLAS threads and4 Torch threads within qualification processes, preserving every exact original check. "
    "Preserve the failed coordinator and empty pre-inference replay prefix byte-for-byte. Clone unchanged "
    "scientific function code into isolated globals with a new scratch replay destination and additive "
    "verification only. Reuse the complete64-attempt/56-update audit and sole step50/43-update selection. "
    "No retraining, new proposal, reselection, tolerance relaxation, model ensemble or consumed confirmation "
    "retry. Keep all22 development and32 confirmation comparisons, all284 reserved confirmation cases and "
    "every frozen real campaign requirement. Frozen benchmark/review subprocesses retain the original launcher "
    "environment, including4 BLAS threads. Preserve original stage budgets and18/6/50 GiB reserves. "
    "Scratch is not provider-persistent backup. This operational recovery does not establish benchmark parity.")


def owned_functions():
    return {name: value for name, value in vars(original).items()
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(original)}


def function_inventory():
    return {name: {"code_sha256": hashlib.sha256(marshal.dumps(inspect.unwrap(value).__code__)).hexdigest(),
        "no_grad": value is not inspect.unwrap(value)} for name, value in owned_functions().items()}


def execution_namespace():
    namespace = {**vars(original), "OUTPUT": OUTPUT, "REPLAY": REPLAY}
    for name, function in owned_functions().items():
        unwrapped = inspect.unwrap(function)
        if unwrapped.__closure__ is not None:
            raise ValueError("Do not retain closures into original pipeline globals")
        cloned = types.FunctionType(unwrapped.__code__, namespace, unwrapped.__name__, unwrapped.__defaults__)
        cloned.__kwdefaults__ = unwrapped.__kwdefaults__
        if function is not unwrapped:
            if name not in {"replay_actual", "confirm"}:
                raise ValueError("Preserve only the original known no-grad wrappers")
            cloned = torch.no_grad()(cloned)
        namespace[name] = cloned
    namespace["verify"] = verify
    return namespace


def blas_profile():
    return [{key: pool[key] for key in ("filepath", "internal_api", "version", "num_threads")}
        for pool in threadpool_info() if pool["user_api"] == "blas"]


@contextmanager
def original_arithmetic():
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            profile = blas_profile()
            if len(profile) != 2 or any(pool["internal_api"] != "openblas"
                    or pool["num_threads"] != BLAS_THREADS for pool in profile):
                raise ValueError("Restore the two original OpenBLAS pools, not an arbitrary runtime")
            yield profile
    finally:
        torch.set_num_threads(prior_threads)


def require_failed_prefix():
    original.verify(original.OUTPUT)
    verify_hashes({str(original.training.OUTPUT / "config.json"): CONFIG_SHA256,
        str(original.training.OUTPUT / "selected_checkpoint.pth"): SELECTED_SHA256,
        str(original.AUDIT / "audit.json"): AUDIT_SHA256})
    for directory, error in ((original.OUTPUT, "RuntimeError('replay failed; preserve its artifacts without restarting')"),
            (original.REPLAY, repr(ValueError(ERROR)))):
        state = original.read(directory / "status.json")
        if (state["state"] != "failed" or state["error"] != error
                or original.process_identity(state["pid"]) is not None):
            raise ValueError("Require the exact stopped prerequisite failure")
    if {path.name for path in original.REPLAY.iterdir()} != {"config.json", "registration.json", "status.json", "run.lock"}:
        raise ValueError("Never recover an endpoint prefix containing inference or materialized tasks")
    log = (original.OUTPUT / "replay.log").read_text()
    if ERROR not in log or "report = prerequisite(output)" not in log:
        raise ValueError("Prove failure before endpoint replay started")
    if original.read(original.training.OUTPUT / "status.json")["best"]["step"] != 50:
        raise ValueError("Preserve the sole selected step50 checkpoint")


def require_unused_confirmation():
    directory = original.CONFIRMATION
    if (original.read(directory / "status.json")["state"] != "registered"
            or {path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}
            or original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Never repeat confirmation or overwrite a publication or campaign")


def diagnose():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Diagnose saved arithmetic with CUDA hidden and uninitialized")
    if DIAGNOSIS.exists():
        raise FileExistsError("Preserve the completed diagnosis")
    require_failed_prefix()
    config = original.read(original.training.OUTPUT / "config.json")
    component = original.blend_replay.layout(original.training.SCRATCH)
    rows = [original.json.loads(line) for line in (component / "tasks.jsonl").read_text().splitlines()]
    legacy = [row for row in rows if row["role"] == "observed_blend"]
    if len(rows) != 1792 or len(legacy) != 128:
        raise ValueError("Diagnose every admitted legacy task, not a favorable subset")
    differences, hashes, cells = [], {}, 0
    with original_arithmetic() as profile:
        for row in legacy:
            path = Path(row["archive"])
            verify_hashes({str(path): row["archive_sha256"]})
            hashes[str(path)] = row["archive_sha256"]
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != {"source", "teacher", "general", "specialist"}:
                    raise ValueError("Use only saved observed sources and full component/target grids")
                source, target = torch.from_numpy(saved["source"]), torch.from_numpy(saved["teacher"])
                general, specialist = torch.from_numpy(saved["general"]), torch.from_numpy(saved["specialist"])
                cells += target.numel()
                for threads in (BLAS_THREADS, 4):
                    with threadpool_limits(limits=threads, user_api="blas"):
                        prediction, decision = original.blend_replay.replay.blend_components(source, general, specialist)
                        routing = original.blend_replay.replay.routing_record("observed_blend", config, decision)
                    changed = [key for key in routing if routing[key] != row["routing"][key]]
                    if not torch.equal(prediction, target) or (threads == BLAS_THREADS and changed):
                        raise ValueError("Recovery must exactly reproduce every original FP32 target and routing field")
                    if changed:
                        if changed != ["specialist_weight"]:
                            raise ValueError("The observed failure must be only the float64 BLAS weight")
                        differences.append({"archive": str(path), "saved_weight": row["routing"]["specialist_weight"],
                            "four_thread_weight": decision["specialist_weight"], "target_cells_changed": 0})
        if len(differences) != 15:
            raise ValueError("Preserve the complete independently observed failure population")
        original.prerequisite(original.OUTPUT)
    hashes[str(original.AUDIT / "audit.json")] = AUDIT_SHA256
    hashes.update({str(component / name): sha256(component / name) for name in ("tasks.jsonl", "summary.json")})
    verify_hashes(hashes)
    libraries = {pool["filepath"]: sha256(pool["filepath"]) for pool in profile}
    DIAGNOSIS.mkdir()
    write_json(DIAGNOSIS / "diagnosis.json", {"complete": True, "policy": POLICY, "legacy_archives": len(legacy),
        "target_cells": cells, "original_exact_mismatches": 0, "four_thread_routing_mismatches": differences,
        "target_cells_changed": 0, "profile": profile, "library_sha256": libraries, "artifact_sha256": hashes,
        "full_original_prerequisite_passed": True, "cuda_initialized": torch.cuda.is_initialized(),
        "source_sha256": {str(Path(__file__).resolve()): sha256(__file__)}})


def require_storage(prepared):
    original.diagnostics.storage.require_mount()
    if (not SCRATCH.parent.is_dir() or SCRATCH.parent.resolve() != SCRATCH.parent
            or not SCRATCH.is_relative_to(original.diagnostics.storage.SCRATCH)
            or SCRATCH.parent.stat().st_dev != original.diagnostics.storage.SCRATCH.stat().st_dev):
        raise ValueError("Keep recovery on the original UUID-checked scratch mount")
    if prepared and (not REPLAY.is_symlink() or REPLAY.readlink() != SCRATCH or not SCRATCH.is_dir()
            or SCRATCH.is_symlink() or SCRATCH.resolve() != SCRATCH):
        raise ValueError("Preserve the exact additive replay destination")


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, REPLAY, SCRATCH)):
        raise FileExistsError("Never restart an existing recovery")
    require_failed_prefix()
    require_unused_confirmation()
    diagnosis = original.read(DIAGNOSIS / "diagnosis.json")
    if (diagnosis["policy"] != POLICY or not diagnosis["complete"] or diagnosis["legacy_archives"] != 128
            or diagnosis["original_exact_mismatches"] != 0 or diagnosis["target_cells_changed"] != 0
            or len(diagnosis["four_thread_routing_mismatches"]) != 15
            or not diagnosis["full_original_prerequisite_passed"] or diagnosis["cuda_initialized"]):
        raise ValueError("Require the complete saved-grid root-cause diagnosis")
    verify_hashes(diagnosis["artifact_sha256"])
    verify_hashes(diagnosis["library_sha256"])
    verify_hashes(diagnosis["source_sha256"])
    original.prerequisite(original.OUTPUT)
    execution_namespace()
    require_storage(False)
    original.require_stage_capacity("replay")
    inputs = [DIAGNOSIS / "diagnosis.json", original.AUDIT / "audit.json",
        original.training.OUTPUT / "selected_checkpoint.pth", original.training.OUTPUT / "endpoint.json"]
    for directory in (original.OUTPUT, original.REPLAY):
        inputs.extend(path for path in directory.iterdir() if path.is_file())
    inputs.extend(original.CONFIRMATION / name for name in ("config.json", "registration.json"))
    config = {"policy": POLICY, "original_pipeline": str(original.OUTPUT), "replay": str(REPLAY),
        "scratch": str(SCRATCH), "blas_threads": BLAS_THREADS, "torch_threads": 4,
        "confirmation_status_sha256": sha256(original.CONFIRMATION / "status.json"),
        "confirmation_cases": original.training.confirmation_cases(), "function_inventory": function_inventory(),
        "input_sha256": {str(path): sha256(path) for path in inputs}, "library_sha256": diagnosis["library_sha256"],
        "source_sha256": {str(path): sha256(path) for path in (Path(__file__).resolve(),
            REPO / "tests/test_single_model_legacy_blend_blas_recovery.py")},
        "minimum_free_bytes": original.diagnostics.RESERVES, "stage_write_budget_bytes": original.STAGE_WRITE_BUDGETS}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
        "registered_unix": time.time(), "scientific_stage_started": False})
    SCRATCH.mkdir()
    REPLAY.symlink_to(SCRATCH, target_is_directory=True)
    write_json(REPLAY / "config.json", {"recovery_config_sha256": sha256(OUTPUT / "config.json"),
        "original_reservation_sha256": sha256(original.REPLAY / "config.json"), "policy": POLICY})
    write_json(REPLAY / "registration.json", {"config_sha256": sha256(REPLAY / "config.json")})
    original.status(REPLAY, "registered")
    original.status(OUTPUT, "registered")
    verify(OUTPUT)


def verify(output):
    if output != OUTPUT:
        raise ValueError("Only the registered additive recovery may execute")
    config = original.read(OUTPUT / "config.json")
    if (original.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["original_pipeline"] != str(original.OUTPUT)
            or config["replay"] != str(REPLAY) or config["scratch"] != str(SCRATCH)
            or config["blas_threads"] != BLAS_THREADS or config["torch_threads"] != 4
            or config["function_inventory"] != function_inventory()
            or config["confirmation_cases"] != original.training.confirmation_cases()
            or config["minimum_free_bytes"] != original.diagnostics.RESERVES
            or config["stage_write_budget_bytes"] != original.STAGE_WRITE_BUDGETS):
        raise ValueError("Never change recovery scope, scientific code, cases or resource budgets")
    for key in ("input_sha256", "source_sha256", "library_sha256"):
        verify_hashes(config[key])
    require_failed_prefix()
    require_storage(True)
    if (original.read(REPLAY / "config.json") != {"recovery_config_sha256": sha256(OUTPUT / "config.json"),
            "original_reservation_sha256": sha256(original.REPLAY / "config.json"), "policy": POLICY}
            or original.read(REPLAY / "registration.json")["config_sha256"] != sha256(REPLAY / "config.json")):
        raise ValueError("Preserve the additive replay reservation")
    return config


def run():
    config = verify(OUTPUT)
    require_unused_confirmation()
    if sha256(original.CONFIRMATION / "status.json") != config["confirmation_status_sha256"]:
        raise ValueError("Do not consume a changed confirmation reservation")
    for stage, cpu in STAGES:
        verify(OUTPUT)
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
    write_json(OUTPUT / "result.json", {"complete": True, "gate": gate, "training_audit_sha256": AUDIT_SHA256,
        "failed_prefix_preserved": str(original.OUTPUT), "recovered_replay": str(REPLAY),
        "training_repeated": False, "reselected": False, "goal_success_not_assessed": True})
    original.status(OUTPUT, "complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnose", "register", "run", "replay", "audit-replay", "confirm", "audit-confirmation"), required=True)
    stage = parser.parse_args().stage
    with original_arithmetic():
        if stage == "diagnose":
            diagnose()
            return
        if stage == "register":
            register()
            return
        verify(OUTPUT)
        directory = OUTPUT if stage == "run" else REPLAY if "replay" in stage else original.CONFIRMATION
        with (directory / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                original.require_stage_capacity("replay" if stage == "run" else stage)
                if stage == "run":
                    if original.read(OUTPUT / "status.json")["state"] != "registered":
                        raise ValueError("Never restart a begun recovery")
                    run()
                else:
                    execution_namespace()[stage.replace("-", "_")](OUTPUT)
            except BaseException as error:
                original.status(directory, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
