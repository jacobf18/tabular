"""Supply V29's missing deepcopy without modifying registered science or failed evidence."""

import argparse
import builtins
from copy import deepcopy
import dis
import fcntl
import inspect
import marshal
from pathlib import Path
import time
import types

import torch

import run_single_model_legacy_strata_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.ROOT / "legacy_strata_v29_replay_recovery"
REPLAY = original.ROOT / "legacy_strata_v29_recovered_endpoint_replay"
SCRATCH = original.diagnostics.storage.SCRATCH / "legacy_strata_v29/qualification/recovered_endpoint_replay"
STAGES = (("replay", False), ("audit-replay", True), ("confirm", False))
SELECTED_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
AUDIT_SHA256 = "deeac0caa8fb0598ea312c9b0269676061a53fbf3136f378fe6d71e4347c5409"
POLICY = ("Operational dependency recovery only. Preserve the original failed coordinator and empty failed replay "
    "prefix byte-for-byte. No endpoint teacher or student inference occurred in that prefix. Reuse the completed "
    "64-attempt training audit and sole selected step25 checkpoint, without retraining or reselection. Clone original "
    "pipeline function code objects into isolated globals supplying deepcopy, retaining no-grad semantics. Change "
    "only verification/orchestration and the new explicitly UUID-bound scratch replay location. Preserve all original "
    "scientific arithmetic,22 retention guards,152-case selected replay and unused284-case confirmation reservation. "
    "Execute each actual neural stage and independent audit once;1136 confirmation grids precede any publication. "
    "Only a passed original gate allows the full frozen tabular, four-dataset recommender and LLM campaign/review. "
    "Keep18/6/50 GiB reserves and original stage budgets. Scratch is machine-local, not provider-persistent storage "
    "or external backup. No original source mutation, scientific retry, relaxed gate, inference ensemble, hidden "
    "regression or historical full-autograd replay claim. This recovery does not establish goal success.")


def owned_functions():
    return {name: value for name, value in vars(original).items()
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(original)}


def function_inventory():
    return {name: {"code_sha256": original.hashlib.sha256(marshal.dumps(inspect.unwrap(value).__code__)).hexdigest(),
                   "no_grad": value is not inspect.unwrap(value)}
            for name, value in owned_functions().items()}


def code_globals(code):
    result = {instruction.argval for instruction in dis.get_instructions(code)
              if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            result.update(code_globals(constant))
    return result


def missing_globals(namespace):
    return {name: sorted(missing) for name, value in owned_functions().items()
            if (missing := code_globals(inspect.unwrap(value).__code__) - namespace.keys() - vars(builtins).keys())}


def execution_namespace():
    namespace = {**vars(original), "deepcopy": deepcopy, "OUTPUT": OUTPUT, "REPLAY": REPLAY}
    for name, value in owned_functions().items():
        unwrapped = inspect.unwrap(value)
        if unwrapped.__closure__ is not None:
            raise ValueError("Do not retain closures into the original execution namespace")
        cloned = types.FunctionType(unwrapped.__code__, namespace, unwrapped.__name__, unwrapped.__defaults__)
        cloned.__kwdefaults__ = unwrapped.__kwdefaults__
        if value is not unwrapped:
            if name not in {"replay_actual", "confirm"}:
                raise ValueError("Only the known torch.no_grad wrappers may be reconstructed")
            cloned = torch.no_grad()(cloned)
        namespace[name] = cloned
    namespace["verify"] = verify
    if missing_globals(namespace):
        raise ValueError("Every original pipeline global dependency must resolve, including nested code")
    return namespace


def require_failed_prefix():
    original.verify(original.OUTPUT)
    for directory, error in ((original.OUTPUT, "RuntimeError('replay failed; preserve its artifacts without restarting')"),
                             (original.REPLAY, 'NameError("name \'deepcopy\' is not defined")')):
        state = original.read(directory / "status.json")
        if (state["state"] != "failed" or state["error"] != error
                or original.process_identity(state["pid"]) is not None):
            raise ValueError("Require the exact stopped pre-inference dependency failure")
    expected = {"config.json", "registration.json", "status.json", "run.lock", "endpoint_tasks"}
    tasks = original.REPLAY / "endpoint_tasks"
    if ({path.name for path in original.REPLAY.iterdir()} != expected or not tasks.is_dir()
            or tasks.is_symlink() or any(tasks.iterdir())):
        raise ValueError("Never recover if the failed prefix contains task or prediction artifacts")
    if missing_globals(vars(original)) != {"materialize_endpoint_tasks": ["deepcopy"]}:
        raise ValueError("Only the observed single missing dependency may be recovered")


def require_unused_confirmation():
    directory = original.CONFIRMATION
    if (original.read(directory / "status.json")["state"] != "registered"
            or {path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}):
        raise ValueError("Never recover a consumed confirmation reservation")
    if (original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Never recover over a published checkpoint or a real campaign")


def require_storage(prepared):
    original.diagnostics.storage.require_mount()
    if (not SCRATCH.parent.is_dir() or SCRATCH.parent.resolve() != SCRATCH.parent
            or not SCRATCH.is_relative_to(original.diagnostics.storage.SCRATCH)
            or SCRATCH.parent.stat().st_dev != original.diagnostics.storage.SCRATCH.stat().st_dev
            or REPLAY.parent != original.ROOT):
        raise ValueError("Bind new replay diagnostics to the existing UUID-checked scratch mount")
    if prepared and (not REPLAY.is_symlink() or REPLAY.readlink() != SCRATCH or REPLAY.resolve() != SCRATCH
                     or not SCRATCH.is_dir() or SCRATCH.is_symlink() or SCRATCH.resolve() != SCRATCH):
        raise ValueError("Preserve the exact registered canonical link and scratch destination")


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, REPLAY, SCRATCH)):
        raise FileExistsError("Never restart an existing recovery")
    require_failed_prefix()
    require_unused_confirmation()
    original.prerequisite(original.OUTPUT)
    selected = original.training.OUTPUT / "selected_checkpoint.pth"
    verify_hashes({str(selected): SELECTED_SHA256, str(original.AUDIT / "audit.json"): AUDIT_SHA256})
    execution_namespace()
    require_storage(False)
    original.require_stage_capacity("replay")
    inputs = [selected, original.AUDIT / "audit.json", original.training.OUTPUT / "endpoint.json"]
    for directory in (original.OUTPUT, original.REPLAY):
        inputs.extend(path for path in directory.iterdir() if path.is_file())
    inputs.extend(original.CONFIRMATION / name for name in ("config.json", "registration.json"))
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_legacy_strata_replay_recovery.py"]
    config = {"policy": POLICY, "original_pipeline": str(original.OUTPUT), "failed_replay": str(original.REPLAY),
              "replay": str(REPLAY), "scratch": str(SCRATCH), "confirmation": str(original.CONFIRMATION),
              "confirmation_status": original.read(original.CONFIRMATION / "status.json"),
              "confirmation_status_sha256": sha256(original.CONFIRMATION / "status.json"),
              "function_inventory": function_inventory(), "input_sha256": {str(path): sha256(path) for path in inputs},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "qualification_stages": [list(stage) for stage in STAGES],
              "minimum_free_bytes": original.diagnostics.RESERVES,
              "stage_write_budget_bytes": original.STAGE_WRITE_BUDGETS,
              "confirmation_cases": original.training.confirmation_cases(), "scratch_is_provider_persistent": False}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "scientific_stage_started": False, "training_repeated": False})
    SCRATCH.mkdir()
    REPLAY.symlink_to(SCRATCH, target_is_directory=True)
    write_json(REPLAY / "config.json", {"recovery_config_sha256": sha256(OUTPUT / "config.json"),
               "original_reservation_sha256": sha256(original.REPLAY / "config.json"), "policy": POLICY})
    write_json(REPLAY / "registration.json", {"config_sha256": sha256(REPLAY / "config.json"),
               "registered_unix": time.time(), "scientific_stage_started": False})
    original.status(REPLAY, "registered")
    original.status(OUTPUT, "registered")
    verify(OUTPUT)


def verify(output):
    if output != OUTPUT:
        raise ValueError("Only the registered additive recovery may execute")
    config = original.read(OUTPUT / "config.json")
    if (original.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["original_pipeline"] != str(original.OUTPUT)
            or config["failed_replay"] != str(original.REPLAY) or config["replay"] != str(REPLAY)
            or config["scratch"] != str(SCRATCH) or config["confirmation"] != str(original.CONFIRMATION)
            or config["function_inventory"] != function_inventory()
            or config["confirmation_cases"] != original.training.confirmation_cases()
            or config["qualification_stages"] != [list(stage) for stage in STAGES]
            or config["minimum_free_bytes"] != original.diagnostics.RESERVES
            or config["stage_write_budget_bytes"] != original.STAGE_WRITE_BUDGETS
            or config["scratch_is_provider_persistent"] is not False):
        raise ValueError("Preserve the exact registered recovery scope, scientific code and storage budgets")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    require_failed_prefix()
    require_storage(True)
    if (original.read(REPLAY / "config.json") != {
            "recovery_config_sha256": sha256(OUTPUT / "config.json"),
            "original_reservation_sha256": sha256(original.REPLAY / "config.json"), "policy": POLICY}
            or original.read(REPLAY / "registration.json")["config_sha256"] != sha256(REPLAY / "config.json")):
        raise ValueError("Preserve the new replay reservation and the unchanged original confirmation")
    return config


def run():
    config = verify(OUTPUT)
    require_unused_confirmation()
    if sha256(original.CONFIRMATION / "status.json") != config["confirmation_status_sha256"]:
        raise ValueError("The reserved confirmation status changed before recovery execution")
    namespace = execution_namespace()
    namespace["prerequisite"](OUTPUT)
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
    write_json(OUTPUT / "result.json", {"complete": True, "gate": gate,
               "training_audit_sha256": sha256(original.AUDIT / "audit.json"),
               "failed_replay_preserved": str(original.REPLAY), "recovered_replay": str(REPLAY),
               "campaign": str(original.CAMPAIGN) if gate["passed"] else None,
               "review": str(original.REVIEW) if gate["passed"] else None,
               "scientific_stages_repeated": False, "no_inference_ensemble": True, "goal_success_not_assessed": True})
    original.status(OUTPUT, "complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit-replay", "confirm", "audit-confirmation"), required=True)
    stage = parser.parse_args().stage
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
