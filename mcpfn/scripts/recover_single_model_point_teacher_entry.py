"""Finish V21's unchanged negative qualification after a teacher CLI failed before execution."""

import fcntl
from pathlib import Path
import shutil

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import run_single_model_point_inclusive_pipeline as pipeline
import run_single_model_point_teacher_replay as gateway


OUTPUT = gateway.RECOVERY
TRAINING_AUDIT_SHA256 = "dcb90b80f1382755f9d8f8699ba11837ddda20e6477ecd4fe68b0729c440b1c7"
ORIGINAL_ERROR = "RuntimeError('teacher-replay failed; preserve its artifacts without restarting')"
RESERVATION_FILES = {"config.json", "registration.json", "status.json"}


def require_absent(pid):
    if type(pid) is not int or pid <= 0 or Path(f"/proc/{pid}").exists():
        raise ValueError("Require the original trainer and failed coordinator to be absent")


def validate_failure(failed, teacher_status, teacher_files, replay_status, confirmation_status, log, audit):
    if (failed.get("state") != "failed" or failed.get("error") != ORIGINAL_ERROR
            or "NameError: name 'main' is not defined" not in log
            or 'replay_single_model_point_inclusive_teachers.py", line 14, in <module>' not in log
            or teacher_status.get("state") != "registered" or set(teacher_files) != RESERVATION_FILES
            or replay_status.get("state") != "registered" or confirmation_status.get("state") != "registered"):
        raise ValueError("Recover only the documented failure before any teacher, endpoint or confirmation execution")
    if (not audit.get("complete") or not audit.get("complete_training_audit") or audit.get("through_step") != 64
            or audit.get("attempted_batches") != 64 or audit.get("accepted_updates") != 63
            or audit.get("training_episodes") != 1792 or audit.get("teacher_identity_checked_tasks") != 1792
            or audit.get("prediction_grids") != 1216 or audit.get("cuda_initialized") is not False
            or audit.get("best", {}).get("step") != 0 or not audit.get("raw_endpoint", {}).get("actual_final_parameters_checked")):
        raise ValueError("Require the complete unchanged training audit and original nonselection")


def failure_proof():
    failed = pipeline.read(pipeline.OUTPUT / "status.json")
    terminal = pipeline.read(pipeline.training.OUTPUT / "status.json")
    require_absent(failed["pid"])
    require_absent(terminal["pid"])
    if terminal["state"] != "training_complete" or terminal["step"] != 64 or terminal["accepted_updates"] != 63:
        raise ValueError("Preserve the original completed training outcome")
    audit_path = pipeline.AUDIT / "audit.json"
    if sha256(audit_path) != TRAINING_AUDIT_SHA256:
        raise ValueError("Preserve the original independently completed full training audit")
    audit = pipeline.read(audit_path)
    validate_failure(failed, pipeline.read(pipeline.TEACHERS / "status.json"),
                     {path.name for path in pipeline.TEACHERS.iterdir()}, pipeline.read(pipeline.REPLAY / "status.json"),
                     pipeline.read(pipeline.CONFIRMATION / "status.json"),
                     (pipeline.OUTPUT / "teacher-replay.log").read_text(), audit)
    for folder in (pipeline.REPLAY, pipeline.CONFIRMATION):
        if {path.name for path in folder.iterdir()} != RESERVATION_FILES:
            raise ValueError("Neither original endpoint replay nor confirmation may have started")
    if (pipeline.CAMPAIGN.exists() or pipeline.REVIEW.exists()
            or (pipeline.training.OUTPUT / "selected_checkpoint.pth").exists()
            or (pipeline.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("No selection, best alias or real campaign may be invented during this recovery")
    pipeline.verify(pipeline.OUTPUT)
    verify_hashes(audit["artifact_sha256"])
    inputs = [pipeline.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "teacher-replay.log", "training_audit.log")]
    inputs.extend([audit_path, pipeline.training.OUTPUT / "status.json"])
    inputs.extend(pipeline.TEACHERS / name for name in ("config.json", "registration.json"))
    return {str(path): sha256(path) for path in inputs}


def recover():
    if OUTPUT.exists():
        raise FileExistsError("Never restart or overwrite an entry-point recovery")
    with (pipeline.OUTPUT / "run.lock").open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = failure_proof()
        sources = [Path(__file__).resolve(), Path(gateway.__file__).resolve(), Path(pipeline.__file__).resolve(),
                   Path(gateway.teacher.__file__).resolve(), Path(gateway.teacher.baseline.__file__).resolve(),
                   REPO / "tests/test_single_model_point_teacher_entry_recovery.py"]
        OUTPUT.mkdir()
        for name in ("config.json", "registration.json"):
            shutil.copy2(pipeline.OUTPUT / name, OUTPUT / name)
        shutil.copy2(pipeline.TEACHERS / "status.json", OUTPUT / "original_teacher_status.json")
        inputs[str(OUTPUT / "original_teacher_status.json")] = sha256(OUTPUT / "original_teacher_status.json")
        config = {"policy": gateway.POLICY, "original_pipeline": str(pipeline.OUTPUT), "teacher_output": str(pipeline.TEACHERS),
                  "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sources},
                  "training_audit_sha256": TRAINING_AUDIT_SHA256, "training_restarted": False,
                  "original_failed_entry_invoked_no_teacher_function": True, "original_nonselection": True}
        write_json(OUTPUT / "recovery_config.json", config)
        write_json(OUTPUT / "recovery_registration.json", {"recovery_config_sha256": sha256(OUTPUT / "recovery_config.json"),
                   "original_failure_preceded_teacher_execution": True, "teacher_predictions_repeated": False,
                   "scientific_configuration_changed": False, "confirmation_repeated": False})
        try:
            gateway.verify_recovery()
            pipeline.verify(OUTPUT)
            pipeline.run_child(OUTPUT, "teacher-replay", Path(gateway.__file__).resolve(), ["--stage", "replay"])
            pipeline.run_child(OUTPUT, "audit-teachers", Path(gateway.__file__).resolve(), ["--stage", "audit"], cpu=True)
            pipeline.prerequisite(OUTPUT)
            for stage, cpu in (("replay", False), ("audit-replay", True), ("confirm", False)):
                pipeline.run_child(OUTPUT, stage, Path(pipeline.__file__).resolve(), ["--stage", stage, "--output", str(OUTPUT)], cpu=cpu)
            gate = pipeline.read(pipeline.CONFIRMATION / "gate.json")
            if (gate["passed"] or gate["selected_step"] != 0 or gate["confirmation_evaluated"]
                    or gate["reason"] != pipeline.no_confirmation_reason({"step": 0}, {})
                    or pipeline.CAMPAIGN.exists() or pipeline.REVIEW.exists()):
                raise ValueError("Preserve the original nonselection and leave confirmation and real benchmarks unused")
            for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth"):
                if (pipeline.training.OUTPUT / name).exists():
                    raise ValueError("No unqualified checkpoint may be selected or promoted")
            gateway.verify_recovery()
            pipeline.verify(OUTPUT)
            result = {"complete": True, "gate": gate, "training_audit_sha256": TRAINING_AUDIT_SHA256,
                      "teacher_audit_sha256": sha256(pipeline.TEACHERS / "audit.json"),
                      "endpoint_replay_audit_sha256": sha256(pipeline.REPLAY / "audit.json"),
                      "campaign": None, "review": None, "no_inference_ensemble": True, "goal_success_not_assessed": True}
            write_json(OUTPUT / "result.json", result)
            write_json(OUTPUT / "recovery_audit.json", {"complete": True, "original_failure_preserved": True,
                       "training_restarted": False, "teacher_predictions_repeated": False, "scientific_configuration_changed": False,
                       "original_full_training_audit_reused_unchanged": True, "all_qualification_gates_unchanged": True,
                       "result_sha256": sha256(OUTPUT / "result.json"),
                       "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json")})
            pipeline.status(OUTPUT, "complete", confirmed=False, original_failure_preserved=True)
        except BaseException as error:
            pipeline.status(OUTPUT, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    recover()
