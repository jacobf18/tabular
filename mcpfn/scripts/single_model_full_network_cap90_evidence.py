"""Bind the failed run, measured resident-memory probe and lossless capacity restoration."""

from pathlib import Path

import torch

import probe_single_model_full_network_resident_memory as memory
import relocate_single_model_failed_initial_checkpoint as relocation


POLICY = ("V35 is a separately registered fresh-seed scientific trial from the same V29 initializer, not "
    "a restart of failed V34. Change only the process GPU allocation fraction from.8 to.9. Preserve the "
    "entire V34 scientific recipe, ordinary inference, priors, teachers, objectives, optimizer and all gates. "
    "The consumed2000x1000 task fits with84 resident gradient rows and four extra parameter vectors at the "
    "initializer; this is not a guarantee of every later shape/state. GPU backward hashes differed between "
    "the two allocation diagnostics despite identical task/teacher bytes; do not claim bitwise GPU-gradient "
    "equivalence or reconstruction of lost step8. Neither diagnostic trained or selected a model. Preserve "
    "all failed-run records and reserved seeds, original storage budgets and reserves.")


def validate_probe(result, status, config_digest, result_digest):
    gradients = result["gradient_records"]
    if (status["state"] != "complete" or status["result_sha256"] != result_digest
            or result["complete"] is not True or result["config_sha256"] != config_digest
            or result["memory_fraction"] != .9 or result["parent_sha256"] != memory.training.PARENT_SHA256
            or result["historical_task_and_teacher_digest_equal"] is not True
            or result["reconstructed_step8_state"] is not False or result["gpu_gradient_rows"] != 84
            or result["gpu_gradient_bytes"] != 26207494656 or result["headroom_bytes"] != 1247975936
            or len(gradients) != 3 or [record["kind"] for record in gradients] != ["composite", "rmse", "common_nrmse"]
            or any(record["parameters"] != 77998496 for record in gradients)
            or result["frozen"]["changed_trainable_tensors"] or result["frozen"]["frozen_tensors"] != 16
            or result["frozen"]["unselected_parameters_and_all_buffers_bitwise_unchanged"] is not True
            or not .8 * result["gpu_total_gib"] < result["peak_allocated_gib"] <= result["peak_reserved_gib"] < .9 * result["gpu_total_gib"]
            or result["training_updates_performed"] != 0
            or any(result[key] is not False for key in ("optimizer_stepped", "checkpoint_written", "validation_evaluated",
                "real_benchmarks_evaluated", "all_training_tasks_memory_qualified"))):
        raise ValueError("Require the completed full resident-memory measurement without overstating its scope")


def require_evidence():
    training = memory.training
    result = training.read(memory.OUTPUT / "result.json")
    status = training.read(memory.OUTPUT / "status.json")
    validate_probe(result, status, memory.sha256(memory.OUTPUT / "config.json"), memory.sha256(memory.OUTPUT / "result.json"))
    if Path(f'/proc/{status["pid"]}').exists():
        raise ValueError("Stop the completed probe before a fresh scientific trial")
    paths = []
    for directory in (memory.template.OUTPUT, memory.OUTPUT):
        config = training.read(directory / "config.json")
        training.verify_hashes(config["input_sha256"])
        training.verify_hashes(config["source_sha256"])
        paths.extend(directory / name for name in ("config.json", "registration.json", "status.json", "result.json"))
    receipt = training.read(relocation.OUTPUT / "independent_recheck.json")
    current = training.read(relocation.OUTPUT / "status.json")
    config = training.read(relocation.OUTPUT / "config.json")
    if (current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or receipt["complete"] is not True or receipt["canonical_artifacts_checked"] != 1
            or receipt["failed_training_run_qualified"] is not False
            or receipt["zero_update_model_equals_retained_parent"] is not True
            or receipt["selected_model_only_checkpoints_retained_on_original_volume"] is not True
            or receipt["config_sha256"] != memory.sha256(relocation.OUTPUT / "config.json")):
        raise ValueError("Bind the losslessly preserved zero-update duplicate without qualifying failed training")
    training.verify_hashes(receipt["artifact_sha256"])
    for name in ("source_sha256", "input_sha256"):
        training.verify_hashes(config[name])
    source = training.OUTPUT / "latest_checkpoint.pth"
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    relocation.checkpoint_identity(payload, training.read(training.OUTPUT / "status.json"),
        relocation.OWNERS["full_network_v34"], memory.sha256(training.OUTPUT / "config.json"))
    paths.extend(relocation.OUTPUT / name for name in ("config.json", "status.json", "independent_recheck.json"))
    return {str(path): memory.sha256(path) for path in paths}
