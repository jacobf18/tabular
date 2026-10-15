"""Test training-only batch admission and independent legacy-stratum protection."""

import hashlib
import inspect

import torch

import preflight_single_model_anchor_small_step as template
from preflight_single_model_anchor_small_step import (
    composite, diagnostic, preceding, rehearsal, replicated, schedule, storage, teacher_policy, top_two,
)
import diagnose_single_model_batch_admission as diagnosis
import single_model_legacy_stratum_protection as paired
import single_model_training_admission as admission
from single_model_structured_mixture import StructuredSampler
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES, PARENT_SHA256 = template.PREFIXES, template.PARENT_SHA256
POLICY_TEXT = ("One fresh28-task optimizer feasibility test from audited but unqualified V27 selected step10, "
    "not incomplete V28 or any preflight endpoint. Preserve V28's terminal failure and all original held-out "
    "qualification rules. " + rehearsal.POLICY + " Override only its legacy paired training guard: "
    + paired.POLICY + " " + admission.POLICY + " Use fresh5e-6 AdamW, original general-imputation and "
    "specialist-rating teachers, original Huber/NLL/CDF/query objectives,62 final-two-block/decoder tensors "
    "and192 bitwise-frozen state tensors in the ordinary77,998,496-parameter inference model. If data is "
    "admitted, preserve all84 raw gradients and independently audit the optimizer, all28 teacher grids and "
    "all before/candidate grids. If rejected, record and independently regenerate the complete data-only "
    "attempt; no model setup, optimizer, teacher replay, metric fabrication or checkpoint exists. No retry, "
    "validation, confirmation, nomination, real benchmark, inference ensemble or promotable endpoint. This "
    "tests feasibility, not causal attribution or accuracy. Preserve18/6/50 GiB reserves and1/0/16 GiB "
    "checkpoint/root/scratch budgets. Scratch is machine-local, not provider-persistent storage or backup.")


def require_predecessor():
    evidence = template.template.require_predecessor()
    if sha256(diagnosis.DIAGNOSIS) != diagnosis.DIAGNOSIS_SHA256:
        raise ValueError("Preserve the original V28 terminal failure")
    failure = read(diagnosis.DIAGNOSIS)
    admission_replay = read(diagnosis.OUTPUT)
    verify_hashes(failure["input_sha256"])
    verify_hashes(admission_replay["input_sha256"])
    verify_hashes(admission_replay["source_sha256"])
    original_config = read(ROOT / "anchor_rehearsal_v28/config.json")
    diagnosis.check_record(admission_replay["record"], original_config["batches"][61])
    if (admission_replay["diagnosis_complete"] is not True
            or admission_replay["matching_cpu_regenerations"] != 2
            or admission_replay["training_performed"] is not False):
        raise ValueError("Require the completed data-only full-batch diagnosis")
    return {**evidence, **failure["input_sha256"],
            str(diagnosis.DIAGNOSIS): diagnosis.DIAGNOSIS_SHA256, str(diagnosis.OUTPUT): sha256(diagnosis.OUTPUT)}


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw"):
    source = inspect.getsource(getattr(template.template.template.storage_template, name))
    if name == "require_space":
        source = replace_once(source, "(SCRATCH.parent, 50, 12)", "(SCRATCH.parent, 50, 16)")
    elif name == "frozen_inventory":
        source = replace_once(source, "len(expected) != 34", "len(expected) != 62")
        source = replace_once(source, "and34 last-block/decoder tensors", "and62 final-two-block/decoder tensors")
    exec(compile(source, __file__, "exec"), globals())

exec(compile(inspect.getsource(template.template.template.expected_role), __file__, "exec"), globals())
for name in ("check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(template.template.template.teacher_template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, "small-step-anchor preflight", "legacy-stratum preflight")
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, "test_single_model_anchor_small_step.py", "test_single_model_legacy_strata_preflight.py")
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'REPO / "tests/test_single_model_anchor_rehearsal.py",',
    'REPO / "tests/test_single_model_training_admission.py", REPO / "tests/test_single_model_legacy_stratum_protection.py",')
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "admission_policy": admission.POLICY, "training_guard_policy": paired.POLICY,')
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())


def admit_batch(output, config, device="cuda"):
    sampler = StructuredSampler(config["base"], device=device)
    checker = admission.TrainingBatchAdmission(sampler, config["batch"])
    write_json(output / "data_admission.json", checker.record)
    if not checker.eligible:
        checker.assert_consumed()
        write_json(output / "result.json", {"training_pid": os.getpid(), "preflight_only": True,
            "data_admission_only": True, "reason": "data_admission_rejected", "accepted_updates": 0,
            "optimizer_proposal_performed": False, "model_setup_performed": False,
            "admission_sha256": sha256(output / "data_admission.json"), "goal_success_not_assessed": True})
        status(output, "awaiting_data_admission_audit")
    return checker


def audit_admission(output, config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Admission regeneration must be CPU-only")
    saved = read(output / "data_admission.json")
    result = read(output / "result.json")
    sampler = StructuredSampler(config["base"], device="cpu")
    replay = admission.TrainingBatchAdmission(sampler, config["batch"])
    if (replay.record != saved or result["admission_sha256"] != sha256(output / "data_admission.json")
            or result["training_pid"] == os.getpid()
            or saved["task_count"] != 28 or saved["structured_draws"] != 24
            or saved["deferred_task_indices"] != [0, 13, 14, 27]):
        raise ValueError("Independently reproduce the complete registered admission attempt")
    hashes = {str(output / name): sha256(output / name) for name in
              ("data_admission.json", "result.json", "config.json", "registration.json")}
    if saved["eligible"]:
        for row in saved["tasks"]:
            if row["state"] != "structured_draw_inspected":
                continue
            path = output / "training_batch" / f'task_{row["task"]["index"]:02d}.pt'
            cached = torch.load(path, map_location="cpu", weights_only=False)
            episode = training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
            if (cached["task"] != row["task"]
                    or admission.episode_digest(episode) != row["episode_content_sha256"]):
                raise ValueError("Actual captured training inputs must equal the originally admitted draws")
            hashes[str(path)] = sha256(path)
    else:
        replay.assert_consumed()
        forbidden = ("training_batch", "teacher_replay", "parameter_inventory.json", "decoder_inventory.json",
                     "raw_update.pt", "preflight_checkpoint.pth", "before_predictions.json", "candidate_predictions.json")
        if (result["data_admission_only"] is not True or result["reason"] != "data_admission_rejected"
                or result["accepted_updates"] != 0 or result["optimizer_proposal_performed"] is not False
                or result["model_setup_performed"] is not False
                or any((output / name).exists() for name in forbidden) or any(SCRATCH.iterdir())):
            raise ValueError("A data-only rejected attempt cannot contain model, teacher or optimizer evidence")
    return {"complete": True, "eligible": saved["eligible"], "structured_draws": saved["structured_draws"],
            "deferred_task_indices": saved["deferred_task_indices"],
            "rejected_task_indices": saved["rejected_task_indices"], "artifact_sha256": hashes}


teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "anchor_small_step_preflight_v1"': 'OUTPUT = ROOT / "legacy_strata_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_small_step_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_strata_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 621100901, 622100901': 'DATA_SEED, GRADIENT_SEED = 651100901, 652100901',
    'or config["policy"] != POLICY':
        'or config["admission_policy"] != admission.POLICY or config["training_guard_policy"] != paired.POLICY\n'
        '            or config["policy"] != POLICY',
    '    wait_for_gpu(output, True, False)\n    require_space()':
        '    require_space()\n    status(output, "admitting_training_data")\n'
        '    admitted_sampler = admit_batch(output, config)\n'
        '    if not admitted_sampler.eligible:\n        return\n    wait_for_gpu(output, True, False)',
    '        del old_optimizer': '        del old_optimizer\n        sampler = admitted_sampler',
    '        paired.validate_layout(entries, config["batch"]["tasks"], before)':
        '        sampler.assert_consumed()\n        paired.validate_layout(entries, config["batch"]["tasks"], before)',
    'write_json(output / "result.json", {"training_pid": os.getpid(),':
        'write_json(output / "result.json", {"training_pid": os.getpid(), "data_admission_only": False,\n'
        '                    "admission_sha256": sha256(output / "data_admission.json"),',
    '    teacher_evidence = audit_teacher_replay(output, config)':
        '    admission_evidence = audit_admission(output, config)\n    teacher_evidence = audit_teacher_replay(output, config)',
    '    hashes.update(teacher_evidence["artifact_sha256"])':
        '    hashes.update(admission_evidence["artifact_sha256"])\n    hashes.update(teacher_evidence["artifact_sha256"])',
    '"teacher_replay": teacher_evidence,': '"data_admission": admission_evidence, "teacher_replay": teacher_evidence,',
    '# Small-step anchor rehearsal optimizer preflight': '# Independent legacy-stratum optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted, replay_admitted = audit, replay_targets


def replay_targets(output):
    verify(output)
    if read(output / "data_admission.json")["eligible"]:
        replay_admitted(output)
    elif read(output / "status.json")["state"] != "awaiting_data_admission_audit":
        raise ValueError("Rejected data-only attempts have no teacher targets to replay")


def audit(output):
    config = verify(output)
    if read(output / "data_admission.json")["eligible"]:
        audit_admitted(output)
        return
    if read(output / "status.json")["state"] != "awaiting_data_admission_audit":
        raise ValueError("Audit a data-only attempt once, without fabricating a normal optimizer endpoint")
    evidence = audit_admission(output, config)
    verify(output)
    write_json(output / "audit.json", {"complete": True, "data_admission_only": True,
        "data_admission": evidence, "training_tasks": 0, "prediction_grids": 0, "accepted_updates": 0,
        "optimizer_proposal_performed": False, "model_setup_performed": False, "cuda_initialized": False,
        "artifact_sha256": evidence["artifact_sha256"], "goal_success_not_assessed": True})
    write_json(output / "gate.json", {"passed": False, "preflight_only": True,
        "reason": "data_admission_rejected", "audit_sha256": sha256(output / "audit.json"),
        "real_benchmark_success": False})
    status(output, "complete", accepted=False, data_admission_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
