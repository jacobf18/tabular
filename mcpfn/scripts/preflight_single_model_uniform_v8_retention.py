"""Probe a uniform V8 reference only after V43's complete negative qualification."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import diagnose_single_model_family_balanced_initializer as diagnosis
import preflight_single_model_family_balanced_retention as previous_probe
from preflight_single_model_family_balanced_retention import blend_replay, replace_function
import single_model_uniform_v8_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


ARITHMETIC = {"blas_threads": 24, "torch_threads": 4}
ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
CLOSED_REPORTS = {
    "family_balanced_retention_v43/config.json":
        "275ef4280315099d0dc66ba8eabcb6056ca02d12f3265b2c5e52dd0daa4b88b5",
    "family_balanced_retention_v43_audit/audit.json":
        "96e43a7033c9bc40580446e14bb0873e16ccf575dc34d62376f5fc1cd1204ba1",
    "family_balanced_retention_v43_endpoint_replay/audit.json":
        "15eda3e41112f3d21457262c74426ab73edc74ff4adfa3ea7226910f1867ed0b",
    "family_balanced_retention_v43_confirmation/gate.json":
        "2cd03d3a11b59c10fd38ad562b468ed0597df82ac1678cf55c14048fb8b78ad5",
    "family_balanced_retention_v43_confirmation/audit.json":
        "30d75cfbbf2bc904625d80cbaed717db01e4d92e074d30354b548da112ce3c44",
    "family_balanced_retention_v43_orphan_recovery_v1/result.json":
        "24db7b0d712033174d05443eb29e860f99c7b949e08534f980528a2d8e29ffd3",
    "family_balanced_retention_v43_initializer_diagnosis_v1/audit.json":
        "2cb6360ff27803119602ed80bc420b83e1fd20bc6063feddd88fd5ae842f718d",
}
POLICY_TEXT = ("One fresh28-task full-network feasibility probe after V43's fully audited six-failure "
    "confirmation and284-case initializer diagnosis. " + cohort_retention.POLICY
    + " Keep V29 step25/98 inherited updates as initializer, never V43 or a probe endpoint. Keep all original "
    "238 tensors/77,998,496 trainable parameters,16 frozen buffers,84 raw gradients,30 protections, fresh5e-6 "
    "AdamW,.1 inward margin,fixed-half displacement,2x norm bound and exact rollback. Every one of26 original "
    "imputation tasks uses V8; both rating tasks exclude retention. Keep exact equal-family retention "
    "multipliers and all original supervised task weights. Archive every reference/weighting decision and "
    "independently replay all complete reference grids and original teacher/component predictions. Preserve "
    "24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB allowances. "
    "No new prior, altered target, metric change, omitted column, scientific retry, relaxed guard or inference "
    "ensemble. Acceptance proves feasibility only, not full training, confirmation or real-benchmark parity.")


def validate_completed(training_audit, replay, gate, confirmation_audit, closure, diagnostic):
    expected = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 56, "training_episodes": 1792,
        "prediction_grids": 1368, "full_network_training": True, "fixed_half_training": True,
        "displacement_fraction": .5, "independently_regenerated_training_tasks": 1792,
        "independently_regenerated_rating_contexts": 128, "independently_regenerated_structured_draws": 1536,
        "full_optimizer_trajectory_replayed": False, "autograd_replayed": False}
    if (any(training_audit.get(key) != value for key, value in expected.items())
            or training_audit["best"]["step"] != 25 or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or replay["complete"] is not True or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or gate["passed"] is not False or gate["selected_step"] != 25 or gate["cases"] != 284
            or gate["confirmation_evaluated"] is not True or confirmation_audit["complete"] is not True
            or confirmation_audit["prediction_grids"] != 1136 or confirmation_audit["cuda_initialized"] is not False
            or closure["complete"] is not True or closure["gate"] != gate
            or closure["campaign"] is not None or closure["review"] is not None
            or diagnostic["complete"] is not True or diagnostic["prediction_grids"] != 284
            or len(diagnostic["comparisons"]) != 22 or diagnostic["post_hoc_consumed_bank"] is not True
            or any(diagnostic[key] is not False for key in ("cuda_initialized", "training_performed",
                "confirmation_repeated", "checkpoint_selection_performed", "real_benchmarks_evaluated"))):
        raise ValueError("Require V43's complete negative qualification and full post-hoc initializer diagnosis")


def require_completed_v43():
    diagnosis.verify()
    identities = {str(ROOT / name): digest for name, digest in CLOSED_REPORTS.items()}
    verify_hashes(identities)
    reports = [read(ROOT / name) for name in CLOSED_REPORTS if not name.endswith("/config.json")]
    validate_completed(*reports)
    directories = (diagnosis.training.OUTPUT, diagnosis.original.REPLAY, diagnosis.original.CONFIRMATION,
        diagnosis.recovery.OUTPUT, diagnosis.OUTPUT,
        diagnosis.original.AUDIT.with_name(diagnosis.original.AUDIT.name + "_task_replay"))
    for directory in directories:
        path = directory / "status.json"
        current = read(path)
        if current["state"] not in ("complete", "training_complete") or diagnosis.original.process_identity(current["pid"]) is not None:
            raise ValueError("All predecessor producers must be stopped before a fresh probe")
        identities[str(path)] = sha256(path)
    for report in (reports[0], reports[1], reports[3], reports[5]):
        verify_hashes(report["artifact_sha256"])
    return identities


for name in ("reference_identities", "replay_cohort_targets"):
    function = getattr(previous_probe, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain predecessor runtime closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

SOURCE = previous_probe.SOURCE
for old, new in {
    "single_model_family_balanced_retention as retention": "single_model_uniform_v8_retention as retention",
    "family_balanced_retention_preflight_v1": "uniform_v8_retention_preflight_v1",
    "1081100901, 1082100901": "1111100901, 1112100901",
    "require_completed_v42": "require_completed_v43",
    "test_single_model_family_balanced_retention.py": "test_single_model_uniform_v8_retention.py",
    "test_single_model_family_balanced_retention_preflight.py": "test_single_model_uniform_v8_retention_preflight.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit uniform-reference probe adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
SOURCE = replace_once(SOURCE,
    'set(record["reference_role"] for task, record, episode, payload in records) != {"v8", "v29"}',
    'set(record["reference_role"] for task, record, episode, payload in records) != {"v8"}')
exec(compile(SOURCE, __file__, "exec"), globals())

unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    stage = parser.parse_args().stage
    with diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if stage == "register":
            register(OUTPUT)
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error), failed_stage=stage)
                raise


if __name__ == "__main__":
    main()
