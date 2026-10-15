"""Qualify the reconstructed V22 candidate with the original unconsumed confirmation cases."""

import hashlib

import run_single_model_teacher_continuation_pipeline as baseline
import replay_single_model_cached_continuation_teachers as teacher_replay
from single_model_registered_adaptation import replace_once, replace_assignment


TEACHERS = teacher_replay.OUTPUT
MINIMUM_FREE = 18 * 1024**3
STAGE_WRITE_BUDGETS = {name: amount * 1024**2 for name, amount in {
    "training_audit": 32, "teacher-replay": 32, "audit-teachers": 16, "replay": 512,
    "audit-replay": 16, "confirm": 2560, "audit-confirmation": 400, "benchmarks": 512, "review": 16,
}.items()}
POLICY_TEXT = (baseline.POLICY + " The original failed run and unused qualification stay immutable. Qualify only "
    "the separately registered bounded cached51-64 reconstruction. Bind its exact51-63 record comparisons and "
    "explicitly unmatched original64 outcome, use original absolute task-artifact paths and corrected84-row/"
    "28-row proposal-prefix storage. No changed scientific thresholds, fallback, resampling, new confirmation "
    "cases or interpretation of the scalar replay as a full-gradient trajectory audit. Reserve declared "
    "stage write budgets above an18 GiB free-space floor; stop before a stage that cannot fit, without "
    "discarding evidence or silently repeating completed evaluation.")


def require_stage_capacity(label):
    if label not in STAGE_WRITE_BUDGETS:
        raise ValueError("Every qualification stage needs a declared write budget")
    free = shutil.disk_usage(ROOT).free
    if free - STAGE_WRITE_BUDGETS[label] < MINIMUM_FREE:
        raise OSError(f"Preserve18 GiB before {label}: free={free}, budget={STAGE_WRITE_BUDGETS[label]}")


def original_reservation_hashes():
    baseline.verify(baseline.OUTPUT)
    directories = (baseline.OUTPUT, baseline.TEACHERS, baseline.REPLAY, baseline.CONFIRMATION)
    if any(baseline.read(directory / "status.json")["state"] != "registered" for directory in directories):
        raise ValueError("Original qualification and every confirmation case must remain unused")
    return {str(directory / name): baseline.sha256(directory / name) for directory in directories
            for name in ("config.json", "registration.json", "status.json")}


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_teacher_continuation as training_audit':
        'import audit_single_model_cached_continuation as training_audit',
    'import train_single_matrixformer_teacher_continuation as training':
        'import recover_single_model_cached_continuation as training',
    'OUTPUT = ROOT / "teacher_continuation_v22_pipeline"':
        'OUTPUT = ROOT / "teacher_continuation_v22_cached_recovery_v1_pipeline"',
    'AUDIT = ROOT / "teacher_continuation_v22_audit"':
        'AUDIT = ROOT / "teacher_continuation_v22_cached_recovery_v1_audit"',
    'REPLAY = ROOT / "teacher_continuation_v22_endpoint_replay"':
        'REPLAY = ROOT / "teacher_continuation_v22_cached_recovery_v1_endpoint_replay"',
    'CONFIRMATION = ROOT / "teacher_continuation_v22_confirmation"':
        'CONFIRMATION = ROOT / "teacher_continuation_v22_cached_recovery_v1_confirmation"',
    'CAMPAIGN = ROOT / "teacher_continuation_v22_benchmarks"':
        'CAMPAIGN = ROOT / "teacher_continuation_v22_cached_recovery_v1_benchmarks"',
    'REVIEW = ROOT / "teacher_continuation_v22_review"':
        'REVIEW = ROOT / "teacher_continuation_v22_cached_recovery_v1_review"',
    '    inputs = [training.OUTPUT / "config.json", training.OUTPUT / "registration.json",':
        '    recovery_inputs = training.audit_recovery()\n'
        '    recovery_inputs.update(original_reservation_hashes())\n'
        '    inputs = [training.OUTPUT / "config.json", training.OUTPUT / "registration.json",',
    '"input_sha256": {str(path): sha256(path) for path in inputs},':
        '"input_sha256": {**recovery_inputs, **{str(path): sha256(path) for path in inputs}},',
    '    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())':
        '    sources.add(REPO / "tests/test_single_model_cached_continuation_pipeline.py")\n'
        '    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())',
    '    teacher_replay.verify(TEACHERS)\n    return config':
        '    if config["stage_write_budget_bytes"] != STAGE_WRITE_BUDGETS or config["minimum_free_bytes"] != MINIMUM_FREE:\n'
        '        raise ValueError("Preserve the registered stage storage budgets")\n'
        '    original_reservation_hashes()\n    teacher_replay.verify(TEACHERS)\n    return config',
    '"endpoint_replay": str(REPLAY), "confirmation": str(CONFIRMATION), "campaign": str(CAMPAIGN), "review": str(REVIEW)})':
        '"endpoint_replay": str(REPLAY), "confirmation": str(CONFIRMATION), "campaign": str(CAMPAIGN), "review": str(REVIEW),\n'
        '               "stage_write_budget_bytes": STAGE_WRITE_BUDGETS, "minimum_free_bytes": MINIMUM_FREE})',
    'def run_child(output, label, script, arguments, cpu=False):\n    command =':
        'def run_child(output, label, script, arguments, cpu=False):\n    require_stage_capacity(label)\n    command =',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
