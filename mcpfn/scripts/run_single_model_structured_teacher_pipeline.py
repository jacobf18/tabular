"""Qualify the structured-general teacher trial using unchanged held-out and real gates."""

import hashlib

import run_single_model_paired_decoder_pipeline as baseline
import replay_single_model_training_teachers as teacher_replay
from single_model_registered_adaptation import replace_once, replace_assignment


TEACHERS = teacher_replay.OUTPUT
POLICY_TEXT = (baseline.POLICY.replace("V19 attempts", "V20 attempts")
               + " Before endpoint qualification, independently replay and CPU-audit all1792 training teacher "
                 "grids from the original three checkpoints. Preserve the identical full-grid bytes in the "
                 "original cached training artifacts without redundant arrays. Teacher routing and new seeds "
                 "change training only; no development, confirmation, original-general safeguard or real gate changes.")


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_paired_decoder as training_audit': 'import audit_single_model_structured_teacher as training_audit',
    'import train_single_matrixformer_paired_decoder as training': 'import train_single_matrixformer_structured_teacher as training',
    'OUTPUT = ROOT / "paired_decoder_v19_pipeline"': 'OUTPUT = ROOT / "structured_teacher_v20_pipeline"',
    'AUDIT = ROOT / "paired_decoder_v19_audit"': 'AUDIT = ROOT / "structured_teacher_v20_audit"',
    'REPLAY = ROOT / "paired_decoder_v19_endpoint_replay"': 'REPLAY = ROOT / "structured_teacher_v20_endpoint_replay"',
    'CONFIRMATION = ROOT / "paired_decoder_v19_confirmation"': 'CONFIRMATION = ROOT / "structured_teacher_v20_confirmation"',
    'CAMPAIGN = ROOT / "paired_decoder_v19_benchmarks"': 'CAMPAIGN = ROOT / "structured_teacher_v20_benchmarks"',
    'REVIEW = ROOT / "paired_decoder_v19_review"': 'REVIEW = ROOT / "structured_teacher_v20_review"',
    '    return config\n': '    teacher_replay.verify(TEACHERS)\n    return config\n',
    '(output, AUDIT, REPLAY, CONFIRMATION, CAMPAIGN, REVIEW)': '(output, AUDIT, REPLAY, CONFIRMATION, CAMPAIGN, REVIEW, TEACHERS)',
    'REPO / "tests/test_single_model_paired_decoder_training.py"': 'REPO / "tests/test_single_model_structured_teacher_training.py"',
    'REPO / "tests/test_single_model_paired_decoder_pipeline.py"':
        'REPO / "tests/test_single_model_structured_teacher_pipeline.py",\n'
        '               REPO / "tests/test_single_model_training_teacher_replay.py"',
    '    status(output, "registered")':
        '    teacher_replay.register(TEACHERS, output / "config.json")\n    status(output, "registered")',
    '    return audit\n':
        '    if audit["teacher_identity_checked_tasks"] != 1792 or audit["training_teacher_policy"] != training.teacher_policy.POLICY:\n'
        '        raise ValueError("Require every original cached training-teacher identity")\n'
        '    teacher_replay.verify_audit(TEACHERS)\n    return audit\n',
    '    for stage, cpu in (("replay", False), ("audit-replay", True), ("confirm", False)):':
        '    run_child(output, "teacher-replay", Path(teacher_replay.__file__).resolve(), ["--stage", "replay"])\n'
        '    run_child(output, "audit-teachers", Path(teacher_replay.__file__).resolve(), ["--stage", "audit"], cpu=True)\n'
        '    for stage, cpu in (("replay", False), ("audit-replay", True), ("confirm", False)):',
    '"campaign": str(CAMPAIGN) if gate["passed"] else None,':
        '"teacher_audit_sha256": sha256(TEACHERS / "audit.json"),\n'
        '               "campaign": str(CAMPAIGN) if gate["passed"] else None,',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
