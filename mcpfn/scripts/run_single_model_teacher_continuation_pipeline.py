"""Qualify a V20-start continuation without replacing V8 or relaxing any frozen gate."""

import hashlib

import run_single_model_structured_teacher_pipeline as baseline
import replay_single_model_continuation_teachers as teacher_replay
from single_model_registered_adaptation import replace_once, replace_assignment


TEACHERS = teacher_replay.OUTPUT
POLICY_TEXT = (baseline.POLICY.replace("V20 attempts", "V22 attempts")
               + " Initialize from the audited unqualified V20 state, not a changed teacher or baseline. "
                 "Independently rescore its152 extra parent grids for1368 total training-audit grids. "
                 "Preserve shared raw gradients and the original V8 confirmation baseline.")
ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_structured_teacher as training_audit':
        'import audit_single_model_teacher_continuation as training_audit',
    'import train_single_matrixformer_structured_teacher as training':
        'import train_single_matrixformer_teacher_continuation as training',
    'OUTPUT = ROOT / "structured_teacher_v20_pipeline"': 'OUTPUT = ROOT / "teacher_continuation_v22_pipeline"',
    'AUDIT = ROOT / "structured_teacher_v20_audit"': 'AUDIT = ROOT / "teacher_continuation_v22_audit"',
    'REPLAY = ROOT / "structured_teacher_v20_endpoint_replay"': 'REPLAY = ROOT / "teacher_continuation_v22_endpoint_replay"',
    'CONFIRMATION = ROOT / "structured_teacher_v20_confirmation"': 'CONFIRMATION = ROOT / "teacher_continuation_v22_confirmation"',
    'CAMPAIGN = ROOT / "structured_teacher_v20_benchmarks"': 'CAMPAIGN = ROOT / "teacher_continuation_v22_benchmarks"',
    'REVIEW = ROOT / "structured_teacher_v20_review"': 'REVIEW = ROOT / "teacher_continuation_v22_review"',
    'audit["prediction_grids"] != 1216': 'audit["prediction_grids"] != 1368',
    'REPO / "tests/test_single_model_structured_teacher_training.py"':
        'REPO / "tests/test_single_model_teacher_continuation.py"',
    'REPO / "tests/test_single_model_structured_teacher_pipeline.py"':
        'REPO / "tests/test_single_model_teacher_continuation_pipeline.py"',
    'REPO / "tests/test_single_model_training_teacher_replay.py"':
        'REPO / "tests/test_single_model_teacher_continuation_pipeline.py"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
