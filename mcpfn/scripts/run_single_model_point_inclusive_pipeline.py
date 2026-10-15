"""Qualify the half-rate point-inclusive single model with unchanged complete gates."""

import hashlib

import run_single_model_structured_teacher_pipeline as baseline
import replay_single_model_point_inclusive_teachers as teacher_replay
from single_model_registered_adaptation import replace_once, replace_assignment


TEACHERS = teacher_replay.OUTPUT
POLICY_TEXT = (baseline.POLICY.replace("V20 attempts", "V21 attempts")
               + " Audit point-inclusive1e-5 proposals and the lossless shared-gradient endpoint. "
                 "This changes training only, never any development, confirmation or frozen real qualification gate.")


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_structured_teacher as training_audit': 'import audit_single_model_point_inclusive_training as training_audit',
    'import train_single_matrixformer_structured_teacher as training': 'import train_single_matrixformer_point_inclusive as training',
    'OUTPUT = ROOT / "structured_teacher_v20_pipeline"': 'OUTPUT = ROOT / "point_inclusive_v21_pipeline"',
    'AUDIT = ROOT / "structured_teacher_v20_audit"': 'AUDIT = ROOT / "point_inclusive_v21_audit"',
    'REPLAY = ROOT / "structured_teacher_v20_endpoint_replay"': 'REPLAY = ROOT / "point_inclusive_v21_endpoint_replay"',
    'CONFIRMATION = ROOT / "structured_teacher_v20_confirmation"': 'CONFIRMATION = ROOT / "point_inclusive_v21_confirmation"',
    'CAMPAIGN = ROOT / "structured_teacher_v20_benchmarks"': 'CAMPAIGN = ROOT / "point_inclusive_v21_benchmarks"',
    'REVIEW = ROOT / "structured_teacher_v20_review"': 'REVIEW = ROOT / "point_inclusive_v21_review"',
    'REPO / "tests/test_single_model_structured_teacher_training.py"': 'REPO / "tests/test_single_model_point_inclusive_training.py"',
    'REPO / "tests/test_single_model_structured_teacher_pipeline.py"': 'REPO / "tests/test_single_model_point_inclusive_pipeline.py"',
    'REPO / "tests/test_single_model_training_teacher_replay.py"': 'REPO / "tests/test_single_model_point_inclusive_teachers.py"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
