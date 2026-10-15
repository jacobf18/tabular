"""Qualify V41 only after every original audit and complete legacy component replay."""

import run_single_model_all_imputation_retention_pipeline as previous_pipeline
import single_model_legacy_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ADAPTATION_SOURCE = previous_pipeline.ADAPTATION_SOURCE
for old, new in {
    "train_single_matrixformer_all_imputation_retention": "train_single_matrixformer_legacy_blend_teacher",
    "audit_single_model_all_imputation_retention": "audit_single_model_legacy_blend_teacher",
    "all_imputation_retention_v40": "legacy_blend_teacher_v41",
    "test_single_model_all_imputation_retention_trial": "test_single_model_legacy_blend_teacher_trial",
    "test_single_model_all_imputation_retention_pipeline": "test_single_model_legacy_blend_teacher_pipeline",
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit V41 qualification adaptation: {old}")
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace("V40", "V41")
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = replace_once(SOURCE, \'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()\',\n'
    '    \'ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, "    retention_training.require_report(training, audit)", \'\n'
    '    \'"    retention_training.require_report(training, audit)\\\\n    blend_replay.require_report(training, audit)")\\n\'\n'
    '    \'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()\')\n'
    'exec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, "exec"), globals())
