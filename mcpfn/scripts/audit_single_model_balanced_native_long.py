"""Audit every V56 task and original endpoint without claiming historical autograd replay."""

import audit_single_model_balanced_native_quarter as audit_blueprint
from audit_single_model_balanced_native_quarter import admission_source
import single_model_fixed_quarter_trial as fixed_trial
import single_model_balanced_native_long_retention as retention_training
import single_model_balanced_native_long_blend_replay as blend_replay
import single_model_balanced_native_long_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


FULL_AUDIT_SOURCE = audit_blueprint.FULL_AUDIT_SOURCE
for old, new in {
    'train_single_matrixformer_balanced_native_quarter': 'train_single_matrixformer_balanced_native_long',
    'balanced_native_quarter_v55': 'balanced_native_long_v56',
    'V55': 'V56',
}.items():
    if old not in FULL_AUDIT_SOURCE:
        raise ValueError(f'Missing long-budget full-audit adaptation: {old}')
    FULL_AUDIT_SOURCE = FULL_AUDIT_SOURCE.replace(old, new)
FULL_AUDIT_SOURCE = replace_once(FULL_AUDIT_SOURCE,
    'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
    "ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'full64-attempt qualification audit', 'full256-attempt qualification audit')\n"
    'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')
exec(compile(FULL_AUDIT_SOURCE, __file__, 'exec'), globals())
