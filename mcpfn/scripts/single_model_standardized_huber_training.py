"""Replay original teacher and retention components with standardized teacher Huber."""

from collections import Counter
from pathlib import Path
import types

import single_model_all_imputation_column_mean_training as predecessor
import single_model_standardized_teacher_huber as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
EVALUATION_BASELINE = predecessor.EVALUATION_BASELINE
EVALUATION_BASELINE_SHA256 = predecessor.EVALUATION_BASELINE_SHA256
POLICY_TEXT = (
    'Full-trial integration of standardized teacher-mean Huber on all imputation queries. '
    + cohort_retention.POLICY
    + ' Preserve every original teacher grid, supervised/retention weight and scale1 gradient. '
    'Replay26 retention targets/50 components per admitted batch,24 V8/26 V29, with no general '
    'retention inference. Preserve original qualification and teacher-component checks. '
    'No historical full-autograd or optimizer-trajectory claim. Require separate trial registration.'
)
for name in ('reference_identities', 'verify_references', 'original_identity', 'original_cohort_identity',
        'native_cohort_identity', 'cohort_identity', 'validate_saved_components', 'require_report_components',
        'expected_component_counts'):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError('Trial callbacks must bind unchanged references and standardized mean geometry')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    'single_model_all_imputation_column_mean as retention', 'single_model_standardized_teacher_huber as retention')
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, 'POLICY', POLICY_TEXT)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, 'exec'), globals())
