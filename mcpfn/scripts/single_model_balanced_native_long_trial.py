"""Integrate the longer fixed budget without changing V55's science or qualification."""

import inspect
from math import prod

import train_single_matrixformer_balanced_native_quarter as previous_training
import single_model_balanced_native_quarter_trial as predecessor
import single_model_balanced_native_long_budget as budget
import single_model_balanced_native_long_storage as component_storage
import single_model_all_imputation_trial as reference_template
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
wrap_cli = predecessor.wrap_cli
require_probe = predecessor.require_probe
POLICY = budget.POLICY + (
    " One77,998,496-parameter network,238 trainable tensors and16 frozen buffers; unchanged "
    "single-network inference. Native/structured/rating masses.5/.25/.25, standardized teacher "
    "Huber20/delta1, native V29 retention, fixed FP32 half V8/V29 structured retention and no "
    "rating retention. Fresh5e-6 AdamW, quarter projected displacement,.1 inward margin,84 "
    "raw gradients,30 grouped protections,2x norm cap and exact rollback. Keep the accepted "
    "original quarter-step feasibility evidence; no altered computation needs a new consumed "
    "probe. Preserve18/6/50 GiB reserves and prospective4096 MiB/64 MiB/192 GiB training "
    "allowances,24 BLAS/4 Torch threads and.9 allocator cap. Require complete closed-V55 "
    "attribution and independently verified byte-preserving capacity preparation before "
    "separate registration. Require the full480-comparison real review,392 required/18 primary, "
    "after fresh confirmation. Scratch is machine-local, not external backup."
)
QUALIFICATION = replace_once(previous_training.QUALIFICATION, "all64 attempts", "all256 attempts")
QUALIFICATION = replace_once(QUALIFICATION, "1368 original/V29-parent/development grids",
    "1824 original/V29-parent/development grids")
QUALIFICATION = replace_once(QUALIFICATION, "boundaries0/10/25/50/64",
    "boundaries0/10/25/50/64/128/192/256")
REFERENCE_SOURCE = budget.adapt_reference_storage(inspect.getsource(reference_template.reference_storage))
exec(compile(REFERENCE_SOURCE, __file__, "exec"), globals())


def reference_bytes(batches):
    accounting = reference_storage(batches)
    return sum(value for key, value in accounting.items() if key not in {"tasks", "cells"})


def planning_batches(training):
    return budget.batches(training)


def storage_evidence(capacity, receipt_sha256):
    if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64:
        raise ValueError("Require the completed independent long-budget capacity receipt before registration")
    return capacity.require_completed(receipt_sha256)


def adapt_training(source):
    source = predecessor.adapt_training(source)
    source = replace_once(source, "STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)",
        "STEPS, BOUNDARIES = 256, (0, 10, 25, 50, 64, 128, 192, 256)")
    return replace_once(source, 'or config["native_coverage"] != native.SPECIFICATION',
        'or config["native_coverage"] != native.SPECIFICATION\n'
        '            or config["duration_extension"] != budget.SPECIFICATION')


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    source = replace_once(source, 'len(fresh) != 3868', 'len(fresh) != 14620')
    source = replace_once(source, 'Reserve all3584 fresh data/backward seeds',
        'Reserve all14336 fresh data/backward seeds')
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "duration_extension": budget.SPECIFICATION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_balanced_native_long_budget.py", '
        'REPO / "tests/test_single_model_balanced_native_long_trial.py", '
        'REPO / "tests/test_single_model_balanced_native_long_full_audit.py", '
        'REPO / "tests/test_single_model_balanced_native_long_pipeline.py", '
        'REPO / "tests/test_single_model_balanced_native_long_capacity.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    source = replace_once(source, 'import single_model_balanced_native_quarter_storage as diagnostics',
        'import single_model_balanced_native_long_storage as diagnostics')
    source = replace_once(source, 'import single_model_all_imputation_blend_full_replay as blend_replay',
        'import single_model_balanced_native_long_blend_replay as blend_replay')
    source = replace_once(source, 'TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION',
        'TRIAL_QUALIFICATION = all_trial.QUALIFICATION')
    for old, new in (
        ('"metadata_and_audit_allowance": 256 * 1024**2', '"metadata_and_audit_allowance": 1024 * 1024**2'),
        ('"prediction_task_allowance": 2 * 1024**3', '"prediction_task_allowance": 8 * 1024**3'),
        ('"logs_and_registration_allowance": 16 * 1024**2', '"logs_and_registration_allowance": 64 * 1024**2'),
    ):
        source = replace_once(source, old, new)
    if source.count('previous_training.batches()') != 3:
        raise ValueError("Expand all three complete-grid storage estimates, never just one reference role")
    return source.replace('previous_training.batches()', 'all_trial.planning_batches(previous_training)')


def adapt_pipeline(source):
    source = replace_once(source, 'audit["through_step"] != 64', 'audit["through_step"] != training.STEPS')
    source = replace_once(source, 'audit["attempted_batches"] != 64', 'audit["attempted_batches"] != training.STEPS')
    source = replace_once(source, 'audit["prediction_grids"] != 1368',
        'audit["prediction_grids"] != (4 + len(training.BOUNDARIES)) * 152')
    source = replace_once(source, '64 attempts and1368 reference/parent/development grids',
        '256 attempts and1824 reference/parent/development grids')
    return replace_once(source, 'Wait for the exact V56 trainer/auditor chain.',
        'Wait for the exact V56 trainer/auditor chain with the prospectively fixed256-attempt budget.')


def adapt_pipeline_recipe(source):
    return replace_once(source, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
        'ADAPTED_SOURCE = cohort_trial.adapt_pipeline(ADAPTED_SOURCE)\n'
        'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')
