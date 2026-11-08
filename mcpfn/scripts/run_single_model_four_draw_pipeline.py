"""Qualify the sole V57 checkpoint with every original synthetic and real comparison."""

import hashlib
import inspect
import types

import run_single_model_balanced_native_long_pipeline as blueprint
import run_single_model_legacy_strata_pipeline as admission_pipeline
import train_single_matrixformer_four_draw as training
import audit_single_model_four_draw as training_audit
import single_model_four_draw_full_storage as diagnostics
import single_model_four_draw_protection as paired
import single_model_four_draw_trial_retention as retention_training
import single_model_four_draw_trial_blend_replay as blend_replay
import single_model_four_draw_fixed_trial as fixed_trial
from single_model_four_draw_full_integration import clone_owned, function_source
from single_model_registered_adaptation import replace_assignment, replace_once


clone_owned(blueprint, globals())
MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in blueprint.STAGE_WRITE_BUDGETS}
AUDITOR_COMMAND = "python scripts/audit_single_model_four_draw.py"
POLICY_TEXT = training.POLICY + (
    " Qualify only the registered same-PID/start trainer-to-auditor chain. Independently audit "
    "all128 attempts, every admitted56-task batch,48 structured admission draws per attempt, "
    "every gradient transfer and1824 full reference/parent/development grids. Reconstruct the "
    "last actual quarter-step endpoint and replay56 or112 original endpoint grids, then the "
    "sole selected152-case state. An all-data-rejected trial proves unchanged initialization "
    "without neural endpoint replay. Never reselect, repeat confirmation or fabricate outputs."
)
ADAPTED_SOURCE = blueprint.ADAPTED_SOURCE
for before, after in (
        ('import audit_single_model_balanced_native_long as training_audit', 'import audit_single_model_four_draw as training_audit'),
        ('import train_single_matrixformer_balanced_native_long as training', 'import train_single_matrixformer_four_draw as training'),
        ('expected_count = 28 if payload["record"]["point_after"] is None else 56',
         'expected_count = 56 if payload["record"]["point_after"] is None else 112'),
        ('test_single_model_balanced_native_long_trial.py', 'test_single_model_four_draw_full_trial.py'),
        ('test_single_model_balanced_native_long_pipeline.py', 'test_single_model_four_draw_pipeline.py'),
):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, before, after)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "balanced_native_long_v56_{suffix}"', f'ROOT / "four_draw_v57_{suffix}"')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
replay_actual, audit_replay_actual, pipeline_main = replay, audit_replay, main
for name in ("replay", "audit_replay"):
    exec(compile(inspect.getsource(getattr(admission_pipeline, name)), __file__, "exec"), globals())
for name, before, after in (("require_compact_audit", 24, 48), ("materialize_endpoint_tasks", 28, 56),
        ("check_materialized_task", 28, 56)):
    function = getattr(blueprint, name)
    if function.__code__.co_consts.count(before) != 1 or function.__closure__ is not None:
        raise ValueError("Change only the explicit task/draw count in the complete final callback")
    constants = tuple(after if value == before else value for value in function.__code__.co_consts)
    cloned = types.FunctionType(function.__code__.replace(co_consts=constants), globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
CHAIN_SOURCE = function_source(blueprint.SOURCE, "chain_is_live")
CHAIN_SOURCE = replace_once(CHAIN_SOURCE, "train_single_matrixformer_balanced_native_long.py", "train_single_matrixformer_four_draw.py")
exec(compile(CHAIN_SOURCE, __file__, "exec"), globals())
REGISTER_CHAIN_SOURCE = inspect.getsource(admission_pipeline.chain_template.register_chain_identity)
for before, after in (("train_single_matrixformer_compact_continuation.py", "train_single_matrixformer_four_draw.py"),
        ("audit_single_model_compact_continuation.py", "audit_single_model_four_draw.py"), ("V23 one-shot", "V57 one-shot")):
    REGISTER_CHAIN_SOURCE = replace_once(REGISTER_CHAIN_SOURCE, before, after)
exec(compile(REGISTER_CHAIN_SOURCE, __file__, "exec"), globals())


def main():
    with training.arithmetic_recovery.previous.original_arithmetic():
        pipeline_main()


if __name__ == "__main__":
    main()
