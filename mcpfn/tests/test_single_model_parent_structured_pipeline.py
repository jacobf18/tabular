import inspect
import types

import test_single_model_native_column_pipeline as blueprint
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


SOURCE = blueprint.SOURCE
for old, new in {
    "run_single_model_native_column_mean_pipeline": "run_single_model_parent_structured_pipeline",
    "audit_single_model_native_column_mean": "audit_single_model_parent_structured",
    "train_single_matrixformer_native_column_mean": "train_single_matrixformer_parent_structured",
    "native_column_mean_v48": "parent_structured_v49",
}.items():
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
import run_single_model_parent_structured_pipeline as pipeline

for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",
        "test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged",
        "test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads",
        "test_coordinator_never_owns_cuda_and_child_visibility_is_explicit"):
    function = getattr(blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned
for name in ("test_actual_failed_qualification_consumes_no_confirmation",
        "test_actual_coordinator_stops_before_real_benchmarks_on_failed_gate"):
    function = getattr(blueprint, name)
    cloned = types.FunctionType(function.__code__, {**function.__globals__, "pipeline": pipeline}, name, function.__defaults__)
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_pipeline_binds_every_new_stage_and_preserves_all_frozen_scope():
    assert pipeline.training.OUTPUT.name == "parent_structured_v49"
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.blend_replay is pipeline.training.blend_replay
    assert pipeline.AUDITOR_COMMAND == "python scripts/audit_single_model_parent_structured.py"
    for module in (pipeline, pipeline.training_audit):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert not unresolved_globals(value)
