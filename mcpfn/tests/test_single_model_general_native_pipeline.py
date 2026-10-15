import inspect
import types

import test_single_model_stronger_mean_pipeline as test_blueprint
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


SOURCE = test_blueprint.SOURCE
for old, new in {
    "run_single_model_stronger_mean_pipeline": "run_single_model_general_native_retention_pipeline",
    "audit_single_model_stronger_mean": "audit_single_model_general_native_retention",
    "train_single_matrixformer_stronger_mean": "train_single_matrixformer_general_native_retention",
    "stronger_mean_distillation_v46": "general_native_retention_v47",
}.items():
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
import run_single_model_general_native_retention_pipeline as pipeline

for name in ("test_actual_prerequisite_requires_all_original_and_added_evidence",
        "test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged",
        "test_pipeline_pins_identical_arithmetic_in_every_child_stage_without_changing_torch_threads"):
    function = getattr(test_blueprint, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_pipeline_preserves_scope_and_resolves_every_stage():
    assert pipeline.training.OUTPUT.name == "general_native_retention_v47"
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.blend_replay is pipeline.training.blend_replay
    assert pipeline.AUDITOR_COMMAND == "python scripts/audit_single_model_general_native_retention.py"
    for module in (pipeline, pipeline.training_audit):
        for value in vars(module).values():
            if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(module):
                assert not unresolved_globals(value)


def test_coordinator_never_owns_cuda_and_child_visibility_is_explicit():
    assert 'if training.torch.cuda.is_initialized():' in pipeline.ADAPTED_SOURCE
    assert 'environment.pop("CUDA_VISIBLE_DEVICES", None)' in pipeline.ADAPTED_SOURCE
    assert 'environment["CUDA_VISIBLE_DEVICES"] = ""' in pipeline.ADAPTED_SOURCE
