"""Qualify V47 with unchanged synthetic guards and every frozen real benchmark."""

import run_single_model_stronger_mean_pipeline as pipeline_blueprint
import single_model_all_imputation_blend_full_replay as blend_replay
import single_model_fixed_half_trial as fixed_trial
import single_model_general_native_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC_PATCH = pipeline_blueprint.ARITHMETIC_PATCH
for old, new in (
    ('def run(output):\n    verify(output)',
     'def run(output):\n    if training.torch.cuda.is_initialized():\n'
     '        raise RuntimeError("The persistent coordinator must never initialize CUDA")\n    verify(output)'),
    ('    if cpu:\n        environment["CUDA_VISIBLE_DEVICES"] = ""',
     '    if training.torch.cuda.is_initialized():\n'
     '        raise RuntimeError("Never retain a coordinator GPU allocation while spawning a worker")\n'
     '    if cpu:\n        environment["CUDA_VISIBLE_DEVICES"] = ""\n'
     '    else:\n        environment.pop("CUDA_VISIBLE_DEVICES", None)'),
):
    ARITHMETIC_PATCH = replace_once(ARITHMETIC_PATCH,
        "ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()",
        f"ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, {old!r}, {new!r})\n"
        "ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()")
PIPELINE_ADAPTATION_SOURCE = pipeline_blueprint.ADAPTATION_SOURCE
for old, new in {
    "train_single_matrixformer_stronger_mean": "train_single_matrixformer_general_native_retention",
    "audit_single_model_stronger_mean": "audit_single_model_general_native_retention",
    "single_model_uniform_v8_retention_training": "single_model_general_native_retention_training",
    "stronger_mean_distillation_v46": "general_native_retention_v47",
    "test_single_model_uniform_v8_retention_trial.py": "test_single_model_general_native_trial.py",
    "test_single_model_uniform_v8_retention_pipeline.py": "test_single_model_general_native_pipeline.py",
    "V46": "V47",
}.items():
    if old not in PIPELINE_ADAPTATION_SOURCE:
        raise ValueError(f"Missing explicit native-general qualification adaptation: {old}")
    PIPELINE_ADAPTATION_SOURCE = PIPELINE_ADAPTATION_SOURCE.replace(old, new)
exec(compile(PIPELINE_ADAPTATION_SOURCE, __file__, "exec"), globals())
