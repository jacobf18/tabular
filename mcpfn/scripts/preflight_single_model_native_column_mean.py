"""One fresh native-column teacher-mean feasibility update after closed V47."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import analyze_single_model_general_native_development as development
import summarize_single_model_general_native_outcome as outcome
import preflight_single_model_general_native_retention as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_native_column_mean as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = outcome.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
previous_probe = probe_blueprint.previous_probe
diagnosis = probe_blueprint.diagnosis
DEVELOPMENT_HASHES = {
    "config.json": "36f31a5ac16f58000e12bed959fda3f2d1d3f469fa93f56ff0594c05719e068a",
    "summary.json": "69795c68753c7762a1575f0597d85df113f36ce3ed39fa186df8a8004897356c",
    "audit.json": "c17bd6606868cf1e3c31f70cfc3a314879ff220db8471e23ccf08b8da5c72fd7",
}
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after V47's independently audited "
    "negative closure and complete all-five-boundary development decomposition. Start from actual "
    "unchanged V29, not the ineligible V47 tail or another probe. " + cohort_retention.POLICY
    + " Keep77,998,496 trainable parameters/238 tensors,16 frozen buffers,84 raw gradients,30 "
    "grouped protections,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm bound and exact "
    "rollback. Independently replay52 imputation teacher component grids and26 retention grids "
    "(two general-native and24 V8-structured). Preserve24 BLAS/4 Torch threads,.9 allocator cap, "
    "18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB probe budgets. No new confirmation, full trial, "
    "seed retry, warm start from this probe, changed evaluation or benchmark-success claim. "
    "Keep all original22 development guards,32 fresh confirmation requirements and full real review."
)


def require_completed_v47():
    documents, boundaries, comparisons, trace, evidence = outcome.require_closed()
    paths = {str(development.OUTPUT / name): digest for name, digest in DEVELOPMENT_HASHES.items()}
    verify_hashes(paths)
    summary = read(development.OUTPUT / "summary.json")
    audit_record = read(development.OUTPUT / "audit.json")
    config = read(development.OUTPUT / "config.json")
    if (read(development.OUTPUT / "status.json")["state"] != "complete"
            or summary["complete"] is not True or audit_record["complete"] is not True
            or summary["cases"] != 760 or summary["imputation_columns"] != 135775
            or summary["empty_query_columns"] != 21835 or summary["stored_metric_checks"] != 8160
            or summary["boundaries"] != [0, 10, 25, 50, 64]
            or audit_record["case_metric_checks"] != 2880 or audit_record["imputation_case_boundaries"] != 680
            or audit_record["column_boundaries"] != 135775
            or audit_record["summary_sha256"] != DEVELOPMENT_HASHES["summary.json"]
            or summary["config_sha256"] != DEVELOPMENT_HASHES["config.json"]
            or any(summary[key] is not False for key in ("cuda_initialized", "inference_performed", "training_performed",
                "confirmation_evaluated", "checkpoint_selection_performed", "gate_changed", "real_data_used"))
            or audit_record["csv_regrouping_recomputed"] is not True
            or audit_record["input_and_source_hashes_rechecked"] is not True
            or audit_record["cuda_initialized"] is not False or audit_record["raw_grid_replay_performed"] is not False):
        raise ValueError("Require the complete unchanged V47 diagnosis without inference, omissions or reselection")
    verify_hashes(summary["artifact_sha256"])
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    evidence.update(paths)
    evidence.update({str(development.OUTPUT / "status.json"): sha256(development.OUTPUT / "status.json")})
    evidence.update(summary["artifact_sha256"])
    return evidence


for name in ("reference_identities", "replay_cohort_targets"):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Bind reference identities and complete replay only to this new probe")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


NATIVE_PROBE_SOURCE = probe_blueprint.PROBE_SOURCE
for old, new in {
    "single_model_general_native_retention as retention": "single_model_native_column_mean as retention",
    "general_native_retention_preflight_v1": "native_column_mean_preflight_v1",
    "1201100901, 1202100901": "1231100901, 1232100901",
    "require_completed_v46": "require_completed_v47",
}.items():
    if old not in NATIVE_PROBE_SOURCE:
        raise ValueError(f"Missing explicit native-column probe adaptation: {old}")
    NATIVE_PROBE_SOURCE = NATIVE_PROBE_SOURCE.replace(old, new)
NATIVE_PROBE_SOURCE = replace_assignment(NATIVE_PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
registration_patches = [
    ('config = {"policy": POLICY,', 'config = {"policy": POLICY, "native_teacher_mean": retention.MEAN_SPECIFICATION,'),
    ('    sources.update(Path(module.__file__).resolve()',
     '    sources.update({REPO / "tests/test_single_model_native_column_mean.py", '
     'REPO / "tests/test_single_model_native_column_mean_preflight.py"})\n'
     '    sources.update(Path(module.__file__).resolve()'),
]
registration_source = "\n".join(f"REGISTER_SOURCE = replace_once(REGISTER_SOURCE, {old!r}, {new!r})" for old, new in registration_patches)
NATIVE_PROBE_SOURCE = replace_once(NATIVE_PROBE_SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
runtime_source = "ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, " + repr('or config["mean_distillation"] != stronger.SPECIFICATION') + ", " + repr(
    'or config["mean_distillation"] != stronger.SPECIFICATION\n'
    '            or config["native_teacher_mean"] != retention.MEAN_SPECIFICATION') + ")"
NATIVE_PROBE_SOURCE = replace_once(NATIVE_PROBE_SOURCE, 'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)',
    runtime_source + '\nADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)')
exec(compile(NATIVE_PROBE_SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Keep all original teacher component replay coverage and capacity")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    stage = parser.parse_args().stage
    with diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if stage == "register":
            register(OUTPUT)
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error), failed_stage=stage)
                raise


if __name__ == "__main__":
    main()
