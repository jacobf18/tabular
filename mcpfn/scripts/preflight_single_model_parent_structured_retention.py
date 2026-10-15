"""One fresh parent-structured retention probe after fully closed negative V48."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import diagnose_single_model_native_column_initializer as diagnosis
import preflight_single_model_native_column_mean as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_parent_structured_retention as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
DIAGNOSTIC_HASHES = {
    "config.json": "9b24898181d3f7751234fb8993e57c63e6b4cc9a260955275f36cf4845572a84",
    "initializer.json": "aaa10397cefb7080010e139b097fc3eaf3b9037f8d56aa6cc3f205dbe77e740b",
    "audit.json": "304392f158e37abab73eb8f1f51a3ac71879b4466fa59ed5f2440ca7eb46e04e",
    "RESULTS.md": "ef93f9aa06f6b47a93c246a741eba0605d3dc8ae61a7129a7992fa8a1d9312bf",
}
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after V48's complete audited "
    "negative closure and284-grid actual V29 initializer attribution. Start from unchanged "
    "actual V29, never V48 or a probe endpoint. " + cohort_retention.POLICY
    + " This separate registration authorizes only one feasibility update and its independent "
    "audits. Preserve77,998,496 trainable parameters/238 tensors,16 frozen buffers,84 raw "
    "gradients,30 grouped guards,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm cap "
    "and exact rejection rollback. Replay52 imputation teacher-component grids and26 retention "
    "grids independently: two general-native and24 frozen V29-structured. Keep native mean "
    "weighting, mean20 and all remaining loss terms, priors, teachers and task masses unchanged. "
    "Keep24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB "
    "probe budgets. No seed retry, full trial, warm start from this probe, new confirmation, "
    "consumed-bank reselection, changed evaluation or benchmark-success claim. Keep original "
    "V8/general22 development guards,32 fresh confirmation comparisons and full real review."
)


def require_completed_v48():
    diagnosis.verify()
    evidence = {str(diagnosis.OUTPUT / name): digest for name, digest in DIAGNOSTIC_HASHES.items()}
    verify_hashes(evidence)
    report = read(diagnosis.OUTPUT / "audit.json")
    state = read(diagnosis.OUTPUT / "status.json")
    if (state["state"] != "complete" or diagnosis.original.process_identity(state["pid"]) is not None
            or report["complete"] is not True or report["prediction_grids"] != 284
            or len(report["comparisons"]) != 32 or report["post_hoc_consumed_bank"] is not True
            or report["goal_success_not_assessed"] is not True
            or any(report[key] is not False for key in ("cuda_initialized", "training_performed",
                "confirmation_repeated", "checkpoint_selection_performed", "real_benchmarks_evaluated"))):
        raise ValueError("Require complete stopped V48 attribution without repurposing consumed confirmation")
    general_failures = {("general", group, "rmse") for group in ("imputation", "structured_overall")}
    expected = {
        "initializer_failed": general_failures | {("V8", "imputation", "rmse")},
        "candidate_failed": general_failures | {
            ("V8", "mnar_two_phase_subset", "common_nrmse"), ("V8", "mnar_skip_logic", "common_nrmse")},
    }
    for field, failures in expected.items():
        actual = {(row["reference"], row["group"], row["metric"])
                  for row in report["comparisons"] if row[field]}
        if actual != failures:
            raise ValueError("Preserve all inherited, introduced and repaired failures in the V48 attribution")
    verify_hashes(report["artifact_sha256"])
    evidence.update(report["artifact_sha256"])
    evidence.update({str(diagnosis.OUTPUT / name): sha256(diagnosis.OUTPUT / name)
                     for name in ("registration.json", "status.json")})
    evidence.update({str(path): sha256(path) for path in (diagnosis.CLOSURE, diagnosis.TRAINING_AUDIT,
        diagnosis.REPLAY / "gate.json", diagnosis.REPLAY / "audit.json",
        diagnosis.original.CONFIRMATION / "gate.json", diagnosis.original.CONFIRMATION / "audit.json")})
    return evidence


def reference_identities():
    return {"v29": {"checkpoint": str(probe_blueprint.PARENT),
                    "sha256": cohort_retention.REFERENCE_SHA256S["v29"]},
            "general": {"checkpoint": str(GENERAL), "sha256": cohort_retention.REFERENCE_SHA256S["general"]}}


function = probe_blueprint.replay_cohort_targets
if function.__closure__ is not None:
    raise ValueError("Bind the unchanged complete reference replay only to this new probe")
replay_cohort_targets = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
replay_cohort_targets.__kwdefaults__ = function.__kwdefaults__

PARENT_PROBE_SOURCE = probe_blueprint.NATIVE_PROBE_SOURCE
for old, new in {
    "single_model_native_column_mean as retention": "single_model_parent_structured_retention as retention",
    "native_column_mean_preflight_v1": "parent_structured_retention_preflight_v1",
    "1231100901, 1232100901": "1261100901, 1262100901",
    "require_completed_v47": "require_completed_v48",
}.items():
    if old not in PARENT_PROBE_SOURCE:
        raise ValueError(f"Missing explicit parent-structured feasibility adaptation: {old}")
    PARENT_PROBE_SOURCE = PARENT_PROBE_SOURCE.replace(old, new)
PARENT_PROBE_SOURCE = replace_assignment(PARENT_PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
PARENT_PROBE_SOURCE = replace_once(PARENT_PROBE_SOURCE,
    'Path(reference_identities()["v8"]["checkpoint"])', 'Path(reference_identities()["v29"]["checkpoint"])')
PARENT_PROBE_SOURCE = replace_once(PARENT_PROBE_SOURCE,
    '{"general": 2, "v8": 24}', '{"general": 2, "v29": 24}')
registration_source = "REGISTER_SOURCE = replace_once(REGISTER_SOURCE, " + repr(
    '    sources.update(Path(module.__file__).resolve()') + ", " + repr(
    '    sources.update({REPO / "tests/test_single_model_parent_structured_retention.py", '
    'REPO / "tests/test_single_model_parent_structured_retention_preflight.py"})\n'
    '    sources.update(Path(module.__file__).resolve()') + ")"
PARENT_PROBE_SOURCE = replace_once(PARENT_PROBE_SOURCE,
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
exec(compile(PARENT_PROBE_SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Keep all original teacher component replay coverage and capacity")
    return result


main = types.FunctionType(probe_blueprint.main.__code__, globals(), "main", probe_blueprint.main.__defaults__)


if __name__ == "__main__":
    main()
