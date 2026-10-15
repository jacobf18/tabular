"""One fresh native-general retention probe after the audited negative V46 result."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import diagnose_single_model_stronger_mean_initializer as diagnosis
import preflight_single_model_stronger_mean_distillation as probe_blueprint
import preflight_single_model_uniform_v8_retention as previous_probe
import single_model_all_imputation_blend_replay as blend_replay
import single_model_general_native_retention as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ARITHMETIC = {"blas_threads": 24, "torch_threads": 4}
ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
DIAGNOSTIC_SHA256 = "d75c80670e1de98d8ca48041ae4d7c23ef7f0cc650d32bf4e4eda49cfa9397ac"
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after the completed negative V46 "
    "confirmation and complete284-grid V29 initializer attribution. Start from unchanged V29, "
    "never V46 or another probe endpoint. Change only the native retention reference from V8 to "
    "the original general parent; all structured families retain V8. " + cohort_retention.POLICY
    + " Preserve77,998,496 trainable parameters/238 tensors,16 frozen buffers,84 raw gradients,30 "
    "grouped protections,fresh5e-6 AdamW,.1 projection margin,fixed-half displacement,2x norm bound "
    "and exact rollback. Keep mean20, all original teacher/component targets, priors and task masses. "
    "Independently replay52 imputation teacher components and all26 retention grids: two general "
    "native and24 V8 structured. Retain24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves "
    "and512 MiB/16 MiB/32 GiB probe budgets. One prospective feasibility update and independent "
    "audits only; no seed retry, full trial, modified evaluation, consumed-bank selection, warm "
    "start from a probe, inference ensemble or benchmark-success claim. All original22 development "
    "and32 fresh confirmation comparisons remain mandatory for any separately registered trial."
)


def require_completed_v46():
    diagnosis.verify()
    path = diagnosis.OUTPUT / "audit.json"
    verify_hashes({str(path): DIAGNOSTIC_SHA256})
    report = read(path)
    state = read(diagnosis.OUTPUT / "status.json")
    if (state["state"] != "complete" or diagnosis.original.process_identity(state["pid"]) is not None
            or report["complete"] is not True or report["prediction_grids"] != 284
            or len(report["comparisons"]) != 32 or report["post_hoc_consumed_bank"] is not True
            or any(report[key] is not False for key in ("cuda_initialized", "training_performed",
                "confirmation_repeated", "checkpoint_selection_performed", "real_benchmarks_evaluated"))):
        raise ValueError("Require the complete stopped V46 initializer diagnosis without repurposing its bank")
    expected = {("V8", "mar", "common_nrmse"),
                *(("general", "imputation", metric) for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse"))}
    for field in ("initializer_failed", "candidate_failed"):
        failures = {(row["reference"], row["group"], row["metric"]) for row in report["comparisons"] if row[field]}
        if failures != expected:
            raise ValueError("Preserve every inherited failure, not a favorable subset of the attribution")
    verify_hashes(report["artifact_sha256"])
    evidence = {str(path): DIAGNOSTIC_SHA256}
    evidence.update({str(diagnosis.OUTPUT / name): sha256(diagnosis.OUTPUT / name)
                     for name in ("config.json", "registration.json", "initializer.json", "status.json")})
    evidence.update({str(path): sha256(path) for path in (diagnosis.CLOSURE, diagnosis.TRAINING_AUDIT,
                     diagnosis.original.CONFIRMATION / "gate.json", diagnosis.original.CONFIRMATION / "audit.json")})
    return evidence


def reference_identities():
    return {"v8": {"checkpoint": str(previous_probe.training.alignment.fixed.PARENT),
                   "sha256": cohort_retention.REFERENCE_SHA256S["v8"]},
            "general": {"checkpoint": str(GENERAL), "sha256": cohort_retention.REFERENCE_SHA256S["general"]}}


function = probe_blueprint.replay_cohort_targets
if function.__closure__ is not None:
    raise ValueError("Bind the unchanged full reference replay only to the new probe")
replay_cohort_targets = types.FunctionType(function.__code__, globals(), "replay_cohort_targets", function.__defaults__)

PROBE_SOURCE = probe_blueprint.SOURCE
for old, new in {
    "single_model_uniform_v8_retention as retention": "single_model_general_native_retention as retention",
    "stronger_mean_distillation_preflight_v1": "general_native_retention_preflight_v1",
    "1171100901, 1172100901": "1201100901, 1202100901",
    "require_completed_v45": "require_completed_v46",
    "test_single_model_uniform_v8_retention.py": "test_single_model_general_native_retention.py",
    "test_single_model_uniform_v8_retention_preflight.py": "test_single_model_general_native_preflight.py",
}.items():
    if old not in PROBE_SOURCE:
        raise ValueError(f"Missing explicit native-general feasibility adaptation: {old}")
    PROBE_SOURCE = PROBE_SOURCE.replace(old, new)
PROBE_SOURCE = replace_assignment(PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
PROBE_SOURCE = replace_once(PROBE_SOURCE,
    'Path(reference_identities()["v8"]["checkpoint"]), PARENT,',
    'Path(reference_identities()["v8"]["checkpoint"]), Path(reference_identities()["general"]["checkpoint"]), PARENT,')
PROBE_SOURCE = replace_once(PROBE_SOURCE,
    'set(record["reference_role"] for task, record, episode, payload in records) != {"v8"}',
    'Counter(record["reference_role"] for task, record, episode, payload in records) != {"general": 2, "v8": 24}')
exec(compile(PROBE_SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Keep complete teacher component replay and its original capacity allowance")
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
