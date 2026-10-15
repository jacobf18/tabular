"""Test one fresh anchor-stratified update from the unqualified V27 selection."""

import hashlib
import inspect

import torch

import compare_single_model_consumed_banks as diagnostic
import preflight_single_model_general_imputation_teacher as template
import run_single_model_general_imputation_teacher_pipeline as preceding
import single_model_anchor_rehearsal as rehearsal
from preflight_single_model_general_imputation_teacher import (
    composite, paired, replicated, schedule, storage, teacher_policy, top_two,
)
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES = template.PREFIXES
PARENT_SHA256 = "ef775cef8dee57338132f1756ad39042995950976d5df15c18445b205b99002d"
POLICY_TEXT = ("One fresh28-task feasibility update from audited but unqualified V27 selected step10, with9 "
    "accepted continuation updates beyond V26, not V27's terminal63 updates or any diagnostic endpoint. "
    + rehearsal.POLICY + " Original general teaches all imputation and original specialist teaches ratings; "
    "original V8 remains the retention baseline. Same Huber/NLL/CDF/query objectives, constant1e-5 fresh AdamW, "
    "paired0.1 protection,2x norm bound and exact rollback. Train62 final-two-block/decoder tensors with "
    "27,766,670 parameters; preserve192 other state tensors bitwise and the ordinary77,998,496-parameter "
    "single-network inference architecture. Preserve all84 gradients and the exact28-row proposal alias. "
    "Independently replay all28 complete teacher grids and CPU-audit56 before/candidate grids and the "
    "optimizer endpoint. No historical full-autograd replay claim. No validation, confirmation, nomination, "
    "publication or real benchmarks, and this endpoint cannot initialize a later model. Both consumed "
    "confirmation banks remain diagnostic, never training or fresh confirmation. Keep canonical evidence "
    "and the diagnostic checkpoint on checkpoint storage, bulky raw endpoint evidence on UUID-bound local "
    "scratch, not provider-persistent storage or external backup. Preserve18/6/50 GiB reserves and1/0/16 GiB "
    "checkpoint/root/scratch stage write budgets. No retry or change to any registered source or gate.")


def validate_predecessor(result, audit, replay, confirmation, digest):
    if (result["complete"] is not True or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True
            or result["campaign"] is not None or result["review"] is not None
            or audit["complete"] is not True or audit["complete_training_audit"] is not True
            or audit["through_step"] != 64 or audit["accepted_updates"] != 63 or audit["best"]["step"] != 10
            or audit["training_episodes"] != 1792 or audit["prediction_grids"] != 1368
            or audit.get("top_two_block_training") is not True
            or audit["independently_regenerated_training_tasks"] != 1792
            or audit["independently_regenerated_rating_contexts"] != 128
            or digest != PARENT_SHA256 or audit["artifact_sha256"].get(str(PARENT)) != digest
            or replay["complete"] is not True or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or confirmation["complete"] is not True or confirmation["cases"] != 284
            or confirmation["prediction_grids"] != 1136):
        raise ValueError("Require the complete audited but unqualified V27 selected checkpoint")


def require_predecessor():
    config = diagnostic.verify()
    diagnostic.require_stopped(diagnostic.OUTPUT, {"complete"})
    comparison = read(diagnostic.OUTPUT / "audit.json")
    if (comparison["complete"] is not True or comparison["config_sha256"] != sha256(diagnostic.OUTPUT / "config.json")
            or comparison["prediction_grids"] != 1136 or comparison["new_grids"] != 568
            or comparison["paired_cases"] != 568 or comparison["summary_comparisons"] != 144
            or any(comparison[field] is not False for field in ("training_performed", "selection_performed",
                "qualification_changed", "bank_regenerated", "teacher_change_alone_causally_isolated"))):
        raise ValueError("Require the complete diagnosis-only paired comparison, without reselection")
    verify_hashes(comparison["artifact_sha256"])
    directories = (preceding.training.OUTPUT, preceding.OUTPUT, preceding.REPLAY, preceding.CONFIRMATION,
                   preceding.AUDIT.with_name(preceding.AUDIT.name + "_task_replay"))
    for directory in directories:
        diagnostic.require_stopped(directory, {"complete", "training_complete"})
    audit = preceding.prerequisite(preceding.OUTPUT)
    replay, confirmation = [read(directory / "audit.json") for directory in (preceding.REPLAY, preceding.CONFIRMATION)]
    validate_predecessor(read(preceding.OUTPUT / "result.json"), audit, replay, confirmation, sha256(PARENT))
    records = [json.loads(line) for line in (preceding.training.OUTPUT / "training.jsonl").read_text().splitlines()]
    if (len(records) != 64 or [record["step"] for record in records] != list(range(1, 65))
            or sum(record["accepted"] for record in records[:10]) != 9):
        raise ValueError("Warm-start from nine accepted V27 continuation updates, never the terminal state")
    if ((preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or preceding.CAMPAIGN.exists() or preceding.REVIEW.exists()):
        raise ValueError("Preserve rejected V27 without a published model or real campaign")
    paths = [diagnostic.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "audit.json")]
    paths.extend(preceding.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "status.json"))
    paths.extend(directory / "status.json" for directory in directories)
    paths.extend(directory / name for directory in (preceding.REPLAY, preceding.CONFIRMATION) for name in ("audit.json", "gate.json"))
    paths.extend([preceding.AUDIT / "audit.json", preceding.training.OUTPUT / "config.json",
                  preceding.training.OUTPUT / "registration.json", preceding.training.OUTPUT / "training.jsonl", PARENT])
    return {**config["input_sha256"], **comparison["artifact_sha256"], **{str(path): sha256(path) for path in paths}}


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw"):
    source = inspect.getsource(getattr(template.storage_template, name))
    if name == "require_space":
        source = replace_once(source, "(SCRATCH.parent, 50, 12)", "(SCRATCH.parent, 50, 16)")
    elif name == "frozen_inventory":
        source = replace_once(source, "len(expected) != 34", "len(expected) != 62")
        source = replace_once(source, "and34 last-block/decoder tensors", "and62 final-two-block/decoder tensors")
    exec(compile(source, __file__, "exec"), globals())

exec(compile(inspect.getsource(template.expected_role), __file__, "exec"), globals())
for name in ("check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(template.teacher_template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = inspect.getsource(template.register_trial)
for old, new in {
    "general-imputation-teacher preflight": "anchor-rehearsal preflight",
    "exact audited unqualified V26 selection": "exact audited unqualified V27 selection",
    "test_single_model_general_imputation_teacher.py": "test_single_model_anchor_rehearsal.py",
    "test_single_model_general_imputation_preflight.py": "test_single_model_anchor_preflight.py",
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "general_imputation_teacher_preflight_v1"': 'OUTPUT = ROOT / "anchor_rehearsal_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/general_imputation_teacher_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_rehearsal_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 581100901, 582100901': 'DATA_SEED, GRADIENT_SEED = 611100901, 612100901',
    'return replicated.replicate_batch(schedule.batches()[0], DATA_SEED, GRADIENT_SEED)':
        'return rehearsal.stratify(replicated.replicate_batch(schedule.batches()[0], DATA_SEED, GRADIENT_SEED))',
    '# General-imputation teacher optimizer preflight': '# Anchor-stratified rehearsal optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
