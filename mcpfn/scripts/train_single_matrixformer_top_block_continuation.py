"""Continue the audited, unqualified V24 model on fresh mixed-prior training tasks."""

import hashlib
import inspect

import train_single_matrixformer_top_block as template
import train_single_matrixformer_targeted_decoder as schedule
import run_single_model_top_block_pipeline as parent_pipeline
import single_model_scratch_diagnostics as diagnostics
from train_single_matrixformer_top_block import compact_cache, composite, paired, replicated, storage, teacher_policy, top_block
from single_model_registered_adaptation import replace_once, replace_assignment


RATE = 1e-5
PREFIXES = template.PREFIXES
PARENT_SHA256 = "fe3809df137b2566a4f5c85e460c6149d6573fe8e3391154bdbbf81a7c216769"
PARENT_UPDATES = 61
TRIAL_POLICY = ("Initialize one ordinary network from the independently audited V24 selected step64, which remains "
    "unqualified after failed fresh confirmation. Attempt96 fresh28-task batches with fresh AdamW moments and the "
    "unchanged constant1e-5 composite proposal, paired0.1 protection,2x norm bound and exact finite rollback. Train "
    "only the same34 final-block/decoder tensors,19,496,843 parameters; keep220 other state tensors bitwise frozen. "
    "Preserve .25 legacy/.25 structured/.5 ratings mass, MAR/skip emphasis, shapes/components/regimes, observed-only "
    "normalization and Huber/NLL/CDF/query objectives. Original V8 remains the legacy teacher and all retention/"
    "confirmation baselines; original general teaches structured imputation and original specialist teaches ratings. "
    "V24 is never a teacher or qualified reference. Preserve compact complete task/teacher digests, rating vectors, "
    "all84 terminal gradients and the unchanged28-row proposal alias. Prospectively place only new development, "
    "endpoint and confirmation diagnostics on UUID-bound machine-local scratch; ordinary latest/selected/published "
    "checkpoints remain on checkpoint storage. Scratch is not provider-persistent storage or external backup. "
    "No inference ensemble, new architecture, changed scientific gates, real-data tuning, retry, reuse of consumed "
    "V24 confirmation or promotion of an unchanged parent. This tests additional mixed-prior training.")
TRIAL_QUALIFICATION = ("Independently regenerate/digest-check all2688 new complete task/teacher payloads,192 rating "
    "contexts and all96 attempted updates; audit1368 original reference, V24-parent and scheduled development grids, "
    "all22 original V8 guards,220 frozen tensors and the complete terminal optimizer/gradient endpoint. Do not claim "
    "a full historical autograd trajectory replay. Select only at0/16/32/64/96 under unchanged V8 retention and "
    "strict minimax improvement; step0 is ineligible and ties retain the earlier trained boundary. Independently "
    "reconstruct/replay the endpoint and sole selected152-case development state without reselection. Confirm once "
    "on284 unused reserved cases under unchanged V8/general guards and independently audit all1136 grids. Only "
    "then publish a best alias and run every frozen tabular, four-dataset recommender and LLM comparison/review. "
    "Preserve18/6/50 GiB checkpoint/root/scratch reserves plus each stage's registered per-volume write budget. "
    "No hidden regression, fallback, relaxed threshold or claim of goal success from synthetic improvement.")


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in schedule.batches()[:STEPS]]


def require_capacity():
    diagnostics.storage.require_mount()
    diagnostics.require_capacity("training")


def validate_history(result, audit, replay, confirmation, digest):
    if (not result["complete"] or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True
            or result["campaign"] is not None or result["review"] is not None
            or not audit["complete"] or not audit["complete_training_audit"]
            or audit["through_step"] != 64 or audit["accepted_updates"] != PARENT_UPDATES
            or audit["best"]["step"] != 64 or audit["training_episodes"] != 1792
            or audit["prediction_grids"] != 1216 or audit["decoder_only_training"] is not False
            or audit["top_block_training"] is not True
            or audit["independently_regenerated_training_tasks"] != 1792
            or audit["independently_regenerated_rating_contexts"] != 128
            or audit["artifact_sha256"].get(str(PARENT)) != PARENT_SHA256 or digest != PARENT_SHA256
            or not replay["complete"] or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or not confirmation["complete"] or confirmation["cases"] != 284 or confirmation["prediction_grids"] != 1136):
        raise ValueError("Require the complete independently audited but unqualified V24 lifecycle")


def require_evidence():
    directories = (template.OUTPUT, parent_pipeline.OUTPUT, parent_pipeline.REPLAY, parent_pipeline.CONFIRMATION,
                   parent_pipeline.AUDIT.with_name(parent_pipeline.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Finish and stop every original V24 producer before registering a continuation")
    parent_pipeline.verify(parent_pipeline.OUTPUT)
    audit = parent_pipeline.prerequisite(parent_pipeline.OUTPUT)
    replay = read(parent_pipeline.REPLAY / "audit.json")
    confirmation = read(parent_pipeline.CONFIRMATION / "audit.json")
    validate_history(read(parent_pipeline.OUTPUT / "result.json"), audit, replay, confirmation, sha256(PARENT))
    if ((template.OUTPUT / "best_validation_checkpoint.pth").exists() or parent_pipeline.CAMPAIGN.exists()
            or parent_pipeline.REVIEW.exists()):
        raise ValueError("The failed V24 confirmation must remain unpublished and without a real campaign")
    for report in (replay, confirmation):
        verify_hashes(report["artifact_sha256"])
    paths = [parent_pipeline.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "status.json")]
    paths.extend(directory / "status.json" for directory in directories)
    paths.extend(directory / name for directory in (parent_pipeline.REPLAY, parent_pipeline.CONFIRMATION)
                 for name in ("audit.json", "gate.json"))
    paths.extend([parent_pipeline.AUDIT / "audit.json", template.OUTPUT / "config.json",
                  template.OUTPUT / "registration.json", PARENT])
    return {str(path): sha256(path) for path in paths}


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "top_block_v24"': 'OUTPUT = ROOT / "top_block_continuation_v25"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_block_v24_raw")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/top_block_continuation_v25")',
    'INITIAL_RUN = original.alignment.fixed.PRODUCER': 'INITIAL_RUN = ROOT / "top_block_v24"',
    'PARENT = preflight.PARENT': 'PARENT = INITIAL_RUN / "selected_checkpoint.pth"',
    'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)': 'STEPS, BOUNDARIES = 96, (0, 16, 32, 64, 96)',
    'DATA_SEED, GRADIENT_SEED = 491100901, 492100901': 'DATA_SEED, GRADIENT_SEED = 531100901, 532100901',
    'CONFIRMATION_SEEDS = (501100901, 502100901)': 'CONFIRMATION_SEEDS = (541100901, 542100901)',
    'teacher_policy.identities(config["base"], PARENT)':
        'teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)\n'
        '            or config["parent_selection_qualified"] is not False\n'
        '            or config["parent_training_updates"] != PARENT_UPDATES\n'
        '            or config["diagnostic_storage"] != diagnostics.layout(OUTPUT, SCRATCH, BOUNDARIES)',
    '    return config\n': '    diagnostics.verify(config["diagnostic_storage"], "training")\n    return config\n',
    '"parent_training_updates": 0': '"parent_training_updates": PARENT_UPDATES',
    '    directory.mkdir()\n': '    diagnostics.require_empty_validation(config["diagnostic_storage"], directory)\n',
    'read(INITIAL_RUN / "validation_0.json")["student"]': 'read(INITIAL_RUN / "validation_64.json")["student"]',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("constant_rate", "save_endpoint", "audit_endpoint"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = inspect.getsource(template.register_trial)
for old, new in {
    'if PARENT != original.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:\n'
    '        raise ValueError("Initialize only from original V8")':
        'if PARENT != template.OUTPUT / "selected_checkpoint.pth" or sha256(PARENT) != PARENT_SHA256:\n'
        '        raise ValueError("Initialize only from the audited unqualified V24 selection")',
    'len(fresh) != 3868': 'len(fresh) != 5660',
    'Reserve all3584 fresh data/backward seeds': 'Reserve all5376 fresh data/backward seeds',
    'INITIAL_RUN / "validation_0.json"': 'INITIAL_RUN / "validation_64.json"',
    'REPO / "scripts/audit_single_model_top_block_training.py"':
        'REPO / "scripts/audit_single_model_top_block_continuation.py",\n'
        '               REPO / "tests/test_single_model_scratch_diagnostics.py"',
    'REPO / "tests/test_single_model_top_block_trial.py"': 'REPO / "tests/test_single_model_top_block_continuation.py"',
    '"parent_selection_qualified": True, "retention_baseline": str(PARENT),':
        '"parent_selection_qualified": False, "parent_training_updates": PARENT_UPDATES,\n'
        '              "retention_baseline": str(original.alignment.fixed.PARENT),\n'
        '              "diagnostic_storage": diagnostics.layout(OUTPUT, SCRATCH, BOUNDARIES),',
    'teacher_policy.identities(base, PARENT)': 'teacher_policy.identities(base, original.alignment.fixed.PARENT)',
    'never promote unchanged V8 as a newly trained model': 'never promote unqualified V24 as a newly trained model',
    '    output.mkdir()\n': '    output.mkdir()\n    diagnostics.prepare(config["diagnostic_storage"], "training")\n',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart a begun last-block continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
