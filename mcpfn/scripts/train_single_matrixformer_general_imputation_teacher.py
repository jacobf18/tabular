"""Train V27 from unqualified V26 with the original general teacher for all imputation."""

import hashlib
import inspect

import train_single_matrixformer_top_two_blocks as template
import single_model_general_imputation_teacher as teacher_policy
from train_single_matrixformer_top_two_blocks import baseline, compact_cache, composite, paired, replicated, storage, top_two, diagnostics
from single_model_registered_adaptation import replace_once, replace_assignment


RATE = 1e-5
PREFIXES = template.PREFIXES
PARENT_SHA256 = "5096341d7ed637848b4eb8807a81c96777849fac22e67d3213a1b9ae3cce242a"
PARENT_UPDATES = 64
TRIAL_POLICY = ("Warm-start one ordinary77,998,496-parameter inference network from audited but unqualified V26 "
    "step64, never the diagnostic preflight endpoint. Original V26 is not a teacher or baseline. Original general "
    "teaches both legacy and structured imputation; original specialist teaches ratings; original V8 remains the "
    "retention baseline. Attempt64 fresh28-task batches with fresh AdamW moments and constant1e-5, paired0.1 "
    "protection,2x norm bound and exact rollback. Train the same62 final-two-block/decoder tensors,27,766,670 "
    "parameters, preserving192 other state tensors bitwise. Keep .25 legacy/.25 structured/.5 ratings mass, "
    "MAR/skip emphasis, shapes/components/regimes, observed-only normalization and Huber/NLL/CDF/query objectives. "
    "Preserve every individual-draw regression, exact compact complete task/teacher digests,128 rating contexts, "
    "all84 terminal gradients and the identical28-row proposal alias. Only registered synthetic diagnostics and "
    "bulky endpoint evidence use UUID-bound machine-local scratch, not provider-persistent storage or backup. "
    "Ordinary selected/latest/published models and real campaigns remain on checkpoint storage. Preserve18/6/50 "
    "GiB reserves and declared actual-volume stage write budgets. No inference ensemble, scientific retry, "
    "replacement reference, omitted case, changed normalization, relaxed gate or real-data model selection.")
TRIAL_QUALIFICATION = ("Independently regenerate/digest-check all1792 complete task/teacher payloads and128 rating "
    "contexts, audit all64 attempts and1368 original reference, V26-parent and development grids,62 optimizer "
    "coordinates,192 frozen tensors and the full84-gradient endpoint. No historical full-autograd replay claim. "
    "Select only at0/10/25/50/64 under all22 original V8 guards and strict minimax improvement; step0 is ineligible "
    "and ties retain the earlier trained boundary. Independently reconstruct/replay the actual optimizer endpoint "
    "and sole selected152-case development state without reselection. Confirm once on284 new reserved cases "
    "against original V8/general under unchanged gates and independently audit1136 grids. Only then publish a "
    "best alias and run all frozen tabular, four-dataset recommender and LLM comparisons/review. Preserve every "
    "failure and do not claim goal success from synthetic improvement. Consumed V26 confirmation stays diagnostic.")


for name in ("require_capacity", "validate_preflight", "require_evidence"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_top_two_blocks as preflight':
        'import preflight_single_model_general_imputation_teacher as preflight',
    'OUTPUT = ROOT / "top_two_blocks_v26"': 'OUTPUT = ROOT / "general_imputation_teacher_v27"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_two_blocks_v26")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/general_imputation_teacher_v27")',
    'INITIAL_RUN = original.alignment.fixed.PRODUCER': 'INITIAL_RUN = ROOT / "top_two_blocks_v26"',
    'DATA_SEED, GRADIENT_SEED = 561100901, 562100901': 'DATA_SEED, GRADIENT_SEED = 591100901, 592100901',
    'CONFIRMATION_SEEDS = (571100901, 572100901)': 'CONFIRMATION_SEEDS = (601100901, 602100901)',
    'teacher_policy.identities(config["base"], PARENT)':
        'teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)\n'
        '            or config["parent_training_updates"] != PARENT_UPDATES\n'
        '            or config["retention_baseline"] != str(original.alignment.fixed.PARENT)\n'
        '            or config["input_sha256"][str(PARENT)] != PARENT_SHA256',
    'or config["parent_selection_qualified"] is not True': 'or config["parent_selection_qualified"] is not False',
    '"parent_training_updates": 0': '"parent_training_updates": PARENT_UPDATES',
    'read(INITIAL_RUN / "validation_0.json")["student"]': 'read(INITIAL_RUN / "validation_64.json")["student"]',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("constant_rate", "replicated_batches", "save_endpoint", "audit_endpoint"):
    exec(compile(inspect.getsource(getattr(template.template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'if PARENT != original.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:\n'
    '        raise ValueError("Initialize only from original V8")':
        'if PARENT != INITIAL_RUN / "selected_checkpoint.pth" or sha256(PARENT) != PARENT_SHA256:\n'
        '        raise ValueError("Initialize only the student from audited unqualified V26")',
    'INITIAL_RUN / "validation_0.json"': 'INITIAL_RUN / "validation_64.json"',
    'REPO / "scripts/audit_single_model_top_two_blocks.py"':
        'REPO / "scripts/audit_single_model_general_imputation_teacher.py"',
    'REPO / "tests/test_single_model_top_two_blocks_trial.py"':
        'REPO / "tests/test_single_model_general_imputation_training.py"',
    '"parent_selection_qualified": True, "retention_baseline": str(PARENT),':
        '"parent_selection_qualified": False, "parent_training_updates": PARENT_UPDATES,\n'
        '              "retention_baseline": str(original.alignment.fixed.PARENT),',
    'teacher_policy.identities(base, PARENT)': 'teacher_policy.identities(base, original.alignment.fixed.PARENT)',
    'never promote unchanged V8 as a newly trained model': 'never promote unchanged unqualified V26 as a newly trained model',
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
                raise ValueError("Never restart a begun general-imputation-teacher trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
