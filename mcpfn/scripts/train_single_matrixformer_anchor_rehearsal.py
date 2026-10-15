"""Train V28 with explicit native-anchor rehearsal and unchanged held-out qualification."""

import hashlib
import inspect

import train_single_matrixformer_general_imputation_teacher as template
import train_single_matrixformer_top_two_blocks as framework
import single_model_anchor_rehearsal as rehearsal
from train_single_matrixformer_general_imputation_teacher import (
    baseline, compact_cache, composite, diagnostics, paired, replicated, storage, teacher_policy, top_two,
)
from single_model_registered_adaptation import replace_assignment, replace_once


RATE = 5e-6
PREFIXES = template.PREFIXES
PARENT_SHA256 = "ef775cef8dee57338132f1756ad39042995950976d5df15c18445b205b99002d"
PARENT_UPDATES = 73
TRIAL_POLICY = ("Warm-start one ordinary77,998,496-parameter network from unqualified V27 selected step10, "
    "with9 accepted V27 updates beyond V26's64, never from either diagnostic preflight or the V27 terminal "
    "state. Original general teaches imputation, original specialist teaches ratings and V8 remains the "
    "retention reference. " + rehearsal.POLICY + " Attempt64 fresh28-task batches with fresh AdamW at "
    "constant5e-6, paired0.1 protection,2x norm bound and exact parameter/moment/rate rollback. Train62 "
    "final-two-block/decoder tensors,27,766,670 parameters, preserving192 other state tensors bitwise. "
    "Preserve observed-only normalization, all original Huber/NLL/CDF/query objectives, complete task/teacher "
    "digests,128 rating contexts, all individual regressions and all84 terminal gradients with the exact28-row "
    "proposal alias. Only registered synthetic diagnostics and bulky raw endpoint evidence use UUID-bound "
    "local scratch, not provider-persistent storage or external backup. Ordinary selected/latest/published "
    "checkpoints and real campaigns remain on checkpoint storage. Preserve18/6/50 GiB reserves and declared "
    "actual-volume stage budgets. No inference ensemble, scientific retry, replaced reference or real-data "
    "model selection. This changes prior coverage and step size, not an isolated rate-only causal ablation.")
TRIAL_QUALIFICATION = ("Independently regenerate and digest-check all1792 complete task/teacher payloads and128 "
    "rating contexts, audit all64 attempts and1368 original reference, V27-parent and development grids,62 "
    "optimizer coordinates,192 frozen tensors and the complete84-gradient terminal endpoint. Do not claim "
    "historical full-autograd replay. Select only at0/10/25/50/64 under all22 original V8 guards and strict "
    "minimax improvement; unchanged step0 is ineligible and ties retain the earlier trained boundary. Replay "
    "and independently audit the actual endpoint and sole selected152-case development state without "
    "reselection. Confirm once on284 newly reserved synthetic cases against original V8/general, auditing "
    "all1136 grids under unchanged gates. Only then publish a best alias and run every frozen tabular, "
    "four-dataset recommender and LLM comparison/review. Keep all failures, both earlier consumed confirmation "
    "banks diagnosis-only, and do not infer real goal success from synthetic improvement.")


for name in ("require_capacity", "validate_preflight", "require_evidence"):
    exec(compile(inspect.getsource(getattr(framework, name)), __file__, "exec"), globals())

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_general_imputation_teacher as preflight':
        'import preflight_single_model_anchor_small_step as preflight',
    'OUTPUT = ROOT / "general_imputation_teacher_v27"': 'OUTPUT = ROOT / "anchor_rehearsal_v28"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/general_imputation_teacher_v27")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_rehearsal_v28")',
    'INITIAL_RUN = ROOT / "top_two_blocks_v26"': 'INITIAL_RUN = ROOT / "general_imputation_teacher_v27"',
    'DATA_SEED, GRADIENT_SEED = 591100901, 592100901': 'DATA_SEED, GRADIENT_SEED = 631100901, 632100901',
    'CONFIRMATION_SEEDS = (601100901, 602100901)': 'CONFIRMATION_SEEDS = (641100901, 642100901)',
    'read(INITIAL_RUN / "validation_64.json")["student"]': 'read(INITIAL_RUN / "validation_10.json")["student"]',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("constant_rate", "save_endpoint", "audit_endpoint"):
    exec(compile(inspect.getsource(getattr(framework.template, name)), __file__, "exec"), globals())


def replicated_batches():
    return [rehearsal.stratify(replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED))
            for batch in baseline.previous.batches()]


REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'audited unqualified V26': 'audited unqualified V27',
    'INITIAL_RUN / "validation_64.json"': 'INITIAL_RUN / "validation_10.json"',
    'REPO / "scripts/audit_single_model_general_imputation_teacher.py"':
        'REPO / "scripts/audit_single_model_anchor_rehearsal.py"',
    'REPO / "tests/test_single_model_general_imputation_training.py"':
        'REPO / "tests/test_single_model_anchor_training.py"',
    'never promote unchanged unqualified V26': 'never promote unchanged unqualified V27',
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
                raise ValueError("Never restart a begun anchor-rehearsal trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
