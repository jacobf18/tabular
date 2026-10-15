"""Train one V8-start model using original-general teachers for structured synthetic priors."""

import hashlib

import train_single_matrixformer_paired_decoder as baseline
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import single_model_structured_teacher as teacher_policy
from single_model_registered_adaptation import replace_once, replace_assignment


previous = baseline.previous
constant_rate = baseline.constant_rate
TRIAL_POLICY = ("Initialize one ordinary single-checkpoint network directly from original V8, never V19 or a preflight. "
                "Preserve V19's64-attempt,28-task paired schedule, .25 legacy/.25 structured/.5 ratings mixture, "
                "all shapes/components/regimes/missingness and MAR/skip emphasis. Freeze all248 nondecoder state "
                "tensors. Keep2e-5 composite Adam proposals,0.1 paired protection margin,2x norm bound, paired "
                "finite-point acceptance and exact parameter/moment/preceding-rate rollback. Preserve Huber/NLL/CDF/"
                "query losses and observed-only normalization. Change only training teachers: original V8 for "
                "legacy imputation, original general for every structured-imputation family, original specialist "
                "for ratings. Verify actual frozen teacher tensors against registered original checkpoints. "
                "Use fresh data/backward streams; no inference ensemble, benchmark-based model selection or "
                "matched-seed causal claim. All development/confirmation/real qualification remains unchanged.")
TRIAL_QUALIFICATION = (baseline.QUALIFICATION + " Additionally independently replay all1792 complete training teacher "
                       "targets from original checkpoints before endpoint qualification; preserve the identical "
                       "target bytes in their original cached artifacts, with per-grid digests and independent CPU audit.")


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in previous.batches()]


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_paired_decoder as preflight': 'import preflight_single_model_structured_teacher as preflight',
    'OUTPUT = ROOT / "paired_decoder_v19"': 'OUTPUT = ROOT / "structured_teacher_v20"',
    'DATA_SEED, GRADIENT_SEED = 351100901, 352100901': 'DATA_SEED, GRADIENT_SEED = 381100901, 382100901',
    'CONFIRMATION_SEEDS = (361100901, 362100901)': 'CONFIRMATION_SEEDS = (391100901, 392100901)',
    'or config["mutable_prefix"] != "decoder."':
        'or config["mutable_prefix"] != "decoder."\n'
        '            or config["teacher_policy"] != teacher_policy.POLICY\n'
        '            or config["teacher_identities"] != teacher_policy.identities(config["base"], PARENT)',
    'preflight.verify(preflight.OUTPUT)': 'preflight.verify(preflight.OUTPUT)\n    preflight.require_predecessor()',
    'verify_hashes(audit["artifact_sha256"])':
        'verify_hashes(audit["artifact_sha256"])\n'
        '    if not audit["teacher_replay"]["complete"] or audit["teacher_replay"]["full_grids"] != 28:\n'
        '        raise ValueError("Require all28 independently replayed structured-teacher preflight targets")',
    'REPO / "tests/test_single_model_paired_decoder_training.py"': 'REPO / "tests/test_single_model_structured_teacher_training.py"',
    '"parent_checkpoint": str(PARENT), "parent_selection_qualified": True,':
        '"parent_checkpoint": str(PARENT), "parent_selection_qualified": True,\n'
        '              "teacher_policy": teacher_policy.POLICY, "teacher_identities": teacher_policy.identities(base, PARENT),',
    'with original.alignment.mixture.runtime(config["base"], output):':
        'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


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
                raise ValueError("Never restart a started structured-teacher trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
