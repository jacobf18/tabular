"""Train V30 with more legacy-imputation mass and unchanged independent qualification."""

import hashlib
import types

import train_single_matrixformer_legacy_strata as template
from train_single_matrixformer_legacy_strata import (
    admission, attempts, baseline, compact_cache, composite, diagnostics, paired, rehearsal, replicated,
    storage, teacher_policy, top_two,
)
import single_model_legacy_heavy_mixture as mixture
from single_model_registered_adaptation import replace_assignment, replace_once


RATE, PREFIXES = template.RATE, template.PREFIXES
PARENT_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
PARENT_UPDATES = 98
TRIAL_POLICY = ("Warm-start one ordinary77,998,496-parameter inference network from audited but unqualified "
    "V29 step25, with98 inherited updates; never from the legacy-heavy preflight endpoint or the V29 terminal "
    "state. Attempt64 fresh28-task batches using fresh5e-6 AdamW. Train62 final-two-block/decoder tensors, "
    "27,766,670 parameters, preserving192 other state tensors bitwise. " + mixture.POLICY + " Keep original "
    "general-imputation/specialist-rating teachers, native anchor rehearsal, all Huber/NLL/CDF/query objectives, "
    "observed-only normalization, raw gradient capture and every individual regression. " + paired.POLICY + " "
    + admission.POLICY + " " + attempts.POLICY + " Retain full compact task/teacher digests and the last actual "
    "optimizer endpoint. Keep ordinary selected/latest/published models on checkpoint storage and declared "
    "synthetic diagnostics on UUID-bound machine-local scratch, not provider-persistent storage or backup. "
    "Preserve18/6/50 GiB reserves and original per-volume budgets. No inference ensemble, scientific retry, "
    "changed original reference, real-data selection or claim of an isolated mixture-weight causal ablation.")
TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION.replace("V27-parent", "V29-parent")


for name in ("require_capacity", "require_evidence", "constant_rate", "save_endpoint", "audit_endpoint", "save_last_endpoint"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain helper closures into the previous training namespace")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

validate_preflight = template.validate_preflight
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_legacy_strata as preflight': 'import preflight_single_model_legacy_heavy as preflight',
    'OUTPUT = ROOT / "legacy_strata_v29"': 'OUTPUT = ROOT / "legacy_heavy_v30"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_strata_v29")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_heavy_v30")',
    'INITIAL_RUN = ROOT / "general_imputation_teacher_v27"': 'INITIAL_RUN = ROOT / "legacy_strata_v29"',
    'DATA_SEED, GRADIENT_SEED = 661100901, 662100901': 'DATA_SEED, GRADIENT_SEED = 691100901, 692100901',
    'CONFIRMATION_SEEDS = (671100901, 672100901)': 'CONFIRMATION_SEEDS = (701100901, 702100901)',
    'read(INITIAL_RUN / "validation_10.json")["student"]': 'read(INITIAL_RUN / "validation_25.json")["student"]',
    'or config["policy"] != POLICY':
        'or config["mixture_policy"] != mixture.POLICY or config["mixture_masses"] != mixture.MASSES\n'
        '            or config["policy"] != POLICY',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


def replicated_batches():
    return [mixture.reweight(rehearsal.stratify(replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED)))
            for batch in baseline.previous.batches()]


REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'audited unqualified V27': 'audited unqualified V29',
    'INITIAL_RUN / "validation_10.json"': 'INITIAL_RUN / "validation_25.json"',
    'REPO / "scripts/audit_single_model_legacy_strata.py"':
        'REPO / "scripts/audit_single_model_legacy_heavy.py", REPO / "scripts/run_single_model_legacy_heavy_pipeline.py"',
    'REPO / "tests/test_single_model_legacy_strata_training.py"': 'REPO / "tests/test_single_model_legacy_heavy_training.py"',
    'REPO / "tests/test_single_model_legacy_strata_audit.py"': 'REPO / "tests/test_single_model_legacy_heavy_audit.py"',
    'REPO / "tests/test_single_model_legacy_strata_pipeline.py"': 'REPO / "tests/test_single_model_legacy_heavy_pipeline.py"',
    'config = {"policy": POLICY,': 'config = {"policy": POLICY, "mixture_policy": mixture.POLICY, "mixture_masses": mixture.MASSES,',
    'never promote unchanged unqualified V27': 'never promote unchanged unqualified V29',
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
                raise ValueError("Never restart a begun legacy-heavy continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
