"""Train V31 with point-inclusive proposals and unchanged independent qualification."""

import hashlib
import types

import train_single_matrixformer_legacy_heavy as template
from train_single_matrixformer_legacy_heavy import (
    admission, attempts, baseline, compact_cache, diagnostics, rehearsal, replicated, storage, teacher_policy, top_two,
)
from preflight_single_model_legacy_point import composite, mixture, paired
from single_model_registered_adaptation import replace_assignment, replace_once


RATE, PREFIXES = template.RATE, template.PREFIXES
PARENT_SHA256, PARENT_UPDATES = template.PARENT_SHA256, template.PARENT_UPDATES
TRIAL_POLICY = ("Warm-start only the same audited unqualified V29 step25 as V30, with98 inherited updates. "
    "Never initialize from V30's unselected terminal state or a diagnostic endpoint. Attempt64 fresh28-task "
    "mixed-prior batches at5e-6 fresh AdamW in the ordinary77,998,496-parameter network. Train62 final-two-block/"
    "decoder tensors and preserve192 other state tensors bitwise. " + mixture.POLICY + " " + paired.POLICY + " "
    + composite.POLICY + " " + admission.POLICY + " " + attempts.POLICY + " Retain original general-imputation/"
    "specialist-rating teachers, native-anchor rehearsal, Huber/NLL/CDF/query objectives, observed-only "
    "normalization, exact compact task/teacher digests and every individual regression. Preserve18/6/50 GiB "
    "reserves and actual-volume budgets. Ordinary selected/latest/published models remain on checkpoint "
    "storage; only declared synthetic diagnostics/raw endpoint use UUID-bound machine-local scratch, not "
    "provider-persistent storage or backup. No inference ensemble, scientific retry, changed original "
    "reference, real-data selection or matched-data causal claim.")
TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION


for name in ("require_capacity", "require_evidence", "constant_rate", "save_endpoint", "audit_endpoint", "save_last_endpoint",
             "replicated_batches"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding training helper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def validate_preflight(result, audit, gate, current, digest):
    template.validate_preflight(result, audit, gate, current, digest)
    record = result["record"]
    weights = [task["weight"] for task in preflight.batch()["tasks"]]
    if (audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or record["proposal_policy"] != paired.POLICY
            or record["proposal_weights"] != paired.proposal_weights(record["point_before"], weights)):
        raise ValueError("Require the independently audited full point-inclusive proposal, not a composite prefix")


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_legacy_heavy as preflight': 'import preflight_single_model_legacy_point as preflight',
    'OUTPUT = ROOT / "legacy_heavy_v30"': 'OUTPUT = ROOT / "legacy_point_v31"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_heavy_v30")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_point_v31")',
    'DATA_SEED, GRADIENT_SEED = 691100901, 692100901': 'DATA_SEED, GRADIENT_SEED = 721100901, 722100901',
    'CONFIRMATION_SEEDS = (701100901, 702100901)': 'CONFIRMATION_SEEDS = (731100901, 732100901)',
    'or config["policy"] != POLICY':
        'or config["proposal_policy"] != paired.POLICY or config["gradient_storage_policy"] != composite.POLICY\n'
        '            or config["policy"] != POLICY',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'REPO / "scripts/audit_single_model_legacy_heavy.py"': 'REPO / "scripts/audit_single_model_legacy_point.py"',
    'REPO / "scripts/run_single_model_legacy_heavy_pipeline.py"': 'REPO / "scripts/run_single_model_legacy_point_pipeline.py"',
    'REPO / "tests/test_single_model_legacy_heavy_training.py"': 'REPO / "tests/test_single_model_legacy_point_training.py"',
    'REPO / "tests/test_single_model_legacy_heavy_audit.py"': 'REPO / "tests/test_single_model_legacy_point_audit.py"',
    'REPO / "tests/test_single_model_legacy_heavy_pipeline.py"': 'REPO / "tests/test_single_model_legacy_point_pipeline.py"',
    'config = {"policy": POLICY,':
        'config = {"policy": POLICY, "proposal_policy": paired.POLICY, "gradient_storage_policy": composite.POLICY,',
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
                raise ValueError("Never restart a begun point-inclusive continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
