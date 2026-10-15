"""Test a fresh half-rate anchor-rehearsal update after preserving the rejected probe."""

import hashlib
import inspect

import torch

import preflight_single_model_anchor_rehearsal as template
from preflight_single_model_anchor_rehearsal import (
    composite, diagnostic, paired, preceding, rehearsal, replicated, schedule, storage, teacher_policy, top_two,
)
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES, PARENT_SHA256 = template.PREFIXES, template.PARENT_SHA256
POLICY_TEXT = ("One new fresh28-task native-anchor rehearsal feasibility probe after the fully audited first "
    "probe rejected its1e-5 proposal and rolled back. Preserve that failure without restarting or changing it. "
    "Use5e-6 fresh AdamW, one half of the prior rate, to test a smaller finite displacement; new data/backward "
    "seeds prevent a matched-data causal claim about rate alone. Warm-start only from the same unqualified "
    "V27 selected step10, never the rejected probe. " + rehearsal.POLICY + " Keep original general/specialist "
    "teachers, V8 retention identity, all28 tasks, Huber/NLL/CDF/query objectives, paired0.1 margin,2x norm "
    "bound and strict actual finite-point acceptance with exact rollback. Train the same62 final-two-block/"
    "decoder tensors and preserve192 other state tensors bitwise in the ordinary77,998,496-parameter network. "
    "Independently replay28 complete teacher grids and CPU-audit56 prediction grids, all84 gradients and the "
    "optimizer endpoint. Preserve every individual-draw regression. No full-autograd replay claim, inference "
    "ensemble, validation, confirmation, nomination, publication or real benchmark. This endpoint cannot "
    "initialize later training. Consumed confirmation banks stay diagnosis-only. Preserve18/6/50 GiB reserves "
    "and1/0/16 GiB checkpoint/root/scratch write budgets; keep canonical metadata and the endpoint checkpoint "
    "on checkpoint storage and bulky raw diagnostics on existing UUID-bound local scratch, not an external "
    "backup or provider-persistent storage. No relaxed gate, changed registered source or automatic retry.")


def validate_failed_probe(result, audit, gate, current):
    if (current["state"] != "complete" or current["accepted"] is not False
            or result["preflight_only"] is not True or result["record"]["accepted"] is not False
            or result["record"]["accepted_updates"] != 0 or result["record"]["rate"] != 1e-5
            or result["record"]["reason"] != "finite_metrics_rejected"
            or audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["teacher_replay"]["complete"] is not True or audit["teacher_replay"]["full_grids"] != 28
            or audit["teacher_replay"]["bitwise_equal"] is not True
            or audit["raw_endpoint"]["accepted"] is not False or audit["raw_endpoint"]["accepted_updates"] != 0
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["autograd_replayed"] is not False
            or audit["frozen"]["frozen_tensors"] != 192 or audit["frozen"]["changed_trainable_tensors"] != []
            or gate["passed"] is not False or gate["preflight_only"] is not True
            or gate["reason"] != "finite_metrics_rejected"):
        raise ValueError("Preserve the original completed rejected rehearsal probe and exact rollback")


def require_predecessor():
    evidence = template.require_predecessor()
    template.verify(template.OUTPUT)
    diagnostic.require_stopped(template.OUTPUT, {"complete"})
    paths = [template.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "audit.json", "gate.json", "status.json")]
    result, audit, gate, current = [read(template.OUTPUT / name) for name in ("result.json", "audit.json", "gate.json", "status.json")]
    validate_failed_probe(result, audit, gate, current)
    if gate["audit_sha256"] != sha256(template.OUTPUT / "audit.json"):
        raise ValueError("Bind the unchanged original rejection to its independent audit")
    verify_hashes(audit["artifact_sha256"])
    return {**evidence, **audit["artifact_sha256"], **{str(path): sha256(path) for path in paths}}


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw"):
    source = inspect.getsource(getattr(template.template.storage_template, name))
    if name == "require_space":
        source = replace_once(source, "(SCRATCH.parent, 50, 12)", "(SCRATCH.parent, 50, 16)")
    elif name == "frozen_inventory":
        source = replace_once(source, "len(expected) != 34", "len(expected) != 62")
        source = replace_once(source, "and34 last-block/decoder tensors", "and62 final-two-block/decoder tensors")
    exec(compile(source, __file__, "exec"), globals())

exec(compile(inspect.getsource(template.template.expected_role), __file__, "exec"), globals())
for name in ("check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(template.template.teacher_template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = replace_once(template.REGISTER_SOURCE, "anchor-rehearsal preflight", "small-step-anchor preflight")
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, "test_single_model_anchor_preflight.py", "test_single_model_anchor_small_step.py")
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "anchor_rehearsal_preflight_v1"': 'OUTPUT = ROOT / "anchor_small_step_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_rehearsal_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_small_step_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 611100901, 612100901': 'DATA_SEED, GRADIENT_SEED = 621100901, 622100901',
    'RATE = 1e-5': 'RATE = 5e-6',
    '# Anchor-stratified rehearsal optimizer preflight': '# Small-step anchor rehearsal optimizer preflight',
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
