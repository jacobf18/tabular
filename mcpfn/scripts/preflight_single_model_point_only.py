"""Probe point-only training protection after V31's fully audited failed confirmation."""

import hashlib
import inspect
import types

import torch

import preflight_single_model_legacy_point as template
from preflight_single_model_legacy_point import (
    admission, preceding, rehearsal, replicated, schedule, storage, teacher_policy, top_two, StructuredSampler,
    sync_directory, composite, mixture,
)
import run_single_model_legacy_point_pipeline as latest
import single_model_point_only_protection as paired
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES, PARENT_SHA256 = template.PREFIXES, template.PARENT_SHA256
WRITE_BUDGETS, RESERVES = dict(template.WRITE_BUDGETS), dict(template.RESERVES)
SELECTED_V31_SHA256 = "427165f876fc87e5512bf62d2da4fa0324c1170f917cea86a76cb79f10649898"
POLICY_TEXT = ("One fresh synthetic optimizer feasibility test after fully audited unsuccessful V31 confirmation. "
    "Keep the same unqualified V29 step25 initializer as V30/V31, never V31's selection, its terminal state or "
    "any diagnostic endpoint. " + mixture.POLICY + " " + paired.POLICY + " " + composite.POLICY + " Retain the "
    "ordinary77,998,496-parameter network, original general-imputation/specialist-rating teachers, native-anchor "
    "rehearsal, original Huber/NLL/CDF/query and point losses, observed-only normalization, fresh5e-6 AdamW,62 "
    "final-two-block/decoder tensors and192 bitwise-frozen state tensors. Keep all84 raw gradients when all "
    "point metrics exist; constrain30 separate-legacy/paired-other point directions rather than45 joint "
    "directions. Preserve every composite directional diagnostic even when adverse. Independently regenerate24 "
    "structured admission draws without redraw, replay28 teacher targets and CPU-audit56 before/candidate grids "
    "and actual optimizer state. Rejected admission performs no model setup or fabricated metric. No validation, "
    "confirmation, nomination, real evaluation, seed reuse, scientific retry, altered held-out gate or inference "
    "ensemble. This tests feasibility, not generalization or matched-data causality. Diagnostic endpoints cannot "
    "initialize future training. Preserve18/6/50 GiB reserves plus256 MiB checkpoint,16 MiB root and17 GiB scratch "
    "write budgets. Only new diagnostic checkpoint/raw evidence uses UUID-bound local scratch links, not "
    "provider-persistent storage or external backup.")


def validate_latest(result, training_audit, replay_audit, confirmation_audit, gate, digest):
    if (result["complete"] is not True or result["gate"] != gate or gate["passed"] is not False
            or gate["confirmation_evaluated"] is not True or gate["selected_step"] != 50
            or gate["cases"] != 284 or gate["checkpoint_sha256"] != digest or digest != SELECTED_V31_SHA256
            or result["campaign"] is not None or result["review"] is not None
            or result["no_inference_ensemble"] is not True
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["accepted_updates"] != 63
            or training_audit["best"]["step"] != 50 or training_audit["prediction_grids"] != 1368
            or training_audit["independently_regenerated_training_tasks"] != 1792
            or training_audit["independently_regenerated_rating_contexts"] != 128
            or training_audit["independently_regenerated_structured_draws"] != 1536
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 152 or confirmation_audit["complete"] is not True
            or confirmation_audit["cases"] != 284 or confirmation_audit["prediction_grids"] != 1136):
        raise ValueError("Require complete original V31 failure, including selected replay and fresh confirmation audits")


def require_predecessor():
    evidence = template.require_predecessor()
    directories = (latest.training.OUTPUT, latest.OUTPUT, latest.REPLAY, latest.CONFIRMATION,
                   latest.AUDIT.with_name(latest.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Finish and stop every original V31 scientific stage before another probe")
    report = latest.prerequisite(latest.OUTPUT)
    latest.require_compact_audit(report)
    replay, confirmation = [read(directory / "audit.json") for directory in (latest.REPLAY, latest.CONFIRMATION)]
    result, gate = read(latest.OUTPUT / "result.json"), read(latest.CONFIRMATION / "gate.json")
    selected = latest.training.OUTPUT / "selected_checkpoint.pth"
    validate_latest(result, report, replay, confirmation, gate, sha256(selected))
    for audit in (replay, confirmation):
        verify_hashes(audit["artifact_sha256"])
    if ((latest.training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or latest.CAMPAIGN.exists() or latest.REVIEW.exists()):
        raise ValueError("Preserve unqualified V31 without publishing or starting a real campaign")
    paths = [selected, *[directory / "status.json" for directory in directories]]
    paths.extend(latest.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend([latest.training.OUTPUT / "config.json", latest.AUDIT / "audit.json"])
    paths.extend(directory / name for directory in (latest.REPLAY, latest.CONFIRMATION) for name in ("audit.json", "gate.json"))
    return {**evidence, **{str(path): sha256(path) for path in paths}}


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw", "expected_role", "check_routing",
             "audit_teacher_replay", "admit_batch", "audit_admission", "save_probe_checkpoint", "audit_probe_checkpoint"):
    function = inspect.unwrap(getattr(template, name))
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding preflight closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    "legacy-point preflight directories": "point-only-protection preflight directories",
    'REPO / "tests/test_single_model_legacy_point_preflight.py"': 'REPO / "tests/test_single_model_point_only_preflight.py"',
    'REPO / "tests/test_single_model_legacy_point_proposal.py"': 'REPO / "tests/test_single_model_point_only_protection.py"',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "legacy_point_preflight_v1"': 'OUTPUT = ROOT / "point_only_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_point_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/point_only_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 711100901, 712100901': 'DATA_SEED, GRADIENT_SEED = 741100901, 742100901',
    '# Legacy-stratified point-inclusive optimizer preflight': '# Point-only protected optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted = audit
function = inspect.unwrap(template.replay_admitted)
if function.__closure__ is not None:
    raise ValueError("Never retain a preceding teacher replay closure")
cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
cloned.__kwdefaults__ = function.__kwdefaults__
replay_admitted = torch.no_grad()(cloned)
for name in ("replay_targets", "audit"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding admission wrapper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


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
