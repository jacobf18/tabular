"""Probe direct point-loss proposal mass after the completed, unqualified V30 run."""

import hashlib
import inspect
import types

import torch

import preflight_single_model_legacy_heavy as template
from preflight_single_model_legacy_heavy import (
    admission, preceding, rehearsal, replicated, schedule, storage, teacher_policy, top_two, StructuredSampler,
    sync_directory,
)
import run_single_model_legacy_heavy_pipeline as latest
import single_model_legacy_point_proposal as paired
import single_model_point_gradient_storage as composite
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES, PARENT_SHA256 = template.PREFIXES, template.PARENT_SHA256
WRITE_BUDGETS, RESERVES = dict(template.WRITE_BUDGETS), dict(template.RESERVES)
mixture = types.SimpleNamespace(MASSES=dict(template.mixture.MASSES), reweight=template.mixture.reweight,
    POLICY=("Retain V30's native-anchor-stratified28 tasks and .5 legacy/.25 structured/.25 ratings mass, "
            "all original within-family relative weights, teachers, normalization and raw gradients. "
            "Divide each task's proposal mass between composite and defined point gradients as explicitly "
            "registered; do not reuse V30's composite-only28-row proposal or any consumed data seeds."))
POLICY_TEXT = ("One fresh synthetic optimizer feasibility test after V30's complete audited nonselection. "
    "Warm-start the same unqualified V29 step25 as V30, never a diagnostic or nonselected terminal checkpoint. "
    + mixture.POLICY + " " + paired.POLICY + " " + composite.POLICY + " Keep the ordinary77,998,496-parameter "
    "inference network, original general-imputation/specialist-rating teachers, training-only admission "
    "without redraw, fresh5e-6 AdamW,62 final-two-block/decoder tensors and192 bitwise-frozen state tensors. "
    "Preserve every defined raw composite/point gradient,45 protected directions when all point metrics are "
    "defined, original Huber/NLL/CDF/query losses, and every individual regression. Regenerate24 admission "
    "draws, replay28 teacher targets, independently CPU-audit56 before/candidate grids and actual optimizer "
    "endpoint. Data rejection performs no model setup or fabricated metric. No validation, confirmation, "
    "nomination, real evaluation, seed reuse, scientific retry, changed qualification gate or inference "
    "ensemble. This tests feasibility, not generalization or a matched-data causal comparison. The diagnostic "
    "checkpoint cannot initialize later training. Preserve18/6/50 GiB reserves plus256 MiB checkpoint,16 MiB "
    "root and17 GiB scratch write budgets. Only this new diagnostic checkpoint/raw evidence use canonical "
    "UUID-bound machine-local scratch links, not provider-persistent storage or external backup.")


def validate_latest(result, training_audit, replay_audit, gate):
    if (result["complete"] is not True or result["gate"] != gate or gate["passed"] is not False
            or gate["confirmation_evaluated"] is not False or gate["selected_step"] != 0
            or result["campaign"] is not None or result["review"] is not None
            or result["no_inference_ensemble"] is not True
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["accepted_updates"] != 64
            or training_audit["best"]["step"] != 0 or training_audit["prediction_grids"] != 1368
            or training_audit["independently_regenerated_training_tasks"] != 1792
            or training_audit["independently_regenerated_rating_contexts"] != 128
            or training_audit["independently_regenerated_structured_draws"] != 1536
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 0):
        raise ValueError("Require complete independently audited V30 nonselection without consuming confirmation")


def require_predecessor():
    evidence = template.require_predecessor()
    directories = (latest.training.OUTPUT, latest.OUTPUT, latest.REPLAY, latest.CONFIRMATION,
                   latest.AUDIT.with_name(latest.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Wait for the exact original V30 training/audit/qualification chain to finish")
    report = latest.prerequisite(latest.OUTPUT)
    latest.require_compact_audit(report)
    replay = read(latest.REPLAY / "audit.json")
    result, gate = read(latest.OUTPUT / "result.json"), read(latest.CONFIRMATION / "gate.json")
    validate_latest(result, report, replay, gate)
    verify_hashes(replay["artifact_sha256"])
    if ((latest.training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or latest.CAMPAIGN.exists() or latest.REVIEW.exists()):
        raise ValueError("Preserve V30 nonselection without a published model or real campaign")
    paths = [directory / "status.json" for directory in directories]
    paths.extend(latest.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend([latest.training.OUTPUT / "config.json", latest.AUDIT / "audit.json",
                  latest.REPLAY / "audit.json", latest.REPLAY / "gate.json", latest.CONFIRMATION / "gate.json"])
    return {**evidence, **{str(path): sha256(path) for path in paths}}


for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw", "expected_role", "check_routing",
             "audit_teacher_replay", "admit_batch", "audit_admission", "save_probe_checkpoint", "audit_probe_checkpoint"):
    function = inspect.unwrap(getattr(template, name))
    if function.__closure__ is not None:
        raise ValueError("Never retain helper closures into an earlier preflight")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    "legacy-heavy preflight directories": "legacy-point preflight directories",
    'REPO / "tests/test_single_model_legacy_heavy_preflight.py"':
        'REPO / "tests/test_single_model_legacy_point_preflight.py", '
        'REPO / "tests/test_single_model_legacy_point_proposal.py", REPO / "tests/test_single_model_point_gradient_storage.py"',
    'config = {"policy": POLICY,':
        'config = {"policy": POLICY, "proposal_policy": paired.POLICY, "gradient_storage_policy": composite.POLICY,',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "legacy_heavy_preflight_v1"': 'OUTPUT = ROOT / "legacy_point_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_heavy_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_point_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 681100901, 682100901': 'DATA_SEED, GRADIENT_SEED = 711100901, 712100901',
    'or config["policy"] != POLICY':
        'or config["proposal_policy"] != paired.POLICY or config["gradient_storage_policy"] != composite.POLICY\n'
        '            or config["policy"] != POLICY',
    '# Legacy-heavy mixed-prior optimizer preflight': '# Legacy-stratified point-inclusive optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted = audit
function = inspect.unwrap(template.replay_admitted)
if function.__closure__ is not None:
    raise ValueError("Never retain the preceding teacher replay closure")
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
