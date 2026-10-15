"""Test one prospectively registered legacy-heavy update from unqualified V29 step25."""

import hashlib
import inspect
from pathlib import Path
import shutil
import types

import torch

import preflight_single_model_legacy_strata as template
from preflight_single_model_legacy_strata import (
    admission, composite, paired, rehearsal, replicated, schedule, storage, teacher_policy, top_two, StructuredSampler,
)
import run_single_model_legacy_strata_pipeline as preceding
import recover_single_model_legacy_strata_stable_identity as recovery
import single_model_legacy_heavy_mixture as mixture
from single_model_registered_adaptation import replace_assignment, replace_once
from relocate_single_model_gradient_archive import sync_directory


PREFIXES = template.PREFIXES
PARENT_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
WRITE_BUDGETS = {"checkpoint": 256 * 1024**2, "root": 16 * 1024**2, "scratch": 17 * 1024**3}
RESERVES = {"checkpoint": 18 * 1024**3, "root": 6 * 1024**3, "scratch": 50 * 1024**3}
POLICY_TEXT = ("One fresh synthetic optimizer feasibility test after V29's complete independently audited "
    "but unsuccessful confirmation. Warm-start only the ordinary77,998,496-parameter student from V29 step25, "
    "never a diagnostic endpoint; retain original V8/general/blend/specialist qualification references. "
    + mixture.POLICY + " Keep original general-imputation and specialist-rating teachers, native anchor "
    "rehearsal, training-only data admission without redraw, fresh5e-6 AdamW,62 final-two-block/decoder tensors "
    "and192 bitwise-frozen state tensors. Keep all84 raw gradients, separate legacy protection, paired other "
    "cohorts,0.1 inward margin,2x displacement bound and exact rollback. Independently regenerate24 structured "
    "admission draws, replay all28 teacher targets, CPU-audit all56 before/candidate grids and actual optimizer "
    "endpoint; report individual regressions. Rejected admission performs no model setup and fabricates no "
    "metrics. No validation, confirmation, nomination, real evaluation, scientific retry, changed held-out gate "
    "or inference ensemble. This tests feasibility, not accuracy or causal attribution relative to earlier "
    "different-data probes. Preserve18/6/50 GiB reserves, with256 MiB checkpoint,16 MiB root and17 GiB scratch "
    "write budgets. Only this new nonselectable diagnostic checkpoint and raw optimizer evidence use canonical "
    "links to UUID-bound machine-local scratch, not provider-persistent storage or external backup. Never move "
    "selected models, original teachers, old evidence or active outputs.")


def require_space():
    preceding.diagnostics.storage.require_mount()
    volumes = {"checkpoint": ROOT, "root": Path("/"), "scratch": SCRATCH.parent}
    for name, path in volumes.items():
        if shutil.disk_usage(path).free - WRITE_BUDGETS[name] < RESERVES[name]:
            raise OSError(f"Preserve {RESERVES[name]} bytes plus {WRITE_BUDGETS[name]} budget on {name}")


def validate_predecessor(result, training_audit, replay_audit, confirmation_audit, parent_digest):
    if (result["complete"] is not True or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True or result["gate"]["selected_step"] != 25
            or result["campaign"] is not None or result["review"] is not None
            or result["scientific_stages_repeated"] is not False or result["no_inference_ensemble"] is not True
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["accepted_updates"] != 63
            or training_audit["best"]["step"] != 25 or training_audit["prediction_grids"] != 1368
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 152 or confirmation_audit["complete"] is not True
            or confirmation_audit["prediction_grids"] != 1136 or confirmation_audit["cases"] != 284
            or parent_digest != PARENT_SHA256 or result["gate"]["checkpoint_sha256"] != parent_digest):
        raise ValueError("Require complete rejected V29 and its exact audited unqualified step25, never a new baseline")


def require_predecessor():
    recovery.namespace()["verify"](recovery.OUTPUT)
    training_audit = preceding.prerequisite(preceding.OUTPUT)
    result = read(recovery.OUTPUT / "result.json")
    replay_audit, confirmation_audit = [read(directory / "audit.json") for directory in (recovery.REPLAY, preceding.CONFIRMATION)]
    validate_predecessor(result, training_audit, replay_audit, confirmation_audit, sha256(PARENT))
    directories = (preceding.training.OUTPUT, recovery.OUTPUT, recovery.REPLAY, preceding.CONFIRMATION,
                   preceding.AUDIT.with_name(preceding.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or preceding.process_identity(current["pid"]) is not None:
            raise ValueError("All original V29 science and recovery processes must be complete and stopped")
    for report in (replay_audit, confirmation_audit):
        verify_hashes(report["artifact_sha256"])
    if (result["gate"] != read(preceding.CONFIRMATION / "gate.json")
            or (preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or preceding.CAMPAIGN.exists() or preceding.REVIEW.exists()):
        raise ValueError("Preserve V29's failed qualification and unpublished state")
    paths = [directory / "status.json" for directory in directories]
    paths.extend(recovery.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend(directory / name for directory in (recovery.REPLAY, preceding.CONFIRMATION) for name in ("audit.json", "gate.json"))
    paths.extend([preceding.AUDIT / "audit.json", preceding.training.OUTPUT / "config.json",
                  preceding.training.OUTPUT / "registration.json", preceding.training.OUTPUT / "training.jsonl", PARENT])
    return {str(path): sha256(path) for path in paths}


def save_probe_checkpoint(payload, path):
    target = SCRATCH / "preflight_checkpoint.pth"
    if (path != OUTPUT / "preflight_checkpoint.pth" or path.exists() or path.is_symlink()
            or target.exists() or target.is_symlink() or payload["preflight_only"] is not True
            or payload["selection_qualified"] is not False):
        raise ValueError("Write only a new nonselectable diagnostic checkpoint to declared scratch")
    digest = training.save_tensor(payload, target)
    path.symlink_to(target)
    sync_directory(path.parent)
    write_json(path.with_suffix(".scratch-write.json"), {"path": str(path), "target": str(target), "sha256": digest,
               "preflight_only": True, "selection_qualified": False, "scratch_is_provider_persistent": False})
    return digest


def audit_probe_checkpoint(path):
    target, receipt = SCRATCH / "preflight_checkpoint.pth", path.with_suffix(".scratch-write.json")
    if (path != OUTPUT / "preflight_checkpoint.pth" or not path.is_symlink() or path.readlink() != target
            or path.resolve() != target or target.is_symlink() or read(receipt) != {
                "path": str(path), "target": str(target), "sha256": sha256(path), "preflight_only": True,
                "selection_qualified": False, "scratch_is_provider_persistent": False}):
        raise ValueError("Preserve the exact new diagnostic checkpoint, canonical link and scratch receipt")
    return {str(receipt): sha256(receipt)}


for name in ("frozen_inventory", "save_raw", "audit_raw", "expected_role", "check_routing", "replay_targets",
             "audit_teacher_replay", "admit_batch", "audit_admission"):
    function = template.replay_admitted if name == "replay_targets" else getattr(template, name)
    unwrapped = inspect.unwrap(function)
    if unwrapped.__closure__ is not None:
        raise ValueError("Do not preserve a helper closure into the old preflight namespace")
    cloned = types.FunctionType(unwrapped.__code__, globals(), unwrapped.__name__, unwrapped.__defaults__)
    cloned.__kwdefaults__ = unwrapped.__kwdefaults__
    if function is not unwrapped:
        if name != "replay_targets":
            raise ValueError("Only the known no-grad teacher replay wrapper may be reconstructed")
        cloned = torch.no_grad()(cloned)
    globals()[name] = cloned

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    "legacy-stratum preflight": "legacy-heavy preflight",
    "exact audited unqualified V27 selection": "exact audited unqualified V29 selection",
    'REPO / "tests/test_single_model_legacy_strata_preflight.py"':
        'REPO / "tests/test_single_model_legacy_heavy_mixture.py", REPO / "tests/test_single_model_legacy_heavy_preflight.py"',
    'config = {"policy": POLICY,': 'config = {"policy": POLICY, "mixture_policy": mixture.POLICY,\n'
        '              "mixture_masses": mixture.MASSES, "minimum_free_bytes": RESERVES, "write_budget_bytes": WRITE_BUDGETS,',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "legacy_strata_preflight_v1"': 'OUTPUT = ROOT / "legacy_heavy_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_strata_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_heavy_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 651100901, 652100901': 'DATA_SEED, GRADIENT_SEED = 681100901, 682100901',
    'return rehearsal.stratify(replicated.replicate_batch(schedule.batches()[0], DATA_SEED, GRADIENT_SEED))':
        'return mixture.reweight(rehearsal.stratify(replicated.replicate_batch(schedule.batches()[0], DATA_SEED, GRADIENT_SEED)))',
    'or config["policy"] != POLICY':
        'or config["mixture_policy"] != mixture.POLICY or config["mixture_masses"] != mixture.MASSES\n'
        '            or config["minimum_free_bytes"] != RESERVES or config["write_budget_bytes"] != WRITE_BUDGETS\n'
        '            or config["policy"] != POLICY',
    'checkpoint_digest = training.save_tensor(': 'checkpoint_digest = save_probe_checkpoint(',
    '    hashes.update(admission_evidence["artifact_sha256"])':
        '    hashes.update(audit_probe_checkpoint(checkpoint_path))\n    hashes.update(admission_evidence["artifact_sha256"])',
    '# Independent legacy-stratum optimizer preflight': '# Legacy-heavy mixed-prior optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted, replay_admitted = audit, replay_targets
for name in ("replay_targets", "audit"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())


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
