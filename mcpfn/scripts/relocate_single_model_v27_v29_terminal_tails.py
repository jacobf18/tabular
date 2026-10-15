"""Preserve two audited nonselected terminal states before a new training continuation."""

import inspect
from pathlib import Path

import torch

import relocate_single_model_idle_root_tails as template
from relocate_single_model_idle_root_tails import copying, readers, sha256, verify_hashes, write_json
from single_model_registered_adaptation import replace_once


ROOT = copying.ROOT
OUTPUT = ROOT / "v27_v29_terminal_tail_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/v27_v29_terminal_tails_v1"
OWNERS = {"general_imputation_teacher_v27": (64, 63, 10), "legacy_strata_v29": (64, 63, 25)}
POLICY = ("Storage-only relocation of exactly V27 and V29's fully audited, stopped, nonselected step64 "
    "latest optimizer checkpoints. Their selected step10/step25 models, original teachers, baselines, archives "
    "and every other file remain untouched. Bind each original full training audit, failed fresh confirmation, "
    "owner configuration/status and selected checkpoint. Require no scientific workers or open readers. "
    "Copy/fsync/hash/mmap before atomic canonical symlink publication; independently recheck both files and "
    "all immutable inputs afterward. Preserve18/6/50 GiB checkpoint/root/scratch reserves. No source, metric, "
    "model tensor, optimizer, training trajectory, selection or qualification change. Destination is UUID-bound "
    "machine-local scratch, not provider-persistent storage or an external backup. No selected/published model "
    "moves and no evidence is deleted or recomputed.")


def checkpoint_identity(payload, current, expected, config_digest):
    step, updates, selected = expected
    if (current["state"] != "training_complete" or current["step"] != step or current["accepted_updates"] != updates
            or current["best"]["step"] != selected or payload["step"] != step or payload["attempted_batches"] != step
            or payload["gradient_updates_performed"] != updates or payload["best_step"] != selected or selected >= step
            or payload["best_score"] != current["best"]["score"] or payload["config_sha256"] != config_digest
            or payload["independent_confirmation_passed"] is not False or payload["preflight_only"] is not False
            or payload["top_two_block_training"] is not True or not isinstance(payload["model"], dict)
            or not payload["model"] or not isinstance(payload["optimizer"], dict) or not payload["optimizer"]):
        raise ValueError("Require the exact audited nonselected terminal optimizer state, never a selected model")


def require_idle():
    for name in OWNERS:
        for directory in (ROOT / name, ROOT / f"{name}_audit_task_replay", ROOT / f"{name}_confirmation"):
            readers.terminal(directory / "status.json")
    readers.terminal(ROOT / "legacy_strata_v29_stable_replay_recovery/status.json")
    readers.terminal(ROOT / "legacy_heavy_preflight_v1/status.json")
    readers.require_idle([str(ROOT / name / "latest_checkpoint.pth") for name in OWNERS])


def inventory():
    copying.require_mount()
    require_idle()
    records, inputs = [], {str(copying.UNIT): sha256(copying.UNIT)}
    for name, expected in OWNERS.items():
        directory = ROOT / name
        path, selected = directory / "latest_checkpoint.pth", directory / "selected_checkpoint.pth"
        audit_path, confirmation = ROOT / f"{name}_audit/audit.json", ROOT / f"{name}_confirmation"
        audit, gate = copying.read(audit_path), copying.read(confirmation / "gate.json")
        completed = copying.read(confirmation / "audit.json")
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev or path.samefile(selected)
                or audit["complete"] is not True or audit["complete_training_audit"] is not True
                or audit["through_step"] != expected[0] or audit["accepted_updates"] != expected[1]
                or audit["best"]["step"] != expected[2] or audit["prediction_grids"] != 1368
                or gate["passed"] is not False or gate["confirmation_evaluated"] is not True
                or gate["selected_step"] != expected[2] or gate["checkpoint_sha256"] != sha256(selected)
                or completed["complete"] is not True or completed["cases"] != 284
                or completed["prediction_grids"] != 1136 or gate["audit_sha256"] != sha256(confirmation / "audit.json")
                or (directory / "best_validation_checkpoint.pth").exists()):
            raise ValueError("Keep selected models fixed and require full original training/failed-confirmation evidence")
        digest = sha256(path)
        if audit["artifact_sha256"].get(str(path)) != digest:
            raise ValueError("The terminal bytes must retain the original independent audit hash")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, copying.read(directory / "status.json"), expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / path.name),
                        "sha256": digest, "bytes": path.stat().st_size})
        paths = [directory / filename for filename in ("config.json", "registration.json", "status.json", "selected_checkpoint.pth")]
        paths.extend([audit_path, confirmation / "audit.json", confirmation / "gate.json", confirmation / "status.json",
                      ROOT / f"{name}_audit_task_replay/status.json"])
        inputs.update({str(source): sha256(source) for source in paths})
    for name in ("legacy_strata_v29_stable_replay_recovery", "legacy_heavy_preflight_v1"):
        for filename in ("config.json", "registration.json", "status.json", "result.json"):
            source = ROOT / name / filename
            inputs[str(source)] = sha256(source)
    return records, inputs


MAIN_SOURCE = inspect.getsource(template.main)
for old, new in {
    'REPO / "tests/test_single_model_idle_root_tails.py",':
        'Path(template.__file__).resolve(), REPO / "tests/test_single_model_v27_v29_tail_relocation.py",',
    '"root_free_before": shutil.disk_usage("/").free': '"checkpoint_free_before": shutil.disk_usage(ROOT).free',
    '"root_free_after": shutil.disk_usage("/").free': '"checkpoint_free_after": shutil.disk_usage(ROOT).free',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
exec(compile(inspect.getsource(template.status), __file__, "exec"), globals())
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())
os, shutil, REPO = template.os, template.shutil, template.REPO


if __name__ == "__main__":
    main()
