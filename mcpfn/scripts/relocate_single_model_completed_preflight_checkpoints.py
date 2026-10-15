"""Preserve three idle, never-selectable preflight endpoints on local scratch."""

import inspect
from pathlib import Path

import torch

import relocate_single_model_idle_root_tails as template
from relocate_single_model_idle_root_tails import copying, readers, sha256, verify_hashes, write_json
from single_model_registered_adaptation import replace_once


ROOT = copying.ROOT
OUTPUT = ROOT / "completed_preflight_checkpoint_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/completed_preflight_checkpoints_v1"
OWNERS = {
    "top_two_blocks_preflight_v1": (1, "b43aeff7f6dbdef0605cfd58ba0b91f8031699484f1f2b39a0639d8bd2d1798f"),
    "general_imputation_teacher_preflight_v1": (1, "0c9f92de33a8d2de38e9a703962f68de9985a4960b22ccaabf06171cdd41bb1a"),
    "point_inclusive_preflight_v1": (0, "2362daa30719cfa940d9e26044ad5f9eff88c5739e418956caa2a2a70d2a8d06"),
}
POLICY = ("Storage-only relocation of exactly three stopped, fully audited preflight_checkpoint.pth files. "
    "Each was and remains preflight-only and selection-unqualified, with its original accepted/rejected update "
    "count. Preserve every selected model, original teacher, active/recent anchor probe output, archive and "
    "all original metadata. No live scientific worker or open reader. Verify original audit/result hashes, "
    "copy/fsync/hash/mmap each file before atomic canonical symlink publication, then independently recheck "
    "all three. Preserve18/6/50 GiB checkpoint/root/scratch reserves. This is UUID-bound machine-local scratch, "
    "not provider-persistent storage or an external backup. No formatting, resizing, source change, retry, "
    "training, evaluation, selection or qualification change.")


def checkpoint_identity(payload, current, expected, config_digest):
    updates, digest = expected
    if (current["state"] != "complete" or current["accepted"] is not bool(updates)
            or payload["step"] != 1 or payload["preflight_only"] is not True
            or payload["selection_qualified"] is not False or payload["gradient_updates_performed"] != updates
            or payload["config_sha256"] != config_digest
            or not isinstance(payload["model"], dict) or not payload["model"]
            or not isinstance(payload["optimizer"], dict)):
        raise ValueError("Preserve the exact never-selectable audited diagnostic endpoint")


def require_idle():
    for name in OWNERS:
        readers.terminal(ROOT / name / "status.json")
    readers.require_idle([str(ROOT / name / "preflight_checkpoint.pth") for name in OWNERS])


def inventory():
    copying.require_mount()
    require_idle()
    records, inputs = [], {str(copying.UNIT): sha256(copying.UNIT)}
    for name, expected in OWNERS.items():
        directory = ROOT / name
        path = directory / "preflight_checkpoint.pth"
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev):
            raise ValueError("Move only original single-link diagnostic files on checkpoint storage")
        current, result, audit, gate = [copying.read(directory / filename)
                                      for filename in ("status.json", "result.json", "audit.json", "gate.json")]
        if (audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
                or result["preflight_only"] is not True or result["checkpoint_sha256"] != expected[1]
                or gate["preflight_only"] is not True or gate["passed"] is not bool(expected[0])
                or gate["audit_sha256"] != sha256(directory / "audit.json")
                or audit["artifact_sha256"].get(str(path)) != expected[1] or sha256(path) != expected[1]):
            raise ValueError("Require exact complete original independent diagnostic audits")
        verify_hashes(audit["artifact_sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, current, expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / path.name),
                        "sha256": expected[1], "bytes": path.stat().st_size})
        for filename in ("status.json", "config.json", "registration.json", "result.json", "audit.json", "gate.json"):
            source = directory / filename
            inputs[str(source)] = sha256(source)
    return records, inputs


exec(compile(inspect.getsource(template.status), __file__, "exec"), globals())
MAIN_SOURCE = inspect.getsource(template.main)
for old, new in {
    "len(records) != 2": "len(records) != 3",
    "Preserve exactly two tails": "Preserve exactly three diagnostic endpoints",
    '"copying_idle_nonselected_tails"': '"copying_idle_preflight_endpoints"',
    "optimizer-tail relocation": "preflight-checkpoint relocation",
    'REPO / "tests/test_single_model_idle_root_tails.py",':
        'Path(template.__file__).resolve(), Path(__file__).resolve().parents[1] / "tests/test_single_model_preflight_checkpoint_relocation.py",',
    '"root_free_before": shutil.disk_usage("/").free': '"checkpoint_free_before": shutil.disk_usage(ROOT).free',
    '"canonical_artifacts_checked": 2': '"canonical_artifacts_checked": 3',
    '"root_free_after": shutil.disk_usage("/").free': '"checkpoint_free_after": shutil.disk_usage(ROOT).free',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())

os, shutil, REPO = template.os, template.shutil, template.REPO


if __name__ == "__main__":
    main()
