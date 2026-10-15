"""Preserve V30's audited nonselected terminal checkpoint before another training run."""

import types
from pathlib import Path

import torch

import relocate_single_model_v27_v29_terminal_tails as previous
from relocate_single_model_v27_v29_terminal_tails import copying, readers, sha256, verify_hashes, write_json
from single_model_registered_adaptation import replace_once


ROOT = copying.ROOT
OUTPUT = ROOT / "v30_terminal_tail_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/v30_terminal_tail_v1"
OWNERS = {"legacy_heavy_v30": (64, 64, 0)}
POLICY = ("Storage-only relocation of V30's fully independently audited, stopped, nonselected step64 latest "
    "optimizer checkpoint. Preserve V30's lack of selection, V29 initializer, teachers and every other file. "
    "Bind complete training and endpoint audits, original nonselection result and skipped confirmation. "
    "Require stopped scientific workers and no open readers; copy/fsync/hash/mmap before atomic canonical "
    "symlink publication, then independently recheck all original bytes and inputs. Preserve18/6/50 GiB "
    "reserves. No source, tensor, optimizer, trajectory, selection or qualification changes or recomputation. "
    "Destination is UUID-bound machine-local scratch, not provider-persistent storage or external backup.")
checkpoint_identity = previous.checkpoint_identity


def require_idle():
    for name in ("legacy_heavy_v30", "legacy_heavy_v30_pipeline", "legacy_heavy_v30_audit_task_replay",
                 "legacy_heavy_v30_endpoint_replay", "legacy_heavy_v30_confirmation", "legacy_point_preflight_v1"):
        readers.terminal(ROOT / name / "status.json")
    readers.require_idle([str(ROOT / "legacy_heavy_v30/latest_checkpoint.pth")])


def inventory():
    copying.require_mount()
    require_idle()
    directory = ROOT / "legacy_heavy_v30"
    path = directory / "latest_checkpoint.pth"
    selected = ROOT / "legacy_strata_v29/selected_checkpoint.pth"
    audit_path = ROOT / "legacy_heavy_v30_audit/audit.json"
    replay_path = ROOT / "legacy_heavy_v30_endpoint_replay/audit.json"
    result_path = ROOT / "legacy_heavy_v30_pipeline/result.json"
    gate_path = ROOT / "legacy_heavy_v30_confirmation/gate.json"
    audit, replay, result, gate = [copying.read(source) for source in (audit_path, replay_path, result_path, gate_path)]
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or path.stat().st_dev != ROOT.stat().st_dev or path.samefile(selected)
            or audit["complete"] is not True or audit["complete_training_audit"] is not True
            or audit["through_step"] != 64 or audit["accepted_updates"] != 64 or audit["best"]["step"] != 0
            or audit["prediction_grids"] != 1368 or replay["complete"] is not True
            or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 0
            or result["complete"] is not True or result["training_audit_sha256"] != sha256(audit_path)
            or result["gate"] != gate or result["campaign"] is not None or result["review"] is not None
            or gate["passed"] is not False or gate["selected_step"] != 0 or gate["confirmation_evaluated"] is not False
            or (directory / "selected_checkpoint.pth").exists() or (directory / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Require exact stopped audited V30 nonselection, with no confirmation or published alias")
    digest = sha256(path)
    if audit["artifact_sha256"].get(str(path)) != digest:
        raise ValueError("Preserve the original independently audited terminal checkpoint hash")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_identity(payload, copying.read(directory / "status.json"), OWNERS[directory.name], sha256(directory / "config.json"))
    del payload
    paths = [copying.UNIT, audit_path, replay_path, result_path, gate_path, selected]
    paths.extend(directory / name for name in ("config.json", "registration.json", "status.json"))
    for name in ("legacy_heavy_v30_pipeline", "legacy_heavy_v30_endpoint_replay", "legacy_heavy_v30_confirmation",
                 "legacy_heavy_v30_audit_task_replay", "legacy_point_preflight_v1"):
        paths.append(ROOT / name / "status.json")
    return [{"source": str(path), "destination": str(DESTINATION / directory.name / path.name),
             "sha256": digest, "bytes": path.stat().st_size}], {str(source): sha256(source) for source in paths}


template = previous.template
MAIN_SOURCE = replace_once(previous.MAIN_SOURCE,
    'REPO / "tests/test_single_model_v27_v29_tail_relocation.py",',
    'Path(previous.__file__).resolve(), REPO / "tests/test_single_model_v30_tail_relocation.py",')
MAIN_SOURCE = replace_once(MAIN_SOURCE, '"canonical_artifacts_checked": 2', '"canonical_artifacts_checked": 1')
MAIN_SOURCE = replace_once(MAIN_SOURCE, 'len(records) != 2', 'len(records) != 1')
MAIN_SOURCE = replace_once(MAIN_SOURCE, 'Preserve exactly two tails', 'Preserve exactly one nonselected tail')
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())
os, shutil, REPO = template.os, template.shutil, template.REPO


if __name__ == "__main__":
    main()
