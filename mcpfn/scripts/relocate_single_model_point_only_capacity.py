"""Preserve stopped diagnostic vectors and V31's unselected terminal optimizer tail."""

import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_v27_v29_terminal_tails as previous
from relocate_single_model_v27_v29_terminal_tails import readers, sha256, verify_hashes, write_json
from relocate_single_model_gradient_archive import sync_directory
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "point_only_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/point_only_capacity_v1"
CANDIDATES = ("general_imputation_teacher_preflight_v1", "anchor_rehearsal_preflight_v1",
              "anchor_small_step_preflight_v1", "legacy_strata_preflight_v1",
              "legacy_heavy_preflight_v1", "legacy_point_preflight_v1")
TAIL = "legacy_point_v31"
OWNERS = {**{name: "candidate_vector" for name in CANDIDATES}, TAIL: (64, 63, 50)}
VECTOR_NUMEL = 27766670
POLICY = ("Storage-only preservation of exactly six completed independently audited two-block preflight "
    "candidate-parameter vectors and V31's audited unselected step64 optimizer tail. Retain every selected "
    "checkpoint, teacher, published model, latest point-only preflight and all other artifacts unchanged. "
    "Require terminal absent producers, no scientific workers or open readers, original complete audit "
    "hashes and exact payload identities. Copy/fsync/hash/mmap before atomic canonical symlink publication, "
    "then independently reopen/recheck all seven artifacts and immutable inputs. Preserve18/6/50 GiB reserves. "
    "No model, optimizer, metric, training, selection, qualification, source or canonical-path change. "
    "Destination is UUID-bound machine-local scratch, not provider-persistent storage or external backup.")


def validate_vector(payload, count=VECTOR_NUMEL):
    if (not isinstance(payload, torch.Tensor) or payload.device.type != "cpu" or payload.dtype != torch.float32
            or payload.shape != (count,) or not payload.is_contiguous() or not torch.isfinite(payload).all()):
        raise ValueError("Preserve the complete finite CPU FP32 diagnostic parameter vector")


def checkpoint_identity(payload, current, expected, config_digest):
    if expected == "candidate_vector":
        if current["state"] != "complete":
            raise ValueError("Never move a live or incomplete diagnostic vector")
        validate_vector(payload)
    else:
        previous.checkpoint_identity(payload, current, expected, config_digest)


def require_idle():
    for name in (*CANDIDATES, TAIL, f"{TAIL}_pipeline", f"{TAIL}_audit_task_replay", f"{TAIL}_endpoint_replay",
                 f"{TAIL}_confirmation", "point_only_preflight_v1"):
        readers.terminal(ROOT / name / "status.json")
    protected = [str(ROOT / name / "candidate_parameters.pt") for name in CANDIDATES]
    readers.require_idle([*protected, str(ROOT / TAIL / "latest_checkpoint.pth")])


def inventory():
    previous.copying.require_mount()
    require_idle()
    records, inputs = [], {str(previous.copying.UNIT): sha256(previous.copying.UNIT)}
    for name, expected in OWNERS.items():
        directory = ROOT / name
        filename = "candidate_parameters.pt" if name in CANDIDATES else "latest_checkpoint.pth"
        path = directory / filename
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev or path.resolve() != path):
            raise ValueError("Move only the exact seven regular single-link completed artifacts")
        paths = [directory / item for item in ("config.json", "registration.json", "status.json")]
        if name in CANDIDATES:
            audit_path = directory / "audit.json"
            audit, result, gate = [previous.copying.read(directory / item) for item in ("audit.json", "result.json", "gate.json")]
            if (audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
                    or result["preflight_only"] is not True or result["trainable_parameters"] != VECTOR_NUMEL
                    or audit["frozen"]["frozen_tensors"] != 192 or gate["preflight_only"] is not True
                    or gate["audit_sha256"] != sha256(audit_path)):
                raise ValueError("Require the original complete nonselectable diagnostic audit")
            paths.extend(directory / item for item in ("audit.json", "result.json", "gate.json"))
        else:
            audit_path = ROOT / f"{TAIL}_audit/audit.json"
            audit = previous.copying.read(audit_path)
            confirmation = ROOT / f"{TAIL}_confirmation"
            gate, confirmed = [previous.copying.read(confirmation / item) for item in ("gate.json", "audit.json")]
            result_path = ROOT / f"{TAIL}_pipeline/result.json"
            result = previous.copying.read(result_path)
            selected = directory / "selected_checkpoint.pth"
            if (path.samefile(selected) or audit["complete"] is not True or audit["complete_training_audit"] is not True
                    or audit["through_step"] != 64 or audit["accepted_updates"] != 63 or audit["best"]["step"] != 50
                    or audit["prediction_grids"] != 1368 or gate["passed"] is not False
                    or gate["confirmation_evaluated"] is not True or gate["selected_step"] != 50
                    or gate["checkpoint_sha256"] != sha256(selected) or confirmed["complete"] is not True
                    or confirmed["cases"] != 284 or confirmed["prediction_grids"] != 1136
                    or gate["audit_sha256"] != sha256(confirmation / "audit.json") or result["complete"] is not True
                    or result["gate"] != gate or result["campaign"] is not None or result["review"] is not None
                    or (directory / "best_validation_checkpoint.pth").exists()):
                raise ValueError("Require full V31 rejection and preserve its selected step50 on checkpoint storage")
            paths.extend([audit_path, selected, result_path, confirmation / "gate.json", confirmation / "audit.json"])
            paths.extend(ROOT / f"{TAIL}{suffix}/status.json"
                         for suffix in ("_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation"))
        digest = sha256(path)
        if audit["artifact_sha256"].get(str(path)) != digest:
            raise ValueError("Never move bytes that differ from the original independent audit")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, previous.copying.read(directory / "status.json"), expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / filename),
                        "sha256": digest, "bytes": path.stat().st_size})
        inputs.update({str(source): sha256(source) for source in paths})
    return records, inputs


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Never publish a copy without preserving every original serialized byte")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_identity(payload, previous.copying.read(source.parent / "status.json"), OWNERS[source.parent.name],
                        sha256(source.parent / "config.json"))
    del payload


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_v27_v29_tail_relocation.py",':
        'Path(previous.__file__).resolve(), REPO / "tests/test_single_model_point_only_capacity.py",',
    'len(records) != 2': 'len(records) != 7',
    'Preserve exactly two tails': 'Preserve exactly seven completed artifacts',
    '"canonical_artifacts_checked": 2': '"canonical_artifacts_checked": 7',
    'copying_idle_nonselected_tails': 'copying_completed_point_only_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
