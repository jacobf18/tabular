"""Recover root capacity from proven-idle legacy snapshots, preserving all checkpoint bytes."""

from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import torch

import compress_single_model_gradient_archive as archive
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = REPO / "runs/single_model_goal_20261005"
OUTPUT = ROOT / "legacy_snapshot_capacity_v1"
WORK = REPO / "src/tabimpute/workdir"
QUEUES = REPO / "runs/objective_queues"
NAMES = ("spectral_lr_20260912", "spectral_query_20260910", "spectral_shrinkage_20260910",
         "rowcol_output_remedies_paired_20260907", "rowcol_controlled_20260906",
         "spectral_geometry_20260914", "spectral_geometry_continue_20260915")
MINIMUM_ROOT_FREE = 3 * 1024**3
TARGET_ROOT_FREE = 24 * 1024**3
WRITERS = {"train_direct_pure_rabel_curriculum.py", "run_spectral_lr_followups.py", "run_spectral_query_experiments.py",
           "run_spectral_shrinkage_experiments.py", "run_rowcol_output_remedies.py", "run_rowcol_controlled_experiments.py",
           "run_spectral_geometry_experiments.py", "run_spectral_geometry_continuations.py"}
POLICY = ("Storage only: inspect fixed completed legacy queues, hold their existing exclusive locks and reject live "
          "legacy writers. Archive only numeric/phase snapshots and terminal latest checkpoints, never best or "
          "best-validation files, code, data or benchmark results. Preserve every original path and serialized byte "
          "with persistent read-only mmap-readable SquashFS mounts on root. Stop only at24 GiB measured root free "
          "space or after the fixed candidate inventory is exhausted. Maintain3 GiB root staging reserve and leave "
          "the checkpoint volume's18 GiB reserve untouched. No training, scores, selections or scientific audit "
          "claims are changed. Terminal metadata is storage provenance, not a new audit of the old experiments.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def writer_arguments(arguments):
    return any(Path(argument).name in WRITERS for argument in arguments)


def require_idle():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if writer_arguments(arguments):
            raise ValueError(f"A legacy training/queue writer is live: {process.name}")


@contextmanager
def queue_locks():
    with ExitStack() as stack:
        for name in NAMES:
            stream = stack.enter_context((QUEUES / name / "queue.lock").open("r"))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle()
        yield


def snapshot_epoch(name, final_epoch):
    if name == "latest_checkpoint.pth":
        return final_epoch
    matched = re.fullmatch(r"checkpoint_(?:phase\d+_)?(\d+)\.pth", name)
    if not matched or not 0 < int(matched.group(1)) <= final_epoch:
        raise ValueError("Only explicit non-selected numeric/phase snapshots or terminal latest checkpoints may move")
    return int(matched.group(1))


def queue_runs(name, terminal, results):
    if (terminal.get("state") != "complete" or terminal.get("active") != "complete" or not isinstance(results, list)
            or not results or sorted(terminal.get("completed", [])) != sorted(row["tag"] for row in results)
            or len({row["tag"] for row in results}) != len(results)):
        raise ValueError("Require the complete original queue and every completed result identity")
    runs = []
    for row in results:
        source = Path(row.get("work_dir") or row.get("horizon", {}).get("work_dir", ""))
        expected = WORK / f'tabimpute-{name}-{row["tag"]}'
        if source != expected:
            if source.parent != WORK or not source.name.startswith("tabimpute-rowcol_controlled_20260906-"):
                raise ValueError("Unexpected legacy work directory; never infer unrelated archival ownership")
            continue
        runs.append(source)
    return runs


def inventory():
    require_idle()
    records, inputs, untouched = [], {}, {}
    for name in NAMES:
        queue = QUEUES / name
        terminal, results = read(queue / "status.json"), read(queue / "results.json")
        for filename in ("status.json", "results.json", "manifest.json"):
            inputs[str(queue / filename)] = sha256(queue / filename)
        for source in queue_runs(name, terminal, results):
            config = read(source / "config.json")
            if source.is_symlink() or config["run_name"] != source.name:
                raise ValueError("Require the original explicitly named legacy training directory")
            final_epoch = config["curriculum"][-1]["end_step"]
            if type(final_epoch) is not int or final_epoch <= 0 or not (source / f"checkpoint_{final_epoch}.pth").is_file():
                raise ValueError("Require the declared final training checkpoint")
            inputs[str(source / "config.json")] = sha256(source / "config.json")
            candidates = []
            for path in source.glob("*.pth"):
                if path.name.startswith("checkpoint_") or path.name == "latest_checkpoint.pth":
                    epoch = snapshot_epoch(path.name, final_epoch)
                    stat = path.lstat()
                    if (path.is_symlink() or not path.is_file() or stat.st_nlink != 1
                            or stat.st_mtime > terminal["updated_unix"]):
                        raise ValueError("Require idle original single-link checkpoint bytes predating queue completion")
                    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                    if payload["epoch"] != epoch or not isinstance(payload.get("model"), dict) or not payload["model"]:
                        raise ValueError("Checkpoint identity must match its original model and declared epoch")
                    del payload
                    candidates.append({"source": str(path), "sha256": sha256(path), "bytes": stat.st_size,
                                       "queue": name, "epoch": epoch})
                elif path.name in {"best_checkpoint.pth", "best_validation_checkpoint.pth"}:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("Require selected legacy checkpoints to remain original regular files")
                    untouched[str(path)] = sha256(path)
            if not candidates:
                raise ValueError("Every declared original run must retain a numeric snapshot")
            records.extend(candidates)
    if len({row["source"] for row in records}) != len(records):
        raise ValueError("Never count a reused control directory twice")
    return sorted(records, key=lambda row: (row["bytes"], row["source"])), inputs, untouched


COMPRESSOR_SOURCE = replace_once(inspect.getsource(archive.compress_group),
    'Require net savings and21 GiB free before replacing any original',
    'Require net savings and the registered3 GiB root floor before replacing any original')
COMPRESSOR_SOURCE_SHA256 = hashlib.sha256(COMPRESSOR_SOURCE.encode()).hexdigest()


def compressor():
    namespace = {**archive.__dict__, "ROOT": ROOT, "MINIMUM_FREE": MINIMUM_ROOT_FREE}
    exec(compile(COMPRESSOR_SOURCE, __file__, "exec"), namespace)
    return namespace["compress_group"]


def recheck(config, results):
    hashes = {row["source"]: row["sha256"] for row in config["records"]}
    hashes.update(config["untouched_selected_checkpoints"])
    for index, result in enumerate(results):
        group = Path(result["group"])
        report = read(group / "audit.json")
        row = config["records"][index]
        source = Path(row["source"])
        if (not report["complete"] or report["records"] != [row] or not report["torch_mmap_readable"]
                or not source.is_symlink() or source.resolve() != group / "files" / source.name):
            raise ValueError("Every archived original must resolve to its complete lossless image")
        hashes.update({str(group / "audit.json"): result["audit_sha256"], report["image"]: report["image_sha256"],
                       report["unit"]: report["unit_sha256"]})
        for action in ("is-active", "is-enabled"):
            subprocess.run(["systemctl", action, "--quiet", Path(report["unit"]).name], check=True)
        options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(group / "files")], text=True).strip().split(",")
        if "ro" not in options:
            raise ValueError("Every legacy snapshot archive must remain persistent and read-only")
    for row in config["records"][len(results):]:
        if Path(row["source"]).is_symlink():
            raise ValueError("Untouched candidate paths must not change type")
    if any(Path(path).is_symlink() for path in config["untouched_selected_checkpoints"]):
        raise ValueError("Selected legacy checkpoints must remain untouched regular files")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return hashes


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart this capacity archive")
    with queue_locks():
        records, inputs, untouched = inventory()
        sources = [Path(__file__).resolve(), Path(archive.__file__).resolve(),
                   REPO / "scripts/single_model_registered_adaptation.py", REPO / "tests/test_single_model_legacy_snapshot_archive.py"]
        OUTPUT.mkdir()
        config = {"policy": POLICY, "records": records, "input_sha256": inputs, "untouched_selected_checkpoints": untouched,
                  "source_sha256": {str(path): sha256(path) for path in sources}, "minimum_root_free": MINIMUM_ROOT_FREE,
                  "target_root_free": TARGET_ROOT_FREE, "compressor_source_sha256": COMPRESSOR_SOURCE_SHA256}
        write_json(OUTPUT / "config.json", config)
        status("archiving_legacy_snapshots", candidates=len(records))
        results = []
        compress = compressor()
        for index, row in enumerate(records):
            if shutil.disk_usage(ROOT).free >= TARGET_ROOT_FREE:
                break
            require_idle()
            verify_hashes(inputs)
            if sha256(row["source"]) != row["sha256"]:
                raise ValueError("Original registered checkpoint bytes changed")
            result = compress(OUTPUT / f"group_{index:03d}", [row])
            results.append(result)
            write_json(OUTPUT / "progress.json", {"results": results, "root_free_bytes": shutil.disk_usage(ROOT).free})
            status("archiving_legacy_snapshots", completed=len(results), candidates=len(records), root_free_bytes=shutil.disk_usage(ROOT).free)
            print(json.dumps(result), flush=True)
        hashes = recheck(config, results)
        free = shutil.disk_usage(ROOT).free
        if free < MINIMUM_ROOT_FREE:
            raise OSError("Never spend the root write reserve")
        write_json(OUTPUT / "independent_recheck.json", {"complete": True, "archived_checkpoints": len(results),
                   "candidate_checkpoints": len(records), "untouched_selected_checkpoints": len(untouched),
                   "net_bytes_reclaimed": sum(row["net_bytes_reclaimed"] for row in results), "root_free_bytes": free,
                   "capacity_target_reached": free >= TARGET_ROOT_FREE, "artifact_sha256": hashes,
                   "config_sha256": sha256(OUTPUT / "config.json"), "all_original_bytes_and_paths_preserved": True,
                   "scientific_training_or_evaluation_performed": False})
        status("complete", archived_checkpoints=len(results), capacity_target_reached=free >= TARGET_ROOT_FREE)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
