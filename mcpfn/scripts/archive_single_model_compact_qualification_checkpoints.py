"""Co-archive ten unqualified idle checkpoints without changing their serialized bytes."""

import json
import os
from pathlib import Path
import subprocess
import time

import torch

import archive_single_model_decoder_checkpoints as storage
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = storage.ROOT
OUTPUT = ROOT / "compact_qualification_checkpoint_archive_v1"
GROUPS = (
    tuple((name, "selected_checkpoint.pth") for name in
          ("v8_decoder_v17", "replicated_decoder_v18", "paired_decoder_v19", "structured_teacher_v20")),
    (("structured_teacher_v20", "latest_checkpoint.pth"), ("point_inclusive_v21", "latest_checkpoint.pth")),
    (("teacher_continuation_v22", "latest_checkpoint.pth"),
     ("teacher_continuation_v22_cached_recovery_v1", "latest_checkpoint.pth")),
    (("teacher_continuation_v22", "selected_checkpoint.pth"),
     ("teacher_continuation_v22_cached_recovery_v1", "selected_checkpoint.pth")),
)
POLICY = ("Losslessly co-archive exactly ten fully serialized model/optimizer checkpoints belonging to idle "
          "unqualified V17-V22 runs, including both original failed V22 and its separately audited recovery. "
          "Bind existing independent audit/recovery input hashes and preserve each canonical read, mmap, "
          "original failure, selection and confirmation outcome. Never touch a published best alias, original "
          "V8/general/specialist baseline, active V23 output, prior source or existing archive. Use the existing "
          "RAM-built read-only persistent squashfs co-archiver and its21 GiB concurrent-writer staging floor. "
          "Readers already mmaping old inodes remain valid; their space is released only when readers exit. "
          "Reported net content savings are not a promise of immediate free-space change during live training.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def inventory():
    inputs, audited = {}, {}
    names = sorted({name for group in GROUPS for name, filename in group})
    for name in names:
        directory = ROOT / name
        current = read(directory / "status.json")
        expected = "failed" if name == "teacher_continuation_v22" else "training_complete"
        if current["state"] != expected or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Archive only idle terminal original checkpoints; preserve the failed V22 state")
        for filename in ("config.json", "registration.json", "status.json"):
            path = directory / filename
            inputs[str(path)] = sha256(path)
        if name == "teacher_continuation_v22":
            path = ROOT / "teacher_continuation_v22_cached_recovery_v1/recovery_config.json"
            report = read(path)
            audited[name] = report["input_sha256"]
        else:
            path = ROOT / f"{name}_audit/audit.json"
            report = read(path)
            if not report["complete"] or not report["complete_training_audit"] or report["through_step"] != 64:
                raise ValueError("Bind each original complete independent64-attempt checkpoint audit")
            audited[name] = report["artifact_sha256"]
        inputs[str(path)] = sha256(path)
    groups = []
    for group in GROUPS:
        records = []
        for name, filename in group:
            source = ROOT / name / filename
            expected = audited[name][str(source)]
            if source.is_symlink() or source.stat().st_nlink != 1 or sha256(source) != expected:
                raise ValueError("Archive only each original unlinked, independently fingerprinted checkpoint")
            if filename == "selected_checkpoint.pth" and name != "teacher_continuation_v22":
                result_path = ROOT / f"{name}_pipeline/result.json"
                result = read(result_path)
                if (not result["complete"] or result["gate"]["passed"] is not False
                        or result["campaign"] is not None or result["review"] is not None):
                    raise ValueError("Selected candidates must retain their completed failed qualification")
                inputs[str(result_path)] = sha256(result_path)
            records.append({"source": str(source), "sha256": expected, "archive_name": f"{name}__{filename}"})
        groups.append(records)
    if sum(map(len, groups)) != 10:
        raise ValueError("Preserve the exact ten-checkpoint inventory")
    return groups, inputs


def recheck(config, results):
    hashes = {}
    for records, result in zip(config["groups"], results):
        directory = Path(result["group"])
        report = read(directory / "audit.json")
        if report["records"] != records or not report["complete"] or (directory / "named_inputs").exists():
            raise ValueError("Account for every byte and release staging hardlinks only after verified replacement")
        hashes.update({str(directory / "audit.json"): result["audit_sha256"],
                       report["image"]: report["image_sha256"], report["unit"]: report["unit_sha256"]})
        for action in ("is-active", "is-enabled"):
            subprocess.run(["systemctl", action, "--quiet", Path(report["unit"]).name], check=True)
        options = subprocess.check_output(["findmnt", "-rn", "-o", "OPTIONS", "--mountpoint", str(directory / "files")], text=True).strip().split(",")
        if "ro" not in options:
            raise ValueError("Every persistent archive must remain read-only")
        for row in records:
            source = Path(row["source"])
            if not source.is_symlink() or source.resolve() != directory / "files" / row["archive_name"]:
                raise ValueError("Every canonical checkpoint must retain its exact declared serialized bytes")
            restored = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(restored, dict) or "model" not in restored:
                raise ValueError("Every canonical model checkpoint must retain torch mmap readability")
            del restored
            hashes[str(source)] = row["sha256"]
    if len(results) != len(config["groups"]):
        raise ValueError("Independently recheck all four completed archive groups")
    verify_hashes(hashes)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    write_json(OUTPUT / "independent_recheck.json", {"complete": True, "canonical_checkpoints_checked": 10,
        "all_original_model_optimizer_metadata_bytes_preserved": True, "all_canonical_mmaps_readable": True,
        "published_baselines_and_live_training_outputs_untouched": True, "selected_candidates_losslessly_archived": True,
        "persistent_read_only_mounts_checked": len(results), "artifact_sha256": hashes,
        "net_content_bytes_reclaimed": sum(result["net_bytes_reclaimed"] for result in results),
        "mmap_readers_may_defer_physical_inode_reclamation": True,
        "config_sha256": sha256(OUTPUT / "config.json"), "qualification_or_goal_changed": False})


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a checkpoint capacity attempt")
    groups, inputs = inventory()
    sources = [Path(__file__).resolve(), Path(storage.__file__).resolve(), Path(storage.archive.__file__).resolve(),
               REPO / "tests/test_single_model_compact_qualification_archive.py"]
    config = {"policy": POLICY, "groups": groups, "input_sha256": inputs,
        "source_sha256": {str(path): sha256(path) for path in sources},
        "compressor_source_sha256": storage.COMPRESSOR_SOURCE_SHA256, "minimum_free_bytes": storage.archive.MINIMUM_FREE}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    status("archiving_idle_checkpoints")
    try:
        results = []
        for index, records in enumerate(groups):
            result = storage.compressor()(OUTPUT / f"group_{index:02d}", records)
            results.append(result)
            print(json.dumps(result), flush=True)
        status("independent_recheck")
        recheck(config, results)
        status("complete", net_content_bytes_reclaimed=sum(result["net_bytes_reclaimed"] for result in results))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
