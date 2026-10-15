"""Qualify V23 compact training without weakening any synthetic or frozen-real gate."""

from copy import deepcopy
import hashlib

import run_single_model_paired_decoder_pipeline as baseline
import single_model_compact_training_cache as compact_cache
from single_model_registered_adaptation import replace_once, replace_assignment


MINIMUM_FREE = 18 * 1024**3
STAGE_WRITE_BUDGETS = {name: amount * 1024**2 for name, amount in {
    "replay": 768, "audit-replay": 32, "confirm": 2560, "audit-confirmation": 400,
    "benchmarks": 512, "review": 16,
}.items()}
POLICY_TEXT = ("Wait for all96 V23 attempts and the existing one-shot chain's independently completed2688-task "
    "compact regeneration/training audit; never start a duplicate trainer or auditor. Verify all22 original "
    "V8 development guards, every original teacher identity, all1368 development/reference grids and complete "
    "84-row terminal evidence with the unchanged28-row Adam proposal. Materialize only the28 final task "
    "payloads from compact capsules under the original teacher policy, preserving their exact typed full-task "
    "digests, for unchanged GPU endpoint reconstruction and independent CPU scoring. Replay the sole selected "
    "152-case development checkpoint without reselection, then consume exactly the new registered284 "
    "confirmation cases under the original V8/general guards. Audit all1136 confirmation grids before publishing "
    "any best alias. Only a synthetically qualified candidate receives every frozen tabular, four-dataset "
    "recommender and LLM benchmark and all required comparisons. Preserve18 GiB plus declared stage budgets; "
    "never repeat a failed stage, reuse consumed V22 confirmation, relax gates, hide regressions or substitute "
    "task regeneration for a full historical student-gradient replay. The V22 warm start remains unqualified.")


def process_identity(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        command = (directory / "cmdline").read_bytes().decode().replace("\x00", " ").strip()
        return {"pid": pid, "start_ticks": int(fields[19]), "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def register_chain_identity():
    current = read(training.OUTPUT / "status.json")
    child = process_identity(current["pid"])
    if child is None or "train_single_matrixformer_compact_continuation.py --stage train" not in child["command"]:
        raise ValueError("Register while the exact V23 one-shot training chain is live")
    fields = Path(f'/proc/{child["pid"]}/stat').read_text().rsplit(") ", 1)[1].split()
    parent = process_identity(int(fields[1]))
    if parent is None or "python scripts/audit_single_model_compact_continuation.py" not in parent["command"]:
        raise ValueError("Bind the live existing train-then-audit chain, not a new or inferred process")
    return parent


def wait_for_training_audit(output):
    config = read(output / "config.json")
    directory = AUDIT.with_name(AUDIT.name + "_task_replay")
    while True:
        if (directory / "status.json").exists():
            current = read(directory / "status.json")
            if current["state"] == "failed":
                raise RuntimeError(f"The original one-shot training audit failed: {current}")
            if current["state"] == "complete":
                prerequisite(output)
                return
        expected = config["training_and_audit_chain"]
        if process_identity(expected["pid"]) != expected:
            raise RuntimeError("The original chain stopped without a completed independent compact audit; do not retry")
        status(output, "waiting_for_existing_training_audit", chain_pid=expected["pid"])
        time.sleep(30)


def require_stage_capacity(label):
    if label not in STAGE_WRITE_BUDGETS:
        raise ValueError("Every remaining qualification stage needs its declared capacity budget")
    free = shutil.disk_usage(ROOT).free
    if free - STAGE_WRITE_BUDGETS[label] < MINIMUM_FREE:
        raise OSError(f"Preserve18 GiB before {label}: free={free}, budget={STAGE_WRITE_BUDGETS[label]}")


def require_compact_audit(report):
    directory = AUDIT.with_name(AUDIT.name + "_task_replay")
    result = read(directory / "result.json")
    if (read(directory / "status.json")["state"] != "complete" or result["complete"] is not True
            or result["tasks"] != 2688 or result["audit_sha256"] != sha256(AUDIT / "audit.json")
            or report["teacher_identity_checked_tasks"] != 2688
            or report["training_teacher_policy"] != training.teacher_policy.POLICY
            or report["independently_regenerated_training_tasks"] != 2688
            or report["independently_regenerated_rating_contexts"] != 192
            or report["exact_complete_task_and_teacher_digest_checks"] is not True
            or report["autograd_replayed"] is not False or report["full_optimizer_trajectory_replayed"] is not False):
        raise ValueError("Require complete independent compact input/teacher replay, not feasibility on old tasks")
    verify_hashes(read(directory / "config.json")["source_sha256"])


def materialize_endpoint_tasks(manifest, config):
    if manifest["step"] != training.STEPS or len(manifest["tasks"]) != 28:
        raise ValueError("Materialize only the entire registered final28-task endpoint")
    directory = REPLAY / "endpoint_tasks"
    directory.mkdir()
    result = deepcopy(manifest)
    with training.teacher_policy.runtime(config["base"], directory, config["teacher_identities"]):
        model, optimizer, criterion, teachers, sampler = training.original.alignment.mixture.backend.setup(config["base"])
        del model, optimizer
        for task, original in zip(result["tasks"], config["batches"][-1]["tasks"]):
            capsule_path = Path(task["artifact"])
            if sha256(capsule_path) != task["sha256"]:
                raise ValueError("Final compact task bytes changed before materialization")
            capsule = torch.load(capsule_path, map_location="cpu", weights_only=True)
            if capsule["task"] != original:
                raise ValueError("Materialize only the exact final registered task specification")
            payload = compact_cache.regenerate(capsule, sampler, teachers, criterion,
                                               training.original.alignment.mixture.backend)
            path = directory / f'task_{task["index"]:02d}.pt'
            digest = training.original.save_tensor(payload, path)
            task.update(compact_artifact=str(capsule_path), compact_sha256=task["sha256"],
                        content_sha256=capsule["content_sha256"], artifact=str(path), sha256=digest)
            del payload
    write_json(directory / "manifest.json", result)
    return result


def check_materialized_task(row):
    manifest_path = REPLAY / "endpoint_tasks" / "manifest.json"
    manifest = read(manifest_path)
    original_manifest_path = training.OUTPUT / "training_batches" / f"step_{training.STEPS:03d}" / "manifest.json"
    original_manifest = read(original_manifest_path)
    if (manifest["step"] != training.STEPS or len(manifest["tasks"]) != 28
            or manifest["entries"] != original_manifest["entries"]):
        raise ValueError("Bind the materialized manifest to the complete original final endpoint")
    matches = [task for task in manifest["tasks"] if task["name"] == row["task"]]
    if len(matches) != 1:
        raise ValueError("Every replay row must bind exactly one materialized endpoint task")
    task = matches[0]
    original = original_manifest["tasks"][task["index"]]
    restored_task = {key: value for key, value in task.items()
                     if key not in {"compact_artifact", "compact_sha256", "content_sha256"}}
    restored_task.update(artifact=task["compact_artifact"], sha256=task["compact_sha256"])
    path = REPLAY / "endpoint_tasks" / f'task_{task["index"]:02d}.pt'
    if (restored_task != original or row["artifact"] != str(path) or task["artifact"] != str(path)
            or row["artifact_sha256"] != task["sha256"] or sha256(path) != task["sha256"]
            or sha256(task["compact_artifact"]) != task["compact_sha256"]):
        raise ValueError("Original capsule, regenerated file and neural replay row must retain exact identities")
    capsule = torch.load(task["compact_artifact"], map_location="cpu", weights_only=True)
    compact_cache.validate_capsule(capsule)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (capsule["content_sha256"] != task["content_sha256"]
            or compact_cache.content_digest(payload) != capsule["content_sha256"]):
        raise ValueError("Every materialized episode/teacher byte must reproduce its original compact fingerprint")
    return payload


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_paired_decoder as training_audit':
        'import audit_single_model_compact_continuation as training_audit',
    'import train_single_matrixformer_paired_decoder as training':
        'import train_single_matrixformer_compact_continuation as training',
    'OUTPUT = ROOT / "paired_decoder_v19_pipeline"': 'OUTPUT = ROOT / "compact_continuation_v23_pipeline"',
    'AUDIT = ROOT / "paired_decoder_v19_audit"': 'AUDIT = ROOT / "compact_continuation_v23_audit"',
    'REPLAY = ROOT / "paired_decoder_v19_endpoint_replay"': 'REPLAY = ROOT / "compact_continuation_v23_endpoint_replay"',
    'CONFIRMATION = ROOT / "paired_decoder_v19_confirmation"': 'CONFIRMATION = ROOT / "compact_continuation_v23_confirmation"',
    'CAMPAIGN = ROOT / "paired_decoder_v19_benchmarks"': 'CAMPAIGN = ROOT / "compact_continuation_v23_benchmarks"',
    'REVIEW = ROOT / "paired_decoder_v19_review"': 'REVIEW = ROOT / "compact_continuation_v23_review"',
    'audit["through_step"] != 64': 'audit["through_step"] != 96',
    'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792':
        'audit["attempted_batches"] != 96 or audit["training_episodes"] != 2688',
    'audit["prediction_grids"] != 1216': 'audit["prediction_grids"] != 1368',
    '    return audit\n': '    require_compact_audit(audit)\n    return audit\n',
    'REPO / "tests/test_single_model_paired_decoder_training.py"':
        'REPO / "tests/test_single_model_compact_continuation_audit.py"',
    'REPO / "tests/test_single_model_paired_decoder_pipeline.py"':
        'REPO / "tests/test_single_model_compact_continuation_pipeline.py"',
    '    output.mkdir(parents=True)\n':
        '    chain = register_chain_identity()\n    output.mkdir(parents=True)\n',
    '"endpoint_replay": str(REPLAY), "confirmation": str(CONFIRMATION), "campaign": str(CAMPAIGN), "review": str(REVIEW)})':
        '"endpoint_replay": str(REPLAY), "confirmation": str(CONFIRMATION), "campaign": str(CAMPAIGN), "review": str(REVIEW),\n'
        '               "training_and_audit_chain": chain, "stage_write_budget_bytes": STAGE_WRITE_BUDGETS,\n'
        '               "minimum_free_bytes": MINIMUM_FREE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256})',
    '    return config\n':
        '    if (config["stage_write_budget_bytes"] != STAGE_WRITE_BUDGETS or config["minimum_free_bytes"] != MINIMUM_FREE\n'
        '            or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256):\n'
        '        raise ValueError("Preserve registered compact qualification sources and capacity gates")\n'
        '    return config\n',
    '    manifest = read(payload["task_manifest"])':
        '    manifest = materialize_endpoint_tasks(read(payload["task_manifest"]), config)',
    '        cached = torch.load(row["artifact"], map_location="cpu", weights_only=False)':
        '        cached = check_materialized_task(row)',
    '    verify_hashes(hashes)\n    verify(output)\n    write_json(REPLAY / "audit.json",':
        '    hashes[str(REPLAY / "endpoint_tasks" / "manifest.json")] = sha256(REPLAY / "endpoint_tasks" / "manifest.json")\n'
        '    verify_hashes(hashes)\n    verify(output)\n    write_json(REPLAY / "audit.json",',
    'def run_child(output, label, script, arguments, cpu=False):\n    command =':
        'def run_child(output, label, script, arguments, cpu=False):\n    require_stage_capacity(label)\n    command =',
    '    run_child(output, "training_audit", Path(training_audit.__file__).resolve(), ["--output", str(AUDIT)], cpu=True)':
        '    wait_for_training_audit(output)',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
