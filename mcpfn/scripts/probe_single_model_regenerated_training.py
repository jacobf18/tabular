"""Test exact regeneration of completed V22 synthetic inputs and teacher targets."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import recover_single_model_cached_continuation as training
import run_single_model_cached_continuation_pipeline as pipeline


OUTPUT = training.ROOT / "teacher_continuation_v22_regeneration_probe_v1"
POLICY = ("One non-promotable diagnostic after completed negative V22 qualification. Regenerate every one of "
    "the original64 batches/1792 synthetic episodes from its registered specification using the unchanged "
    "sampler and original three training teachers. Do not restore cached episodes or replay student backward "
    "RNG consumption. Require exact tensor dtype, shape, stride and bytes, complete metadata/routing/task "
    "equality and teacher-target bytes against every original cached artifact. Preserve NaN payloads and "
    "signed zero in tensor comparisons. Save only typed content digests and comparison records on success; "
    "preserve one generated mismatch artifact for diagnosis on failure, then stop without retry or tolerance "
    "search. This tests whether a future prospectively registered training procedure could use regeneration "
    "and per-task digests to reduce new cache storage. It does not authorize deleting old caches, replacing "
    "existing audits, claiming a replay of student gradients or training, changing priors, reselecting a "
    "checkpoint, repeating confirmation or running benchmarks. No model or optimizer update is performed.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_equal(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.layout != torch.strided or expected.layout != torch.strided
                or actual.device.type != "cpu" or expected.device.type != "cpu"
                or actual.stride() != expected.stride() or not training.storage.byte_equal(actual, expected)):
            raise ValueError(f"Regenerated tensor differs exactly: {label}")
    elif type(actual) is not type(expected):
        raise ValueError(f"Regenerated value type differs: {label}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys() or any(not isinstance(key, str) for key in expected):
            raise ValueError(f"Regenerated mapping keys differ: {label}")
        for key in expected:
            require_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"Regenerated sequence length differs: {label}")
        for index, (current, original) in enumerate(zip(actual, expected)):
            require_equal(current, original, f"{label}[{index}]")
    elif expected is None or isinstance(expected, (str, bool, int, float)):
        training.probe.require_same(actual, expected, label)
    else:
        raise TypeError(f"Unsupported exact regeneration value at {label}: {type(expected)}")


def content_digest(payload):
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("Hash complete CPU strided tensor contents only")
            header = ["tensor", str(value.dtype), list(value.shape), list(value.stride())]
            digest.update(json.dumps(header, separators=(",", ":")).encode())
            raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(str(len(raw)).encode() + b":" + raw)
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Use only explicit string mapping keys")
            digest.update(b"dict[")
            for key in sorted(value):
                update(key)
                update(value[key])
            digest.update(b"]")
        elif isinstance(value, (list, tuple)):
            digest.update((type(value).__name__ + "[").encode())
            for nested in value:
                update(nested)
            digest.update(b"]")
        elif value is None or isinstance(value, (str, bool, int, float)):
            encoded = json.dumps([type(value).__name__, value], allow_nan=False, separators=(",", ":")).encode()
            digest.update(str(len(encoded)).encode() + b":" + encoded)
        else:
            raise TypeError(f"Unsupported fingerprint value: {type(value)}")

    update(payload)
    return digest.hexdigest()


def terminal_evidence():
    paths = [training.OUTPUT / "status.json", pipeline.OUTPUT / "status.json", pipeline.OUTPUT / "result.json",
             pipeline.AUDIT / "audit.json", pipeline.TEACHERS / "audit.json", pipeline.REPLAY / "audit.json",
             pipeline.CONFIRMATION / "audit.json", pipeline.CONFIRMATION / "gate.json"]
    terminal, finished, result, audit, teachers, replay, confirmation, gate = map(read, paths)
    if (terminal["state"] != "training_complete" or terminal["step"] != 64 or terminal["accepted_updates"] != 59
            or finished["state"] != "complete" or any(Path(f'/proc/{row["pid"]}').exists() for row in (terminal, finished))
            or not result["complete"] or result["gate"] != gate or gate["passed"] or not gate["confirmation_evaluated"]
            or gate["cases"] != 284 or gate["selected_step"] != 64 or result["campaign"] is not None or result["review"] is not None
            or not audit["complete_training_audit"] or audit["through_step"] != 64 or audit["training_episodes"] != 1792
            or not teachers["complete"] or teachers["full_grids"] != 1792 or not teachers["full_grid_bitwise_equal"]
            or not replay["complete"] or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or not confirmation["complete"] or confirmation["prediction_grids"] != 1136
            or (training.OUTPUT / "best_validation_checkpoint.pth").exists()
            or pipeline.CAMPAIGN.exists() or pipeline.REVIEW.exists()):
        raise ValueError("Require the finished independently audited negative V22 qualification")
    for report in (audit, teachers, replay, confirmation):
        verify_hashes(report["artifact_sha256"])
    return {str(path): sha256(path) for path in paths}


def register():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or repeat the exact-regeneration diagnostic")
    config = training.verify_registration(training.OUTPUT)
    hashes = terminal_evidence()
    for batch in config["batches"]:
        path = training.original_manifest(batch["step"])
        hashes[str(path)] = sha256(path)
        hashes.update({row["artifact"]: row["sha256"] for row in read(path)["tasks"]})
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_regenerated_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "batches": config["batches"], "tasks": 1792,
        "teacher_identities": config["teacher_identities"], "input_sha256": hashes,
        "source_sha256": {**config["source_sha256"], **{str(path): sha256(path) for path in sources}}})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"), "registered_unix": time.time(),
        "new_training_cases": 0, "training_performed": False, "confirmation_repeated": False})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    original = training.verify_registration(training.OUTPUT)
    if (sha256(OUTPUT / "config.json") != read(OUTPUT / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["batches"] != original["batches"] or config["tasks"] != 1792
            or config["teacher_identities"] != original["teacher_identities"]):
        raise ValueError("Preserve the complete declared diagnostic and exact comparison policy")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return original


@torch.no_grad()
def run():
    config = verify()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a begun exact-regeneration diagnostic")
    training.wait_for_gpu(OUTPUT, True, False)
    status("loading_original_sampler_and_teachers")
    count, roles = 0, {"initialization": 0, "general": 0, "specialist": 0}
    with training.teacher_policy.runtime(config["base"], OUTPUT, config["teacher_identities"]):
        model, optimizer, criterion, teachers, sampler = training.original.alignment.mixture.backend.setup(config["base"])
        del model, optimizer
        for batch in config["batches"]:
            manifest_path = training.original_manifest(batch["step"])
            manifest = read(manifest_path)
            for task, logged in zip(batch["tasks"], manifest["tasks"]):
                path = manifest_path.parent / f'task_{task["index"]:02d}.pt'
                if str(path) != logged["artifact"] or sha256(path) != logged["sha256"]:
                    raise ValueError("Bind every original cached artifact before independent regeneration")
                cached = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                training.preflight.task_audit.check_task(cached, task, logged)
                role = training.preflight.check_routing(cached, task, config)
                episode = sampler.sample(step=1, **task["specification"])
                generated = {"episode": training.original.alignment.mixture.backend.cache_episode(episode), "task": task}
                try:
                    require_equal(generated["episode"], cached["episode"], "episode")
                    target, routing = training.original.alignment.mixture.backend.teacher_target(teachers, criterion, episode)
                    generated.update(teacher=target.detach().float().cpu(), routing=routing)
                    require_equal(generated, cached, "complete_task")
                    digest = content_digest(generated)
                    if digest != content_digest(cached):
                        raise ValueError("Exact task equality must also reproduce its typed content digest")
                except BaseException:
                    training.original.save_tensor(generated, OUTPUT / "first_generated_mismatch.pt")
                    write_json(OUTPUT / "mismatch.json", {"step": batch["step"], "task": task, "artifact": str(path),
                        "artifact_sha256": logged["sha256"], "generated_sha256": sha256(OUTPUT / "first_generated_mismatch.pt")})
                    raise
                count += 1
                roles[role] += 1
                training.append_json(OUTPUT / "tasks.jsonl", {"step": batch["step"], "index": task["index"],
                    "name": task["name"], "role": role, "artifact": str(path), "artifact_sha256": logged["sha256"],
                    "exact_episode_and_teacher": True, "content_sha256": digest})
                status("regenerating", completed=count, expected=1792, step=batch["step"], task=task["name"])
                del cached, generated, episode, target
    if count != 1792 or roles != {"initialization": 128, "general": 1536, "specialist": 128}:
        raise ValueError("Never omit any original batch, episode or teacher role")
    verify()
    write_json(OUTPUT / "result.json", {"complete": True, "tasks": count, "role_counts": roles,
        "all_episode_and_teacher_bytes_match": True, "records_sha256": sha256(OUTPUT / "tasks.jsonl"),
        "student_gradients_replayed": False, "training_performed": False, "model_saved": False,
        "cached_artifacts_deleted": False, "confirmation_repeated": False, "real_benchmarks_evaluated": False,
        "independent_audit_of_this_diagnostic": False})
    status("complete", tasks=count, diagnostic_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except BaseException as error:
                status("failed", error=repr(error))
                raise
