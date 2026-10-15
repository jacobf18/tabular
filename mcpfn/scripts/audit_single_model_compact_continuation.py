"""Independently regenerate compact training inputs and audit unchanged V23 decisions."""

from contextvars import ContextVar
import hashlib
import sys
import time

import audit_single_model_teacher_continuation as baseline
import single_model_compact_training_cache as compact_cache
import single_model_compressed_endpoint as compressed_endpoint
import single_model_composite_gradient_storage as composite_storage
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_compact_audit_replay", default=None)
check_gradient_alias = composite_storage.check_alias


def load_regenerated_task(path):
    active = REPLAY.get()
    if active is None or str(path) in active["seen"]:
        raise ValueError("Independently regenerate each declared task once, inside its audit runtime")
    capsule = torch.load(path, map_location="cpu", weights_only=True)
    digest = sha256(path)
    payload = compact_cache.regenerate(capsule, active["sampler"], active["teachers"],
                                      active["criterion"], active["backend"])
    if sha256(path) != digest:
        raise ValueError("A compact task changed during its independent regeneration")
    active["seen"].add(str(path))
    active["rating_contexts"] += int(capsule["scaling"] is not None)
    training.append_json(active["output"] / "tasks.jsonl", {"artifact": str(path), "sha256": digest,
        "content_sha256": capsule["content_sha256"], "task": capsule["task"],
        "all_episode_and_teacher_bytes_match": True, "saved_rating_vectors": capsule["scaling"] is not None})
    return payload


def regeneration_evidence(through_step):
    active = REPLAY.get()
    expected = {str(training.OUTPUT / "training_batches" / f"step_{step:03d}" / f"task_{index:02d}.pt")
                for step in range(1, through_step + 1) for index in range(28)}
    if active is None or active["seen"] != expected or active["rating_contexts"] != 2 * through_step:
        raise ValueError("Audit every new legacy/structured/rating task and teacher, not a subset")
    return {str(active["output"] / name): sha256(active["output"] / name)
            for name in ("config.json", "tasks.jsonl")}


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_teacher_continuation as training':
        'import train_single_matrixformer_compact_continuation as training',
    '    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():\n'
    '        raise ValueError("Training audit must be CPU only")':
        '    if REPLAY.get() is None:\n'
        '        raise ValueError("Compact training audit requires independent GPU task/teacher regeneration")',
    'saved["parent_training_updates"] != 56': 'saved["parent_training_updates"] != 115',
    'Never replace the full64-attempt qualification audit with a partial trial':
        'Never replace the full96-attempt qualification audit with a partial trial',
    'All152 V20 initialization grids are mandatory': 'All152 recovered V22 initialization grids are mandatory',
    'The V22 initialization must reproduce the V20 boundary64 prediction grids':
        'The V23 initialization must reproduce the recovered V22 boundary64 prediction grids',
    '            payload = torch.load(path, map_location="cpu", weights_only=False)':
        '            payload = load_regenerated_task(path)',
    '        hashes.update(shared_storage.audit(payload, raw_path))':
        '        hashes.update(compressed_endpoint.audit(payload, raw_path))',
    '    verify_hashes(hashes)\n    training.verify_registration(run)':
        '    hashes.update(regeneration_evidence(through_step))\n'
        '    verify_hashes(hashes)\n    training.verify_registration(run)',
    '"cuda_initialized": False, "autograd_replayed": False':
        '"cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False,\n'
        '               "independently_regenerated_training_tasks": len(REPLAY.get()["seen"]),\n'
        '               "independently_regenerated_rating_contexts": REPLAY.get()["rating_contexts"],\n'
        '               "exact_complete_task_and_teacher_digest_checks": True',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


def run_audit(output):
    if output.exists():
        raise FileExistsError("Never overwrite an independent compact-training audit")
    config = training.verify_registration(training.OUTPUT)
    terminal = training.read(training.OUTPUT / "status.json")
    if (terminal["state"] != "training_complete" or terminal["step"] != training.STEPS
            or Path(f'/proc/{terminal["pid"]}').exists()):
        raise ValueError("Finish and stop training before independently auditing all new compact tasks")
    directory = output.with_name(output.name + "_task_replay")
    directory.mkdir()
    sources = {Path(__file__).resolve(), Path(compact_cache.__file__).resolve(), Path(compressed_endpoint.__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(training.REPO) and Path(module.__file__).is_file())
    registration = {"training_config_sha256": sha256(training.OUTPUT / "config.json"),
        "adapted_source_sha256": ADAPTED_SOURCE_SHA256, "source_sha256": {str(path): sha256(path) for path in sources},
        "teacher_identities": config["teacher_identities"], "training_tasks": 28 * training.STEPS,
        "policy": "Independently regenerate every complete new task/teacher, then preserve all original training and endpoint audits",
        "qualification_changed": False, "training_performed": False}
    write_json(directory / "config.json", registration)
    training.status(directory, "initializing_independent_task_replay")
    try:
        with training.teacher_policy.runtime(config["base"], directory, config["teacher_identities"]):
            model, optimizer, criterion, teachers, sampler = training.original.alignment.mixture.backend.setup(config["base"])
            del model, optimizer
            torch.cuda.empty_cache()
            active = {"output": directory, "criterion": criterion, "teachers": teachers, "sampler": sampler,
                      "backend": training.original.alignment.mixture.backend, "seen": set(), "rating_contexts": 0}
            token = REPLAY.set(active)
            try:
                audit(output, training.STEPS)
            finally:
                REPLAY.reset(token)
        verify_hashes(registration["source_sha256"])
        report = training.read(output / "audit.json")
        if (report["independently_regenerated_training_tasks"] != 28 * training.STEPS
                or report["independently_regenerated_rating_contexts"] != 2 * training.STEPS):
            raise ValueError("The complete final audit must include every new compact task")
        write_json(directory / "result.json", {"complete": True, "tasks": len(active["seen"]),
            "audit_sha256": sha256(output / "audit.json"), "training_performed": False,
            "confirmation_evaluated": False, "goal_achieved": False})
        training.status(directory, "complete")
    except BaseException as error:
        training.status(directory, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "compact_continuation_v23_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
