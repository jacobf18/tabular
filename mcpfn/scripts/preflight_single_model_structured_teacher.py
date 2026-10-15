"""Fresh non-promotable paired optimizer test with structured-general teacher targets."""

import hashlib

import torch

import preflight_single_model_paired_decoder as previous_preflight
import run_single_model_paired_decoder_pipeline as preceding
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import single_model_structured_teacher as teacher_policy
from single_model_registered_adaptation import replace_once, replace_assignment
from tabimpute.streamed_inference import streamed_point_predictions as teacher_predictions


POLICY_TEXT = ("One fresh28-task decoder-only optimizer preflight directly from original V8 after V19's completed "
               "unsuccessful independent qualification. Keep V19 first-batch priors, shapes, normalization, losses, "
               ".25 legacy/.25 structured/.5 ratings weights, paired training protection,2e-5 Adam proposal,0.1 "
               "margin,2x norm bound and exact finite rollback. Change only training teacher assignment: V8 for "
               "legacy imputation, original general for every structured family, original specialist for ratings. "
               "Verify actual frozen teacher tensors against original checkpoint hashes and independently replay "
               "every full cached teacher target in a fresh process. Preserve all raw gradients, optimizer states "
               "and before/candidate predictions for independent CPU audit. No development, confirmation, "
               "selection, real benchmarks, promotion, inference ensemble or future initialization from this endpoint. "
               "No completed source, seed, qualification threshold or protocol changes.")


def require_space():
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise OSError("Reserve raw28-task evidence and the unchanged18 GiB atomic-write floor")


def require_predecessor():
    result, current = read(preceding.OUTPUT / "result.json"), read(preceding.OUTPUT / "status.json")
    if (not result["complete"] or result["gate"]["passed"] or not result["gate"]["confirmation_evaluated"]
            or current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or result["campaign"] is not None or result["review"] is not None):
        raise ValueError("Require terminal V19 with failed independent qualification and no real campaign")
    preceding.prerequisite(preceding.OUTPUT)
    for directory, key, count in ((preceding.REPLAY, "endpoint_grids", 56),
                                  (preceding.CONFIRMATION, "prediction_grids", 1136)):
        audit = read(directory / "audit.json")
        if not audit["complete"] or audit[key] != count:
            raise ValueError("Preserve complete independent V19 endpoint and confirmation audits")
        verify_hashes(audit["artifact_sha256"])
    if (preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Never compete with a published preceding model")


def register_trial(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use only the fresh structured-teacher preflight")
    require_space()
    require_predecessor()
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    identities = teacher_policy.identities(base, PARENT)
    if PARENT != training.alignment.fixed.PARENT or sha256(PARENT) != identities["initialization"]["sha256"]:
        raise ValueError("Initialize only from original V8")
    earlier = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in earlier))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 56 or fresh.intersection(used):
        raise ValueError("Reserve56 new data/backward seeds, never held-out or previously used cases")
    diagnostic = ROOT / "paired_decoder_v19_teacher_gaps_010"
    gaps = read(diagnostic / "result.json")
    if not gaps["complete"] or gaps["prediction_performed"] or gaps["selection_performed"]:
        raise ValueError("Require the completed saved-development teacher comparison")
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json", diagnostic / "config.json", diagnostic / "result.json",
             preceding.training.OUTPUT / "config.json", preceding.OUTPUT / "config.json",
             preceding.OUTPUT / "registration.json", preceding.OUTPUT / "result.json",
             preceding.AUDIT / "audit.json", preceding.REPLAY / "audit.json",
             preceding.CONFIRMATION / "audit.json", preceding.CONFIRMATION / "gate.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_structured_teacher.py",
               REPO / "tests/test_single_model_structured_teacher_preflight.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": True,
              "retention_baseline_for_future_training": str(PARENT), "mutable_prefix": "decoder.",
              "teacher_policy": teacher_policy.POLICY, "teacher_identities": identities,
              "batch": batch(), "rate": RATE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,
              "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256, "audit_source_sha256": paired.AUDIT_SOURCE_SHA256,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in earlier}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "optimizer_update_performed": False,
               "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


def expected_role(task):
    specification = task["specification"]
    if specification["domain"] == "recsys":
        return "specialist"
    if specification["domain"] != "imputation":
        raise ValueError("Unsupported registered training domain")
    return "general" if "structured" in specification else "initialization"


def check_routing(cached, task, config):
    role = expected_role(task)
    expected = {"specialist_weight": float(role == "specialist"), "target": role,
                "policy": teacher_policy.POLICY, "teacher_identity": config["teacher_identities"][role]}
    if cached["routing"] != expected:
        raise ValueError("Cached teacher routing must bind the original role and checkpoint")
    return role


@torch.no_grad()
def replay_targets(output):
    config = verify(output)
    directory = output / "teacher_replay"
    if read(output / "status.json")["state"] != "awaiting_independent_audit" or directory.exists():
        raise ValueError("Replay all teacher outputs once after the actual optimizer preflight")
    if read(output / "result.json")["training_pid"] == os.getpid():
        raise ValueError("Teacher replay must use a fresh process")
    wait_for_gpu(output, True, False)
    directory.mkdir()
    status(output, "replaying_teacher_targets")
    model, optimizer, criterion, teachers, sampler = training.alignment.mixture.original_setup(config["base"])
    del optimizer, sampler
    models = {"initialization": model.eval().requires_grad_(False), "general": teachers[0], "specialist": teachers[1]}
    states = {role: teacher_policy.verify_model(current, config["teacher_identities"][role]) for role, current in models.items()}
    manifest, rows = read(output / "training_batch/manifest.json"), []
    for task, logged in zip(config["batch"]["tasks"], manifest["tasks"]):
        if sha256(logged["artifact"]) != logged["sha256"]:
            raise ValueError("Cached training task changed before teacher replay")
        cached = torch.load(logged["artifact"], map_location="cpu", weights_only=False)
        task_audit.check_task(cached, task, logged)
        role = check_routing(cached, task, config)
        episode = training.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        with torch.no_grad(), compute_context(episode.source.device):
            prediction = teacher_predictions(models[role], episode.source, criterion, **teacher_policy.STREAM_OPTIONS)
        target = prediction.detach().float().cpu()
        if not torch.equal(target, cached["teacher"]):
            raise ValueError("Independent original-checkpoint replay differs from cached teacher targets")
        path = directory / f'task_{task["index"]:02d}.npz'
        atomic_npz(path, teacher=target.numpy())
        rows.append({"task": task["name"], "index": task["index"], "role": role,
                     "identity": config["teacher_identities"][role], "prediction": str(path),
                     "prediction_sha256": sha256(path), "task_sha256": logged["sha256"], "bitwise_equal": True})
        status(output, "replaying_teacher_targets", completed=len(rows), expected=28)
    write_json(directory / "result.json", {"complete": True, "rows": rows, "model_states": states,
               "pid": os.getpid(), "training_pid": read(output / "result.json")["training_pid"],
               "config_sha256": sha256(output / "config.json"), "full_grid_bitwise_replay": True,
               "student_or_validation_evaluated": False})
    verify(output)
    status(output, "awaiting_independent_audit")


def audit_teacher_replay(output, config):
    path = output / "teacher_replay/result.json"
    report, manifest = read(path), read(output / "training_batch/manifest.json")
    if (not report["complete"] or len(report["rows"]) != 28 or len(manifest["tasks"]) != 28
            or not report["full_grid_bitwise_replay"] or report["student_or_validation_evaluated"]
            or report["training_pid"] != read(output / "result.json")["training_pid"]
            or report["pid"] == report["training_pid"] or report["config_sha256"] != sha256(output / "config.json")):
        raise ValueError("Require all28 independently replayed teacher targets without selection")
    hashes = {str(path): sha256(path), str(output / "training_batch/manifest.json"): sha256(output / "training_batch/manifest.json")}
    if set(report["model_states"]) != set(config["teacher_identities"]):
        raise ValueError("Require all original teacher-state checks")
    for role, identity in config["teacher_identities"].items():
        state = report["model_states"][role]
        if (state["checkpoint"] != identity["checkpoint"] or state["sha256"] != identity["sha256"]
                or not state["bitwise_equal"] or state["state_tensors"] < 1):
            raise ValueError("Teacher replay must verify original checkpoint tensors")
        hashes[identity["checkpoint"]] = identity["sha256"]
    for row, task, logged in zip(report["rows"], config["batch"]["tasks"], manifest["tasks"]):
        cached = torch.load(logged["artifact"], map_location="cpu", weights_only=False)
        task_audit.check_task(cached, task, logged)
        role = check_routing(cached, task, config)
        prediction_path = output / "teacher_replay" / f'task_{task["index"]:02d}.npz'
        if (row["task"] != task["name"] or row["index"] != task["index"] or row["role"] != role
                or row["identity"] != config["teacher_identities"][role] or not row["bitwise_equal"]
                or row["task_sha256"] != logged["sha256"] or row["prediction"] != str(prediction_path)):
            raise ValueError("Teacher replay must preserve every task and original teacher identity")
        with np.load(prediction_path, allow_pickle=False) as saved:
            if set(saved.files) != {"teacher"} or not np.array_equal(saved["teacher"], cached["teacher"].numpy()):
                raise ValueError("Stored independent teacher grids differ from actual cached training targets")
        hashes[str(prediction_path)] = row["prediction_sha256"]
        hashes[logged["artifact"]] = logged["sha256"]
    verify_hashes(hashes)
    return {"complete": True, "full_grids": 28, "bitwise_equal": True, "artifact_sha256": hashes}


ADAPTED_SOURCE = previous_preflight.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "paired_decoder_preflight_v1"': 'OUTPUT = ROOT / "structured_teacher_preflight_v1"',
    'DATA_SEED, GRADIENT_SEED = 341100901, 342100901': 'DATA_SEED, GRADIENT_SEED = 371100901, 372100901',
    'or config["audit_source_sha256"] != paired.AUDIT_SOURCE_SHA256':
        'or config["audit_source_sha256"] != paired.AUDIT_SOURCE_SHA256\n'
        '            or config["teacher_policy"] != teacher_policy.POLICY\n'
        '            or config["teacher_identities"] != teacher_policy.identities(config["base"], PARENT)',
    'with training.alignment.mixture.runtime(config["base"], output):':
        'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):',
    'write_json(output / "result.json", {"record": record,':
        'write_json(output / "result.json", {"training_pid": os.getpid(), "record": record,',
    'result, manifest = read(output / "result.json"), read(output / "training_batch/manifest.json")':
        'teacher_evidence = audit_teacher_replay(output, config)\n'
        '    result, manifest = read(output / "result.json"), read(output / "training_batch/manifest.json")',
    '    verify_hashes(hashes)\n    verify(output)':
        '    hashes.update(teacher_evidence["artifact_sha256"])\n    verify_hashes(hashes)\n    verify(output)',
    '"raw_endpoint": endpoint, "frozen": frozen,':
        '"teacher_replay": teacher_evidence, "raw_endpoint": endpoint, "frozen": frozen,',
    '# Paired-mean decoder optimizer preflight': '# Structured-general teacher optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


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
