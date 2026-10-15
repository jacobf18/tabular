"""Probe native-anchor function retention after V36's fully audited negative result."""

import hashlib
import inspect
import types

import torch

import preflight_single_model_full_network as template
from preflight_single_model_full_network import (
    admission, preceding, rehearsal, replicated, schedule, storage, teacher_policy, StructuredSampler,
    sync_directory, composite, mixture, paired, full_network,
)
import run_single_model_full_network_cap90_v2_pipeline as latest
import single_model_anchor_retention as retention
import single_model_full_network_cap90_recipe as cap90_recipe
from single_model_registered_adaptation import replace_assignment, replace_once
from streamed_large_matrix_training import STREAM_OPTIONS


PREFIXES, PARENT_SHA256 = template.PREFIXES, template.PARENT_SHA256
TRAINABLE_TENSORS, TRAINABLE_PARAMETERS, FROZEN_TENSORS = 238, 77998496, 16
RESERVES, WRITE_BUDGETS = dict(template.RESERVES), dict(template.WRITE_BUDGETS)
STORAGE_ACCOUNTING = dict(template.STORAGE_ACCOUNTING)
POLICY_TEXT = ("One fresh28-task full-network optimizer feasibility test after complete negative V36 training, "
    "independent task/optimizer/development and endpoint audits, with no selection, confirmation or real campaign. "
    "Initialize only from the same V29 step25. Add the explicitly registered native-anchor penalty and frozen "
    "training-only V29 reference, never replace original general/specialist teacher targets. " + retention.POLICY
    + " Preserve238 trainable tensors/77,998,496 parameters,16 frozen buffers, all28 fresh tasks,84 raw gradients, "
    "30 point protections, .5/.25/.25 masses, half-composite/half-point proposal,5e-6 fresh AdamW,0.1 margin, "
    "2x norm bound, original finite acceptance and exact rollback. Keep observed-only normalization, original "
    "masks and every point metric. Retain the measured.9 allocator-only base adjustment. Independently replay "
    "both original teachers and every added retention grid; CPU-audit full eligibility, sources, masks, "
    "reference identity and before/candidate penalty values. All original held-out requirements remain intact. "
    "An initial zero retention residual is expected under identical deterministic predictions; this one-step "
    "test proves implementation feasibility, not regularization efficacy or benchmark accuracy. No new model "
    "selection, validation, confirmation, real evaluation, inference ensemble, data reuse, retry or diagnostic "
    "checkpoint warm start. Preserve18/6/50 GiB reserves and384 MiB/16 MiB/32 GiB stage allowances. Scratch "
    "is machine-local, not provider-persistent backup. All registered historical sources remain unchanged.")


def validate_latest(result, training_audit, replay_audit, replay_gate, gate):
    expected_training = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 61, "prediction_grids": 1368, "full_network_training": True,
        "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
        "independently_regenerated_structured_draws": 1536, "confirmation_evaluated": False}
    if (any(training_audit.get(key) != value for key, value in expected_training.items())
            or training_audit["best"]["step"] != 0
            or training_audit["admission_attempts"]["data_rejected_steps"] != []
            or training_audit["admission_attempts"]["optimizer_attempts"] != 64
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or training_audit["raw_endpoint"]["optimizer_attempt_step"] != 64
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 0 or replay_gate["endpoint_reproduced"] is not True
            or replay_gate["selected_validation_passed"] is not False or replay_gate["selected_step"] != 0
            or result["complete"] is not True or result["gate"] != gate or result["campaign"] is not None
            or result["review"] is not None or result["no_inference_ensemble"] is not True
            or gate["passed"] is not False or gate["selected_step"] != 0
            or gate["confirmation_evaluated"] is not False or gate["real_benchmark_success"] is not False):
        raise ValueError("Require the complete original V36 rejection and both independent audits")


def require_predecessor():
    directories = (latest.training.OUTPUT, latest.OUTPUT, latest.REPLAY, latest.CONFIRMATION)
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Finish and stop the original V36 stages before registering a new probe")
    report = latest.prerequisite(latest.OUTPUT)
    replay, replay_gate = [read(latest.REPLAY / name) for name in ("audit.json", "gate.json")]
    result, gate = read(latest.OUTPUT / "result.json"), read(latest.CONFIRMATION / "gate.json")
    validate_latest(result, report, replay, replay_gate, gate)
    if (result["training_audit_sha256"] != sha256(latest.AUDIT / "audit.json")
            or replay_gate["audit_sha256"] != sha256(latest.REPLAY / "audit.json")):
        raise ValueError("Bind both complete independent V36 audits")
    verify_hashes(replay["artifact_sha256"])
    forbidden = [latest.training.OUTPUT / name for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth")]
    forbidden.extend((latest.CAMPAIGN, latest.REVIEW, latest.CONFIRMATION / "audit.json", latest.CONFIRMATION / "manifest.json"))
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise ValueError("Do not fabricate V36 selection, confirmation or real evaluation")
    paths = [directory / "status.json" for directory in directories]
    paths.extend(latest.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend([latest.training.OUTPUT / "config.json", latest.AUDIT / "audit.json", latest.CONFIRMATION / "gate.json"])
    paths.extend(latest.REPLAY / name for name in ("audit.json", "gate.json"))
    return {str(path): sha256(path) for path in paths}


for name in ("require_space", "require_volume_space", "save_raw", "audit_raw", "expected_role", "check_routing",
             "audit_teacher_replay", "admit_batch", "audit_admission", "save_probe_checkpoint", "audit_probe_checkpoint",
             "frozen_inventory"):
    function = inspect.unwrap(getattr(template, name))
    if function.__closure__ is not None:
        raise ValueError("Never retain an older preflight helper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'base = read(training.alignment.fixed.PRODUCER / "config.json")':
        'base = cap90_recipe.expected_base(read(training.alignment.fixed.PRODUCER / "config.json"))',
    'config = {"policy": POLICY,': 'config = {"policy": POLICY, "anchor_retention": retention.SPECIFICATION, '
        '"retention_backward_sha256": retention.BACKWARD_SOURCE_SHA256,',
    'REPO / "tests/test_single_model_full_network_preflight.py",':
        'REPO / "tests/test_single_model_full_network_preflight.py", REPO / "tests/test_single_model_anchor_retention.py", '
        'REPO / "tests/test_single_model_anchor_retention_preflight.py",',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "full_network_preflight_v1"': 'OUTPUT = ROOT / "anchor_retention_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/full_network_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_retention_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 801100901, 802100901': 'DATA_SEED, GRADIENT_SEED = 871100901, 872100901',
    'config["base"] != read(training.alignment.fixed.PRODUCER / "config.json")':
        'config["base"] != cap90_recipe.expected_base(read(training.alignment.fixed.PRODUCER / "config.json"))',
    'or config["policy"] != POLICY or config["batch"] != batch() or config["rate"] != RATE':
        'or config["policy"] != POLICY or config["batch"] != batch() or config["rate"] != RATE\n'
        '            or config["anchor_retention"] != retention.SPECIFICATION\n'
        '            or config["retention_backward_sha256"] != retention.BACKWARD_SOURCE_SHA256',
    '    verify_hashes(config["input_sha256"])':
        '    cap90_recipe.validate_base(config["base"], read(training.alignment.fixed.PRODUCER / "config.json"),\n'
        '                               training.alignment.fixed.validate_recipe)\n'
        '    verify_hashes(config["input_sha256"])',
    '        model.train()': '        anchor = deepcopy(model).eval().requires_grad_(False)\n        model.train()',
    '        with full_network.capture_coordinates(model, view):':
        '        identity = {"checkpoint": str(PARENT), "sha256": PARENT_SHA256}\n'
        '        with retention.runtime(training.alignment.mixture.backend, output / "retention_targets", anchor, identity,\n'
        '                query_seed_offset=config["base"]["query_seed_offset"]), full_network.capture_coordinates(model, view):',
    '        sampler.assert_consumed()': '        sampler.assert_consumed()\n        del anchor',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted = audit
function = inspect.unwrap(template.replay_admitted)
replay_admitted = torch.no_grad()(types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__))
for name in ("replay_targets", "audit"):
    function = getattr(template, name)
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
replay_original_teachers, audit_original = replay_targets, audit


def retention_records(output):
    manifest = read(output / "retention_targets/manifest.json")
    tasks = read(output / "training_batch/manifest.json")["tasks"]
    registered = read(output / "config.json")["batch"]["tasks"]
    identity = {"checkpoint": str(PARENT), "sha256": PARENT_SHA256}
    if (manifest["complete"] is not True or manifest["policy"] != retention.POLICY
            or manifest["specification"] != retention.SPECIFICATION
            or manifest["backward_source_sha256"] != retention.BACKWARD_SOURCE_SHA256
            or manifest["inference_ensemble"] is not False or len(manifest["records"]) != len(tasks)
            or len(tasks) != 28 or len(registered) != 28
            or manifest["reference"] != {**identity, "state_tensors": 254, "bitwise_equal": True}):
        raise ValueError("Retain the complete additional reference and all28 eligibility decisions")
    seen, records = set(), []
    for specification, task, record in zip(registered, tasks, manifest["records"]):
        if task["sha256"] != sha256(Path(task["artifact"])):
            raise ValueError("Bind every original cached synthetic task")
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        task_audit.check_task(cached, specification, task)
        episode = training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        seed = episode.metadata["seed"]
        if (seed in seen or record["seed"] != seed or record["metadata"] != episode.metadata
                or record["eligible"] is not retention.eligible(episode.metadata)
                or record["shape"] != list(episode.target.shape) or record["queries"] != int(episode.mask.sum())):
            raise ValueError("Every actual task must keep its complete source, mask and native-prior identity")
        seen.add(seed)
        if record["eligible"]:
            path = output / f"retention_targets/reference_{seed}.pt"
            if record["artifact"] != str(path) or record["sha256"] != sha256(path):
                raise ValueError("Bind the canonical complete retention prediction")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload["identity"] != identity or payload["specification"] != retention.SPECIFICATION
                    or payload["metadata"] != episode.metadata
                    or not composite.byte_equal(payload["source"], episode.source)
                    or not composite.byte_equal(payload["mask"], episode.mask)):
                raise ValueError("Never swap the frozen reference, source or original query mask")
            retention.validate_target(episode, payload["prediction"])
            records.append((task, record, episode, payload))
        elif any(key in record for key in ("artifact", "sha256")):
            raise ValueError("Spectral and rating tasks must not acquire a retention reference")
    if not records:
        raise ValueError("The registered native-anchor batch must exercise retention")
    return manifest, records


def replay_targets(output):
    replay_original_teachers(output)
    config = verify(output)
    if read(output / "result.json").get("data_admission_only"):
        return
    manifest, records = retention_records(output)
    directory = output / "retention_replay"
    directory.mkdir()
    status(output, "replaying_retention_teacher")
    model, optimizer, criterion, teachers, sampler = training.alignment.mixture.original_setup(config["base"])
    del optimizer, teachers, sampler
    model.load_state_dict(torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)["model"], strict=True)
    model.eval().requires_grad_(False)
    identity = {"checkpoint": str(PARENT), "sha256": PARENT_SHA256}
    verified = retention.verify_model(model, identity)
    rows = []
    for task, record, episode, payload in records:
        with torch.no_grad(), compute_context(torch.device("cuda")):
            prediction = teacher_predictions(model, episode.source.cuda(), criterion, **STREAM_OPTIONS).float().cpu()
        if not composite.byte_equal(prediction, payload["prediction"]):
            raise ValueError("Independent V29 inference must reproduce every stored retention grid exactly")
        path = directory / f'reference_{record["seed"]}.npz'
        atomic_npz(path, prediction=prediction.numpy())
        rows.append({"seed": record["seed"], "artifact": str(path), "sha256": sha256(path)})
    if retention.verify_model(model, identity) != verified:
        raise ValueError("The independently replayed retention teacher must remain frozen")
    write_json(directory / "manifest.json", {"complete": True, "pid": os.getpid(), "records": rows,
        "reference": verified, "input_manifest_sha256": sha256(output / "retention_targets/manifest.json")})
    status(output, "awaiting_independent_audit")


def audit(output):
    audit_original(output)
    config = verify(output)
    result = read(output / "result.json")
    if result.get("data_admission_only"):
        write_json(output / "retention_audit.json", {"complete": True, "data_admission_only": True,
            "retention_evaluated": False, "original_audit_sha256": sha256(output / "audit.json")})
        return
    manifest, records = retention_records(output)
    replay = read(output / "retention_replay/manifest.json")
    if (replay["complete"] is not True or replay["reference"] != manifest["reference"]
            or replay["pid"] in (os.getpid(), result["training_pid"])
            or replay["input_manifest_sha256"] != sha256(output / "retention_targets/manifest.json")
            or len(replay["records"]) != len(records)):
        raise ValueError("Require independent complete replay of all retention targets")
    measured = {role: read(output / f"{role}_predictions.json")["rows"] for role in ("before", "candidate")}
    penalties, hashes = [], {}
    for (task, record, episode, payload), replayed in zip(records, replay["records"]):
        path = output / f'retention_replay/reference_{record["seed"]}.npz'
        if replayed != {"seed": record["seed"], "artifact": str(path), "sha256": sha256(path)}:
            raise ValueError("Bind each independently replayed retention grid")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["prediction"] or not np.array_equal(archive["prediction"], payload["prediction"].numpy()):
                raise ValueError("Independent retention replay must reproduce every prediction")
        hashes.update({str(path): sha256(path), record["artifact"]: record["sha256"], task["artifact"]: task["sha256"]})
        row = {"task": task["name"], "seed": record["seed"], "queries": int(episode.mask.sum())}
        for role in ("before", "candidate"):
            prediction_row = next(item for item in measured[role] if item["task"] == task["name"])
            prediction_path = Path(prediction_row["prediction"])
            if sha256(prediction_path) != prediction_row["sha256"]:
                raise ValueError("Retain original before/candidate endpoint predictions")
            with np.load(prediction_path, allow_pickle=False) as archive:
                prediction = torch.from_numpy(archive["prediction"])
            difference = prediction[episode.mask].double() - payload["prediction"][episode.mask].double()
            if not torch.isfinite(difference).all():
                raise ValueError("Report finite retention errors without dropping any original query")
            row[role] = retention.SPECIFICATION["weight"] * float(difference.square().mean())
            hashes[str(prediction_path)] = prediction_row["sha256"]
        penalties.append(row)
    paths = [output / name for name in ("audit.json", "config.json", "retention_targets/manifest.json", "retention_replay/manifest.json")]
    hashes.update({str(path): sha256(path) for path in paths})
    verify_hashes(hashes)
    write_json(output / "retention_audit.json", {"complete": True, "data_admission_only": False,
        "eligibility_decisions": 28, "reference_grids": len(records), "endpoint_penalties": penalties,
        "reference": manifest["reference"], "artifact_sha256": hashes, "cuda_initialized": False,
        "original_audit_sha256": sha256(output / "audit.json"), "full_autograd_replayed": False,
        "no_inference_ensemble": True, "goal_success_not_assessed": True})
    status(output, "complete", retention_audited=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    callback = {"register": register, "run": run, "replay": replay_targets, "audit": audit}[arguments.stage]
    try:
        callback(OUTPUT)
    except BaseException as error:
        if OUTPUT.is_dir():
            status(OUTPUT, "failed", error=repr(error), failed_stage=arguments.stage)
        raise
