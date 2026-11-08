"""One prospectively registered full-context four-draw feasibility attempt."""

import ast
from copy import deepcopy
import inspect
import types

import preflight_single_model_balanced_native_quarter as _source_probe


for _name, _value in vars(_source_probe).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
for _name, _value in vars(_source_probe).items():
    if not inspect.isfunction(_value):
        continue
    _raw = inspect.unwrap(_value)
    if _raw.__globals__ is not vars(_source_probe):
        continue
    if _raw.__closure__ is not None:
        raise ValueError("Do not carry a historical probe callback closure")
    _clone = types.FunctionType(_raw.__code__, globals(), _raw.__name__, _raw.__defaults__)
    _clone.__kwdefaults__ = _raw.__kwdefaults__
    if _raw is not _value:
        if _name not in {"predictions", "replay_admitted"}:
            raise ValueError("Rebuild only the known no-grad prediction and teacher replay decorators")
        _clone = torch.no_grad()(_clone)
    globals()[_name] = _clone

import single_model_four_draw_batching as four_draw
import single_model_four_draw_protection as paired
import single_model_four_draw_retention as retention
import single_model_four_draw_gradient_staging as staging
import single_model_balanced_native_long_budget as long_budget
import train_single_matrixformer_balanced_native_quarter as schedule_template
import resource
from single_model_registered_adaptation import replace_once


OUTPUT = ROOT / "four_draw_preflight_v1"
SCRATCH = Path("/mnt/matrixformer_scratch/four_draw_preflight_v1")
DATA_SEED, GRADIENT_SEED = 1501100901, 1502100901
NATIVE_SHAPE = [2000, 1000]
RESERVES = dict(_source_probe.RESERVES)
WRITE_BUDGETS = {"checkpoint": 1536 * 1024**2, "root": 32 * 1024**2, "scratch": 128 * 1024**3}
HOST_WORKING_BYTES, HOST_RESERVE_BYTES = 176 * 1024**3, 32 * 1024**3
STORAGE_ACCOUNTING = {
    "raw_gradient_bytes": 168 * 4 * TRAINABLE_PARAMETERS,
    "snapshot_vector_allowance_bytes": 12 * 4 * TRAINABLE_PARAMETERS,
    "ordinary_checkpoint_allowance_bytes": 8 * 4 * TRAINABLE_PARAMETERS,
    "prediction_task_allowance_bytes": 4 * 1024**3,
    "atomic_raw_allowance_bytes": (168 + 12) * 4 * TRAINABLE_PARAMETERS,
    "checkpoint_artifact_allowance_bytes": 1536 * 1024**2,
}
POLICY = (
    "One separately registered56-task full-context feasibility attempt after V56's fully audited "
    "negative closure. Use actual V29 initialization, never a V55/V56/probe endpoint. Pair the "
    "first two fresh balanced schedule batches without dropping or replacing any task; set all "
    "four native requested shapes to2000x1000 for a resource stress test. Keep four native draws, "
    "all four original components of each of12 structured families, and four rating contexts. "
    "Halve each task weight, preserving.5/.25/.25 domain masses, all losses, teachers, observed-only "
    "normalization and retention multipliers. Preserve fresh5e-6 AdamW, quarter displacement, "
    "0.1 margin,2x norm cap,34 finite slots, up to168 raw gradients and exact rollback. Capture "
    "the complete FP32 gradient bank on CPU; verify a late lossless CUDA copy and release the "
    "CPU bank before unchanged CUDA optimizer arithmetic. Retain.9 allocator cap,24 BLAS/4 Torch "
    "threads,18/6/50 GiB disk reserves and176+32 GiB working/reserve host-memory admission. "
    "Allow1536 MiB checkpoint/32 MiB root/128 GiB scratch, including an atomic raw-file allowance. "
    "Independently regenerate48 admission draws, replay56 teacher targets/104 imputation components, "
    "52 retention targets/100 reference components and audit up to112 original endpoint grids. "
    "Report all individual regressions, null slots, data-only rejection and numerical failures. "
    "One attempt only; no redraw, partial salvage, retry, favorable seed search, validation, "
    "confirmation, checkpoint nomination, full-trial authorization or real benchmark claim. "
    "One ordinary77,998,496-parameter model/238 trainable tensors/16 frozen buffers; no inference "
    "ensemble or changed inference. Preserve all22 development guards,32 fresh confirmation "
    "comparisons and the full frozen480-comparison real review for any later registered trial. "
    "Engineering feasibility is not accuracy, historical full autograd or goal completion. "
    "Scratch is machine-local, not external backup. All registered predecessors stay immutable."
)
POLICY_TEXT = POLICY
mixture = types.SimpleNamespace(MASSES=dict(_source_probe.mixture.MASSES), POLICY=four_draw.POLICY)
cohort_retention = retention
CLOSED = ROOT / "balanced_native_long_v56"
CLOSURE_HASHES = {
    str(ROOT / "balanced_native_long_v56_pipeline/result.json"): "9261d45c59d420cbc901af8f22a234747de7142e4faa422dbbaabf1de5088f96",
    str(ROOT / "balanced_native_long_v56_audit/audit.json"): "1fa9a8b41eb7520f2d6acb1fe92182017828204c55078ddd409f030fe72c8eed",
    str(ROOT / "balanced_native_long_v56_endpoint_replay/audit.json"): "4a4160166c0dc327fe2166d5e692809f4f20035076dfc09bffcb5d71eee9b8e7",
    str(ROOT / "balanced_native_long_v56_endpoint_replay/gate.json"): "c11fef4cc7e3262a0bda167ff3bf340446dbaae17b876fd97c8b26c70b6a27d7",
    str(ROOT / "balanced_native_long_v56_confirmation/gate.json"): "3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94",
}


def function_source(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError("Bind one complete declared inherited function")
    return ast.get_source_segment(source, nodes[0]) + "\n"


def batch():
    result = four_draw.pair_batches(long_budget.batches(schedule_template, DATA_SEED, GRADIENT_SEED))[0]
    for task in result["tasks"]:
        if task["cohort"] == "legacy":
            task["specification"]["shape"] = list(NATIVE_SHAPE)
    retention.validate_task_mixture(result["tasks"])
    return result


def blend_storage(batch):
    tasks = batch["tasks"]
    retention.validate_task_mixture(tasks)
    shapes = [task["specification"].get("structured", task["specification"])["shape"]
        for task in tasks if task["specification"]["domain"] == "imputation"]
    if len(shapes) != 52 or any(len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
            for shape in shapes):
        raise ValueError("Budget every complete requested imputation grid")
    arrays = 8 * sum(shape[0] * shape[1] for shape in shapes)
    return {"blend_tasks": 52, "component_grids": 104, "array_bytes": arrays,
        "serialization_allowance_bytes": 2 * 1024**2, "reserved_checkpoint_bytes": arrays + 2 * 1024**2}


blend_replay = types.SimpleNamespace(**vars(_source_probe.blend_replay))
blend_replay.storage_accounting = blend_storage
blend_replay.POLICY = _source_probe.blend_replay.POLICY + " Preserve all52 four-draw imputation targets/104 components."
BLEND_COMPONENT_ALLOWANCE = blend_storage(batch())["reserved_checkpoint_bytes"]


def require_predecessor():
    evidence = dict(CLOSURE_HASHES)
    verify_hashes(evidence)
    result = read(ROOT / "balanced_native_long_v56_pipeline/result.json")
    audit = read(ROOT / "balanced_native_long_v56_audit/audit.json")
    replay = read(ROOT / "balanced_native_long_v56_endpoint_replay/audit.json")
    gate = read(ROOT / "balanced_native_long_v56_confirmation/gate.json")
    if (result["complete"] is not True or result["gate"] != gate or gate["passed"] is not False
            or gate["selected_step"] != 0 or gate["confirmation_evaluated"] is not False
            or result["campaign"] is not None or result["review"] is not None
            or result["training_audit_sha256"] != CLOSURE_HASHES[str(ROOT / "balanced_native_long_v56_audit/audit.json")]
            or audit["complete_training_audit"] is not True or audit["attempted_batches"] != 256
            or audit["accepted_updates"] != 111 or audit["training_episodes"] != 7140
            or audit["independently_regenerated_structured_draws"] != 6144
            or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 0):
        raise ValueError("Preserve V56's complete negative outcome; audit success is not model success")
    for name in ("balanced_native_long_v56", "balanced_native_long_v56_pipeline",
            "balanced_native_long_v56_endpoint_replay", "balanced_native_long_v56_confirmation",
            "balanced_native_long_v56_audit_task_replay"):
        directory = ROOT / name
        state = read(directory / "status.json")
        expected = "training_complete" if name == "balanced_native_long_v56" else "complete"
        if (state["state"] != expected or type(state.get("pid")) is not int or state["pid"] <= 0
                or Path(f"/proc/{state['pid']}").exists()):
            raise ValueError("Every original scientific owner must be terminal before a new probe")
        evidence[str(directory / "status.json")] = sha256(directory / "status.json")
        config_path = directory / "config.json"
        if config_path.exists():
            evidence[str(config_path)] = sha256(config_path)
            for source, digest in read(config_path).get("source_sha256", {}).items():
                if source in evidence and evidence[source] != digest:
                    raise ValueError("Never reconcile changed historical source identities")
                evidence[source] = digest
    verify_hashes(evidence)
    return evidence


def require_host_space():
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
    available = int(fields["MemAvailable"].split()[0]) * 1024
    if available < HOST_WORKING_BYTES + HOST_RESERVE_BYTES:
        raise OSError("Admit all CPU capture/proposal/serialization copies plus the32 GiB host reserve")
    return {"available_bytes": available, "working_allowance_bytes": HOST_WORKING_BYTES,
        "reserve_bytes": HOST_RESERVE_BYTES, "source": "/proc/meminfo:MemAvailable"}


def require_space():
    scratch_bytes = sum(value for key, value in STORAGE_ACCOUNTING.items() if key != "checkpoint_artifact_allowance_bytes")
    if (scratch_bytes > WRITE_BUDGETS["scratch"]
            or STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] > WRITE_BUDGETS["checkpoint"]):
        raise ValueError("Complete raw and atomic-write evidence must fit beyond every disk reserve")
    require_volume_space()
    require_host_space()


RUN_SOURCE = function_source(_source_probe.ADAPTED_SOURCE, "run")
RUN_SOURCE = replace_once(RUN_SOURCE,
    'gradients = torch.empty((3 * len(config["batch"]["tasks"]), sum(row[2] for row in selected)), device="cuda", dtype=torch.float32)',
    'captured = torch.empty((3 * len(config["batch"]["tasks"]), sum(row[2] for row in selected)), device="cpu", dtype=torch.float32)')
RUN_SOURCE = replace_once(RUN_SOURCE, 'criterion, teachers, sampler, gradients)', 'criterion, teachers, sampler, captured)')
RUN_SOURCE = replace_once(RUN_SOURCE, '        status(output, "protecting_decoder_update")',
    '        capture_peak_allocated = torch.cuda.max_memory_allocated()\n'
    '        capture_peak_reserved = torch.cuda.max_memory_reserved()\n'
    '        torch.cuda.empty_cache()\n'
    '        gradients, transfer = staging.stage_for_update(captured, entries, config["batch"]["tasks"], before, "cuda")\n'
    '        del captured\n'
    '        update_start_allocated = torch.cuda.memory_allocated()\n'
    '        write_json(output / "gradient_transfer.json", transfer)\n'
    '        status(output, "protecting_decoder_update")')
RUN_SOURCE = replace_once(RUN_SOURCE, '"checkpoint_sha256": checkpoint_digest, "peak_reserved_gib":',
    '"gradient_transfer_sha256": sha256(output / "gradient_transfer.json"),\n'
    '                    "capture_peak_allocated_bytes": capture_peak_allocated, "capture_peak_reserved_bytes": capture_peak_reserved,\n'
    '                    "update_start_allocated_bytes": update_start_allocated, "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,\n'
    '                    "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory, "allocator_fraction": config["base"]["memory_fraction"],\n'
    '                    "checkpoint_sha256": checkpoint_digest, "peak_reserved_gib":')
exec(compile(RUN_SOURCE, __file__, "exec"), globals())

AUDIT_ADMITTED_SOURCE = function_source(_source_probe.ADAPTED_SOURCE, "audit")
for _old, _new in (("len(manifest[\"tasks\"]) != 28", "len(manifest[\"tasks\"]) != 56"),
        ("len(report[\"rows\"]) != 28", "len(report[\"rows\"]) != 56"),
        ('"training_tasks": 28', '"training_tasks": 56'),
        ("all twenty-eight fresh tasks", "all fifty-six fresh tasks")):
    AUDIT_ADMITTED_SOURCE = replace_once(AUDIT_ADMITTED_SOURCE, _old, _new)
exec(compile(AUDIT_ADMITTED_SOURCE, __file__, "exec"), globals())
audit_admitted = audit

TEACHER_REPLAY_SOURCE = replace_once(_source_probe.TEACHER_REPLAY_SOURCE, "expected=28", "expected=56")
exec(compile(TEACHER_REPLAY_SOURCE, __file__, "exec"), globals())
replay_admitted = replay_targets
TEACHER_REPLAY_SOURCE_SHA256 = hashlib.sha256(TEACHER_REPLAY_SOURCE.encode()).hexdigest()
TEACHER_REPLAY_AUDIT_SOURCE = _source_probe.TEACHER_REPLAY_AUDIT_SOURCE
for _old, _new in (('len(report["rows"]) != 28', 'len(report["rows"]) != 56'),
        ('len(manifest["tasks"]) != 28', 'len(manifest["tasks"]) != 56'),
        ('"full_grids": 28', '"full_grids": 56'), ("all28 independently", "all56 independently")):
    TEACHER_REPLAY_AUDIT_SOURCE = replace_once(TEACHER_REPLAY_AUDIT_SOURCE, _old, _new)
exec(compile(TEACHER_REPLAY_AUDIT_SOURCE, __file__, "exec"), globals())
TEACHER_REPLAY_AUDIT_SOURCE_SHA256 = hashlib.sha256(TEACHER_REPLAY_AUDIT_SOURCE.encode()).hexdigest()

RETENTION_RECORD_SOURCE = function_source(_source_probe.QUARTER_PROBE_SOURCE, "retention_records")
for _old, _new in (("len(tasks) != 28", "len(tasks) != 56"), ("len(registered) != 28", "len(registered) != 56"),
        ("all28 eligibility", "all56 eligibility"), ("len(records) != 26", "len(records) != 52"),
        ('{"v29": 2, "structured_mean": 24}', '{"v29": 4, "structured_mean": 48}')):
    RETENTION_RECORD_SOURCE = replace_once(RETENTION_RECORD_SOURCE, _old, _new)
exec(compile(RETENTION_RECORD_SOURCE, __file__, "exec"), globals())
RETENTION_REPLAY_SOURCE = _source_probe.RETENTION_REPLAY_SOURCE
RETENTION_REPLAY_SOURCE = replace_once(RETENTION_REPLAY_SOURCE, '{"v8": 24, "v29": 26}', '{"v8": 48, "v29": 52}')
RETENTION_REPLAY_SOURCE = replace_once(RETENTION_REPLAY_SOURCE, "all50 retention components and all26 final grids",
    "all100 retention components and all52 final grids")
exec(compile(RETENTION_REPLAY_SOURCE, __file__, "exec"), globals())
REPLAY_SOURCE = function_source(_source_probe.QUARTER_PROBE_SOURCE, "replay_targets")
exec(compile(REPLAY_SOURCE, __file__, "exec"), globals())


def endpoint_roles(record, candidate_exists):
    measured = record["point_after"] is not None
    if candidate_exists is not measured or (not measured and record["accepted"]):
        raise ValueError("Never fabricate or omit a measured candidate endpoint")
    return ("before", "candidate") if measured else ("before",)


RETENTION_AUDIT_SOURCE = function_source(_source_probe.QUARTER_PROBE_SOURCE, "audit")
for _old, _new in (('{"v8": 24, "v29": 26}', '{"v8": 48, "v29": 52}'),
        ('replay["component_grids"] != 50', 'replay["component_grids"] != 100'),
        ('"eligibility_decisions": 28', '"eligibility_decisions": 56'), ('"component_grids": 50', '"component_grids": 100')):
    RETENTION_AUDIT_SOURCE = replace_once(RETENTION_AUDIT_SOURCE, _old, _new)
RETENTION_AUDIT_SOURCE = replace_once(RETENTION_AUDIT_SOURCE,
    'measured = {role: read(output / f"{role}_predictions.json")["rows"] for role in ("before", "candidate")}',
    'measured = {role: read(output / f"{role}_predictions.json")["rows"] for role in\n'
    '        endpoint_roles(read(output / "result.json")["record"], (output / "candidate_predictions.json").exists())}')
RETENTION_AUDIT_SOURCE = replace_once(RETENTION_AUDIT_SOURCE, '        for role in ("before", "candidate"):',
    '        for role in measured:')
RETENTION_AUDIT_SOURCE = replace_once(RETENTION_AUDIT_SOURCE, '"endpoint_penalties": penalties,',
    '"endpoint_penalties": penalties, "endpoint_roles": list(measured),')
exec(compile(RETENTION_AUDIT_SOURCE, __file__, "exec"), globals())
retention_audit = audit


def audit(output):
    retention_audit(output)
    result = read(output / "result.json")
    if result.get("data_admission_only"):
        return
    transfer = read(output / "gradient_transfer.json")
    record = result["record"]
    expected = {"policy": staging.POLICY, "capacity_rows": 168, "defined_rows": record["protected_tasks"],
        "parameters": TRAINABLE_PARAMETERS, "capture_bytes": 168 * 4 * TRAINABLE_PARAMETERS,
        "transferred_bytes": record["protected_tasks"] * 4 * TRAINABLE_PARAMETERS, "source_device": "cpu",
        "destination_device": "cuda:0", "verified_elements": record["protected_tasks"] * TRAINABLE_PARAMETERS,
        "exact_fp32_transfer": True, "unused_capacity_rows": 168 - record["protected_tasks"],
        "peak_memory_measured": False, "scientific_feasibility_established": False}
    if transfer != expected or sha256(output / "gradient_transfer.json") != result["gradient_transfer_sha256"]:
        raise ValueError("Preserve complete original CPU-to-CUDA gradient transfer evidence")
    write_json(output / "gradient_transfer_audit.json", {"complete": True, "transfer": transfer,
        "transfer_sha256": result["gradient_transfer_sha256"], "raw_sha256": result["raw_sha256"],
        "independent_neural_gradient_replay": False, "goal_success_not_assessed": True})


ADAPTED_SOURCE = RUN_SOURCE + AUDIT_ADMITTED_SOURCE + RETENTION_RECORD_SOURCE + RETENTION_REPLAY_SOURCE + RETENTION_AUDIT_SOURCE
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
REGISTER_SOURCE = _source_probe.REGISTER_SOURCE
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, "len(fresh) != 56", "len(fresh) != 112")
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, "Reserve all56 new data/backward seeds", "Reserve all112 new data/backward seeds")
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "four_draw_batching": four_draw.SPECIFICATION, "gradient_staging_policy": staging.POLICY, '
    '"host_memory_admission": require_host_space(),')
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '    sources.update(Path(module.__file__).resolve()',
    '    sources.update((REPO / "tests").glob("test_single_model_four_draw*.py"))\n'
    '    sources.update({REPO / "scripts/audit_single_model_four_draw_probe.py", REPO / "scripts/run_single_model_four_draw_probe_pipeline.py"})\n'
    '    sources.update(Path(module.__file__).resolve()')
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())


def verify(output):
    config = original_verify(output)
    memory = config["host_memory_admission"]
    if (config["four_draw_batching"] != four_draw.SPECIFICATION or config["gradient_staging_policy"] != staging.POLICY
            or config["native_coverage"] != native.SPECIFICATION or config["native_probe_shape"] != NATIVE_SHAPE
            or config["structured_component_balance"] != components.SPECIFICATION
            or memory["working_allowance_bytes"] != HOST_WORKING_BYTES or memory["reserve_bytes"] != HOST_RESERVE_BYTES
            or memory["available_bytes"] < HOST_WORKING_BYTES + HOST_RESERVE_BYTES):
        raise ValueError("Never alter the complete four-draw schedule, staging or capacity admission")
    return config


if __name__ == "__main__":
    main()
