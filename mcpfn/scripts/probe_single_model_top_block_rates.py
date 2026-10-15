"""Test smaller actual Adam proposals on the completed top-block training batch."""

import inspect

import probe_single_model_decoder_rates as template
import single_model_composite_gradient_storage as composite
import single_model_paired_protection as paired
from single_model_registered_adaptation import install


POLICY_TEXT = ("A bounded reused-training-batch diagnostic, not training or held-out model selection. After the "
    "completed unsuccessful V23 qualification and independently audited top-block preflight, reuse precisely "
    "its28 tasks, all84 raw gradients and original V8 input. Recompute actual fresh fused-CUDA Adam proposals "
    "at2e-5,1e-5,5e-6,2e-6; never rescale rounded parameters or inherit another rate's moments. Preserve the "
    "original28-composite proposal,42 paired protection means,0.1 margin,2x norm bound, teacher targets, losses, "
    "normalization and exact rollback. Reproduce the original rejected2e-5 candidate and optimizer endpoint "
    "before interpreting smaller rates. Repeat every before/candidate full prediction grid twice with original "
    "seeds. A rate is locally feasible only if both repeats retain all defined paired means and one improves, "
    "while every individual-draw change remains visible. Independently CPU-audit all endpoints and280 expected "
    "prediction grids. Save compact diagnostic endpoints on declared machine-local scratch with durable canonical "
    "symlinks and hashes, referencing rather than duplicating original gradients. Scratch is not provider-persistent "
    "storage or a backup. Preserve18 GiB checkpoint,6 GiB root and50 GiB scratch reserves. No new data or autograd, "
    "validation, confirmation, checkpoint nomination, alias, real benchmark, inference ensemble or new training "
    "recipe. No guard relaxation, automatic retry or change to any completed registered source.")


def require_capacity():
    for path, reserve, budget in ((source.ROOT, 18, 1), (Path("/"), 6, 0), (SCRATCH.parent, 50, 4)):
        if shutil.disk_usage(path).free - budget * 1024**3 < reserve * 1024**3:
            raise OSError(f"Preserve {reserve} GiB reserve and {budget} GiB diagnostic budget on {path}")


def register_trial(output):
    if output != OUTPUT or output.exists() or output.is_symlink() or SCRATCH.exists() or SCRATCH.is_symlink():
        raise FileExistsError("Use only the fresh declared rate-probe and scratch directories")
    require_capacity()
    source.verify(source.OUTPUT)
    source.require_predecessor()
    audit, gate = read(source.OUTPUT / "audit.json"), read(source.OUTPUT / "gate.json")
    current = read(source.OUTPUT / "status.json")
    if (current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or audit["complete"] is not True or gate["passed"] is not False or gate["preflight_only"] is not True
            or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["teacher_replay"]["full_grids"] != 28):
        raise ValueError("Require the complete rejected top-block preflight and all independent audits")
    verify_hashes(audit["artifact_sha256"])
    paths = [source.OUTPUT / name for name in ("audit.json", "gate.json", "config.json", "registration.json")]
    paths.extend([source.PARENT, WAIT_FOR / "registration.json", WAIT_FOR / "result.json"])
    inputs = {**audit["artifact_sha256"], **{str(path): sha256(path) for path in paths}}
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_top_block_rates.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    SCRATCH.mkdir()
    output.mkdir()
    write_json(output / "config.json", {"policy": POLICY, "source": str(source.OUTPUT), "rates": list(RATES),
               "repeats": list(REPEATS), "wait_for": str(WAIT_FOR), "input_sha256": inputs,
               "source_sha256": {str(path): sha256(path) for path in sources}, "fresh_data_or_gradients": False,
               "scratch": str(SCRATCH), "scratch_is_provider_persistent": False,
               "adapted_source_sha256": ADAPTED_SOURCE_SHA256})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "model_evaluated": False, "validation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


def save_endpoint(payload, path):
    if (path.parent.parent != OUTPUT or path.parent.name not in {f"rate_{index}" for index in range(len(RATES))}
            or path.name not in {"endpoint.pt", "failed_endpoint.pt"} or path.exists() or path.is_symlink()):
        raise FileExistsError("Write only fresh declared diagnostic endpoints")
    target = SCRATCH / path.parent.name / path.name
    target.parent.mkdir(exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Never overwrite scratch endpoint evidence")
    if shutil.disk_usage(SCRATCH).free - source.training.tensor_bytes(payload) - 4 * 1024**2 < 50 * 1024**3:
        raise OSError("Preserve50 GiB scratch including the entire pending endpoint")
    digest = source.training.save_tensor(payload, target)
    path.symlink_to(target)
    write_json(path.with_suffix(".scratch-write.json"), {"path": str(path), "target": str(target), "sha256": digest,
               "file_bytes": target.stat().st_size, "scratch_is_provider_persistent": False})
    return digest


def audit_endpoint_storage(path):
    path = Path(path)
    receipt, target = path.with_suffix(".scratch-write.json"), SCRATCH / path.parent.name / path.name
    expected = {"path": str(path), "target": str(target), "sha256": sha256(path),
                "file_bytes": target.stat().st_size, "scratch_is_provider_persistent": False}
    if not path.is_symlink() or path.resolve() != target or read(receipt) != expected:
        raise ValueError("Retain exact compact endpoint bytes and declared scratch placement")
    return {str(receipt): sha256(receipt)}


def check_reference(original, record, snapshot, candidate_path):
    compare_nested(record, original["record"])
    for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient", "task_gradients"):
        if not composite.byte_equal(snapshot[name].cpu(), original["snapshot"][name].cpu()):
            raise ValueError("Original2e-5 proposal and rejected endpoint must reproduce bitwise")
    source.checks.optimizer_equal(snapshot["optimizer_before"], original["snapshot"]["optimizer_before"])
    source.checks.optimizer_equal(snapshot["optimizer_after"], original["snapshot"]["optimizer_after"])
    actual = torch.load(candidate_path, map_location="cpu", weights_only=True, mmap=True)
    expected = torch.load(source.OUTPUT / "candidate_parameters.pt", map_location="cpu", weights_only=True, mmap=True)
    if not composite.byte_equal(actual, expected):
        raise ValueError("Original rejected candidate parameters must reproduce bitwise")


install(template, globals(), {
    'import preflight_single_model_targeted_decoder as source': 'import preflight_single_model_top_block as source',
    'import single_model_decoder_training as decoder': 'import single_model_top_block_training as top_block',
    'OUTPUT = source.ROOT / "decoder_rate_probe_v1"':
        'OUTPUT = source.ROOT / "top_block_rate_probe_v1"\nSCRATCH = Path("/mnt/matrixformer_scratch/top_block_rate_probe_v1")',
    'WAIT_FOR = source.ROOT / "targeted_decoder_v15_pipeline_recovery1"': 'WAIT_FOR = source.preceding.OUTPUT',
    'RATES = (2e-6, 2e-5, 1e-4)': 'RATES = (2e-5, 1e-5, 5e-6, 2e-6)',
    'not torch.equal(snapshot["task_gradients"], gradients[:count])':
        'not composite.byte_equal(snapshot["task_gradients"], gradients[:count])',
    inspect.getsource(template.register): 'def register(output):\n    return register_trial(output)\n',
    'or config["wait_for"] != str(WAIT_FOR) or config["source"] != str(source.OUTPUT)':
        'or config["wait_for"] != str(WAIT_FOR) or config["source"] != str(source.OUTPUT)\n'
        '            or config["scratch"] != str(SCRATCH) or config["scratch_is_provider_persistent"] is not False\n'
        '            or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256',
    'source.training.guarded.metric_decision(before[index], candidate[index])':
        'paired.metric_decision(before[index], candidate[index])',
    '    if shutil.disk_usage(source.ROOT).free < 20 * 1024**3:\n'
    '        raise OSError("Keep18 GiB free plus the compact endpoint/prediction budget")': '    require_capacity()',
    'view = decoder.DecoderParameterView(model)': 'view = top_block.TopBlockParameterView(model)',
    'source.training.guarded.protected_update(': 'paired.protected_update(',
    'if index == 0 and (not record["accepted"] or not torch.equal(snapshot["after"], original["snapshot"]["after"])):\n'
    '                raise ValueError("The original2e-6 GPU endpoint must reproduce bitwise before interpreting larger rates")':
        'if index == 0:\n                check_reference(original, record, snapshot, directory / "candidate_parameters.pt")',
    'frozen = decoder.frozen_state_equal(parent, model.state_dict(), {row[0] for row in selected})':
        'frozen = source.frozen_inventory(parent, model.state_dict(), {row[0] for row in selected})',
    'source.training.save_tensor({"snapshot": compact_snapshot(error.snapshot, raw_gradients, 14),':
        'save_endpoint({"snapshot": compact_snapshot(error.snapshot, raw_gradients, 28),',
    'digest = source.training.save_tensor({"snapshot": compact_snapshot(snapshot, raw_gradients, 14),':
        'digest = save_endpoint({"snapshot": compact_snapshot(snapshot, raw_gradients, 28),',
    'len(report["rows"]) != 14': 'len(report["rows"]) != 28',
    'grids, endpoints = 28, []': 'grids, endpoints = 56, []',
    'snapshot = restore_snapshot(payload["snapshot"], original["gradients"], 14)':
        'snapshot = restore_snapshot(payload["snapshot"], original["gradients"], 28)\n'
        '        hashes.update(audit_endpoint_storage(row["endpoint"]))',
    'checks.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before["0"], 42)':
        'paired.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before["0"], 84)',
    'endpoint = checks.check_endpoint(': 'endpoint = paired.check_endpoint(',
    'if row["index"] == 0 and not torch.equal(snapshot["after"], original["snapshot"]["after"]):\n'
    '            raise ValueError("Baseline replay must exactly reproduce the original accepted endpoint")':
        'if row["index"] == 0:\n            check_reference(original, record, snapshot, directory / "candidate_parameters.pt")',
    'grids += 28': 'grids += 56',
    'if row["frozen"]["frozen_tensors"] != 248 or not row["frozen"]["backbone_and_buffers_bitwise_unchanged"]:':
        'if row["frozen"]["frozen_tensors"] != 220 or not row["frozen"]["unselected_parameters_and_all_buffers_bitwise_unchanged"]:',
    '"training_tasks": 14': '"training_tasks": 28',
    '# Decoder learning-rate feasibility probe': '# Last-block plus decoder learning-rate feasibility probe',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
