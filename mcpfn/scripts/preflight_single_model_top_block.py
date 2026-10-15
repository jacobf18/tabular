"""Test a single last-block/decoder update after V23's completed qualification."""

import hashlib
import inspect

import torch

import preflight_single_model_structured_teacher as template
import run_single_model_compact_continuation_pipeline as preceding
import single_model_composite_gradient_storage as composite
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import single_model_shared_tensor_storage as storage
import single_model_structured_teacher as teacher_policy
import single_model_top_block_training as top_block
import train_single_matrixformer_targeted_decoder as schedule
from single_model_registered_adaptation import replace_once, replace_assignment


PREFIXES = ("transformer_blocks.7.", "decoder.")
POLICY_TEXT = ("One fresh28-task last-transformer-block plus decoder optimizer feasibility test from original V8 "
    "after V23's completed unsuccessful independent qualification. Preserve the ordinary network, all other "
    "parameters and every buffer bitwise; introduce no inference branch or ensemble. Keep the existing first-batch "
    "shapes, .25 legacy/.25 structured/.5 ratings mass, MAR/skip emphasis, observed-only normalization, original "
    "Huber/NLL/CDF/query losses, V8/general/specialist teacher assignments,2e-5 Adam proposal, paired0.1 protection, "
    "2x norm bound and exact rollback. Save all84 raw protected gradients with the identical28-row composite "
    "prefix alias; independently replay all28 teacher grids and CPU-audit the optimizer endpoint and all before/"
    "candidate prediction grids. This endpoint cannot initialize a future model. No development, confirmation, "
    "selection, promotion, real benchmark or claim of general training memory feasibility beyond this batch. "
    "Keep canonical evidence and the checkpoint on durable storage; the bulky raw diagnostic endpoint alone "
    "uses explicitly declared machine-local scratch, not a backup or provider-persistent volume. Preserve18 GiB "
    "checkpoint,6 GiB root and50 GiB scratch reserves. No existing registered source, seed or gate changes.")


def require_space():
    for path, reserve, budget in ((ROOT, 18, 1), (Path("/"), 6, 0), (SCRATCH.parent, 50, 12)):
        if shutil.disk_usage(path).free - budget * 1024**3 < reserve * 1024**3:
            raise OSError(f"Preserve {reserve} GiB reserve plus {budget} GiB budget on {path}")


def require_predecessor():
    result, current = read(preceding.OUTPUT / "result.json"), read(preceding.OUTPUT / "status.json")
    if (result["complete"] is not True or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True or current["state"] != "complete"
            or Path(f'/proc/{current["pid"]}').exists() or result["campaign"] is not None or result["review"] is not None):
        raise ValueError("Require terminal V23 with failed independent confirmation and no real campaign")
    preceding.prerequisite(preceding.OUTPUT)
    for directory, key, expected in ((preceding.REPLAY, "endpoint_grids", 56),
                                     (preceding.CONFIRMATION, "prediction_grids", 1136)):
        report = read(directory / "audit.json")
        if report["complete"] is not True or report[key] != expected:
            raise ValueError("Preserve every independent predecessor grid audit")
        verify_hashes(report["artifact_sha256"])
    if (preceding.training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Never compete with a published predecessor")


def register_trial(output):
    if output != OUTPUT or output.exists() or output.is_symlink() or SCRATCH.exists() or SCRATCH.is_symlink():
        raise FileExistsError("Use only new declared top-block preflight and scratch directories")
    require_space()
    require_predecessor()
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    identities = teacher_policy.identities(base, PARENT)
    if PARENT != training.alignment.fixed.PARENT or sha256(PARENT) != identities["initialization"]["sha256"]:
        raise ValueError("Initialize only from original V8, never an unqualified or preflight endpoint")
    earlier = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in earlier))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 56 or fresh.intersection(used):
        raise ValueError("Reserve all56 new data/backward seeds without reusing any held-out case")
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json",
             preceding.training.OUTPUT / "config.json", preceding.OUTPUT / "config.json",
             preceding.OUTPUT / "registration.json", preceding.OUTPUT / "result.json",
             preceding.AUDIT / "audit.json", preceding.REPLAY / "audit.json",
             preceding.CONFIRMATION / "audit.json", preceding.CONFIRMATION / "gate.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_top_block_preflight.py",
               REPO / "tests/test_single_model_top_block_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": True,
              "retention_baseline_for_future_training": str(PARENT), "mutable_prefixes": list(PREFIXES),
              "teacher_policy": teacher_policy.POLICY, "teacher_identities": identities,
              "batch": batch(), "rate": RATE, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,
              "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256, "audit_source_sha256": paired.AUDIT_SOURCE_SHA256,
              "scratch": str(SCRATCH), "scratch_is_provider_persistent": False,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in earlier}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    SCRATCH.mkdir()
    output.mkdir()
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
               "registered_unix": time.time(), "optimizer_update_performed": False,
               "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


def frozen_inventory(initial, current, mutable_names):
    expected = {row[0] for row in read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json")
                if row[0].startswith(PREFIXES)}
    if set(initial) != set(current) or set(mutable_names) != expected or len(expected) != 34:
        raise ValueError("Preserve exact full-model inventory and34 last-block/decoder tensors")
    changed, frozen = [], 0
    for name, before in initial.items():
        after = current[name]
        if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
            raise ValueError("Every model tensor must retain its finite original shape and dtype")
        equal = composite.byte_equal(before.detach().cpu(), after.detach().cpu())
        if name not in expected:
            frozen += 1
            if not equal:
                raise ValueError(f"Frozen parameter or buffer changed: {name}")
        elif not equal:
            changed.append(name)
    return {"frozen_tensors": frozen, "changed_trainable_tensors": changed,
            "unselected_parameters_and_all_buffers_bitwise_unchanged": True}


def save_raw(payload, path):
    if path.parent != OUTPUT or path.exists() or path.is_symlink():
        raise FileExistsError("Publish only a fresh declared raw endpoint")
    gradients, snapshot = composite.share(payload["gradients"], payload["snapshot"], len(payload["batch"]["tasks"]))
    target = SCRATCH / path.name
    digest = storage.save({**payload, "gradients": gradients, "snapshot": snapshot}, target)
    path.symlink_to(target)
    write_json(path.with_suffix(".scratch-write.json"), {"path": str(path), "target": str(target), "sha256": digest,
               "receipt_sha256": sha256(target.with_suffix(".write.json")), "scratch_is_provider_persistent": False})
    return digest


def audit_raw(payload, path):
    receipt = path.with_suffix(".scratch-write.json")
    report, target = read(receipt), SCRATCH / path.name
    if (not path.is_symlink() or path.resolve() != target or report != {"path": str(path), "target": str(target),
            "sha256": sha256(path), "receipt_sha256": sha256(target.with_suffix(".write.json")),
            "scratch_is_provider_persistent": False}):
        raise ValueError("Preserve exact raw endpoint and declared local-scratch placement")
    composite.check_alias(payload)
    return {str(receipt): sha256(receipt), **storage.audit(payload, target)}


for name in ("expected_role", "check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "structured_teacher_preflight_v1"':
        'OUTPUT = ROOT / "top_block_preflight_v1"\nSCRATCH = Path("/mnt/matrixformer_scratch/top_block_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 371100901, 372100901': 'DATA_SEED, GRADIENT_SEED = 481100901, 482100901',
    'replicated.replicate_batch(preceding.training.previous.batches()[0], DATA_SEED, GRADIENT_SEED)':
        'replicated.replicate_batch(schedule.batches()[0], DATA_SEED, GRADIENT_SEED)',
    'or config["mutable_prefix"] != "decoder."':
        'or config["mutable_prefixes"] != list(PREFIXES) or config["scratch"] != str(SCRATCH)\n'
        '            or config["scratch_is_provider_persistent"] is not False',
    'view = decoder.DecoderParameterView(model)': 'view = top_block.TopBlockParameterView(model)',
    'with decoder.decoder_capture_coordinates(model, view):': 'with top_block.capture_coordinates(model, view):',
    'frozen = decoder.frozen_state_equal(parent["model"], model.state_dict(), {row[0] for row in selected})':
        'frozen = frozen_inventory(parent["model"], model.state_dict(), {row[0] for row in selected})',
    'frozen = decoder.frozen_state_equal(parent["model"], checkpoint["model"], {row[0] for row in selected})':
        'frozen = frozen_inventory(parent["model"], checkpoint["model"], {row[0] for row in selected})',
    'bool(frozen["changed_decoder_tensors"])': 'bool(frozen["changed_trainable_tensors"])',
    'digest = training.save_tensor({"snapshot": snapshot,': 'digest = save_raw({"snapshot": snapshot,',
    'training.save_tensor({"snapshot": error.snapshot,': 'save_raw({"snapshot": error.snapshot,',
    'endpoint = paired.check_endpoint(': 'hashes.update(audit_raw(payload, raw_path))\n    endpoint = paired.check_endpoint(',
    '"checkpoint_sha256": checkpoint_digest, "peak_gib":':
        '"checkpoint_sha256": checkpoint_digest, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,\n'
        '                    "peak_gib":',
    '# Structured-general teacher optimizer preflight': '# Last-block plus decoder optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
if ADAPTED_SOURCE.count('row[0].startswith("decoder.")') != 2:
    raise ValueError("Adapt both training and independent-audit coordinate inventories")
ADAPTED_SOURCE = ADAPTED_SOURCE.replace('row[0].startswith("decoder.")', 'row[0].startswith(PREFIXES)')
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
