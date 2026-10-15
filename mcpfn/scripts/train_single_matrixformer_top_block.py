"""Train an ordinary V8-start last-block/decoder model on fresh mixed-prior batches."""

import ast
import hashlib

import train_single_matrixformer_structured_teacher as baseline
import probe_single_model_top_block_rates as rate_probe
import recover_single_model_top_block_rate_audit as rate_audit
import single_model_compact_training_cache as compact_cache
import single_model_composite_gradient_storage as composite
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import single_model_shared_tensor_storage as storage
import single_model_structured_teacher as teacher_policy
import single_model_top_block_training as top_block
from single_model_registered_adaptation import replace_once, replace_assignment


RATE = 1e-5
PREFIXES = ("transformer_blocks.7.", "decoder.")
TRIAL_POLICY = ("Initialize one ordinary single-checkpoint network from original V8, never a preflight, rate-probe "
    "or failed-confirmation endpoint. Train only the existing final transformer block and decoder:34 tensors/"
    "19,496,843 parameters; preserve every other parameter and all buffers,220 state tensors, bitwise. Attempt64 "
    "fresh28-task batches, two independent draws per cohort, with the existing .25 legacy/.25 structured/.5 ratings "
    "mass, MAR/skip emphasis and unchanged shape/component/regime schedule. Keep original V8 legacy, original general "
    "structured and original specialist rating teachers; preserve observed-only normalization and Huber/NLL/CDF/query "
    "objectives. Use constant1e-5 composite Adam proposals, the largest audited rate passing both reused-training "
    "repeats, with unchanged paired0.1 protection,2x norm bound and exact parameter/moment/preceding-rate rollback. "
    "Keep all individual-draw regressions visible. Store exact compact task/teacher digests and rating vectors, and "
    "all84 final gradients with the identical28-row composite prefix alias. The bulky raw diagnostic endpoint alone "
    "uses declared machine-local scratch, not provider-persistent storage or a backup. Ordinary selected/latest "
    "checkpoints remain on durable storage. No inference branch or ensemble, new architecture, held-out rate tuning, "
    "automatic retry, or change to original development/confirmation/real gates.")
TRIAL_QUALIFICATION = ("Independently regenerate/digest-check all1792 complete new task/teacher payloads, audit every "
    "attempt and all1216 development/reference grids, all22 original V8 retention guards, the220 frozen state tensors "
    "and full terminal optimizer/gradient evidence. Earlier direction certificates are not full historical autograd "
    "replays. Independently reconstruct/replay the final neural endpoint and sole selected152-case development "
    "boundary without reselection. Only a trained strict minimax improvement may qualify; step0 is ineligible and "
    "ties retain the earlier boundary. Confirm once on284 fresh reserved cases under unchanged V8/general retention "
    "gates and audit all1136 grids. Only then publish a best alias and run the complete frozen tabular, four-dataset "
    "recommender and LLM campaign/review. Do not conceal any required real comparison, change thresholds or infer "
    "benchmark success from synthetic gains. Later qualification stages require separate capacity checks.")


def constant_rate(step):
    if type(step) is not int or not 1 <= step <= STEPS:
        raise ValueError("Require an actual registered attempt")
    return RATE


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in baseline.previous.batches()]


def require_capacity():
    for path, reserve, budget in ((ROOT, 18, 1.5), (Path("/"), 6, 0), (SCRATCH.parent, 50, 12)):
        if shutil.disk_usage(path).free - int(budget * 1024**3) < reserve * 1024**3:
            raise OSError(f"Preserve {reserve} GiB reserve plus {budget} GiB training budget on {path}")


def require_evidence():
    rate_probe.verify(rate_probe.OUTPUT)
    preflight.require_predecessor()
    recovered = read(rate_audit.OUTPUT / "recovery.json")
    audit = read(rate_audit.OUTPUT / "audit.json")
    current = read(rate_audit.OUTPUT / "status.json")
    if (not recovered["complete"] or recovered["neural_evaluation_repeated"] or not audit["complete"]
            or audit["prediction_grids"] != 280 or audit["training_tasks"] != 28
            or current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or recovered["audit_sha256"] != sha256(rate_audit.OUTPUT / "audit.json")):
        raise ValueError("Require the completed independent full rate-panel audit")
    rows = read(rate_probe.OUTPUT / "result.json")["rates"]
    feasible = [row["rate"] for row in rows if row["finite_feasible"]]
    if (not feasible or RATE != max(feasible) or rows[0]["rate"] != 2e-5 or rows[0]["finite_feasible"]
            or [row["rate"] for row in rows] != list(rate_probe.RATES)):
        raise ValueError("Use the largest originally declared rate passing both repeats, never a probe checkpoint")
    verify_hashes(audit["artifact_sha256"])
    for key in ("input_sha256", "source_sha256"):
        verify_hashes(read(rate_audit.OUTPUT / "config.json")[key])
    paths = [rate_audit.OUTPUT / name for name in ("config.json", "recovery.json", "audit.json", "status.json")]
    paths.extend(rate_probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend(preflight.OUTPUT / name for name in ("config.json", "audit.json", "gate.json"))
    return {str(path): sha256(path) for path in paths}


def register_trial(output):
    if output != OUTPUT or output.exists() or SCRATCH.exists():
        raise FileExistsError("Never overwrite or restart the declared top-block trial")
    require_capacity()
    evidence = require_evidence()
    base = read(original.alignment.fixed.PRODUCER / "config.json")
    if PARENT != original.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:
        raise ValueError("Initialize only from original V8")
    earlier = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(original.alignment.fixed.seeds_in(read(path)) for path in earlier))
    fresh = original.alignment.fixed.seeds_in([batches(), confirmation_cases()])
    if len(fresh) != 3868 or fresh.intersection(used):
        raise ValueError("Reserve all3584 fresh data/backward seeds and284 independent confirmation seeds")
    paths = [PARENT, original.alignment.fixed.PRODUCER / "config.json", original.alignment.fixed.PRODUCER / "synthetic_bank.pt",
             original.alignment.fixed.PRODUCER / "reference.json", INITIAL_RUN / "validation_0.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), REPO / "scripts/audit_single_model_top_block_training.py",
               REPO / "tests/test_single_model_top_block_trial.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "qualification": QUALIFICATION, "base": base, "model": base["model"],
              "parent_checkpoint": str(PARENT), "parent_selection_qualified": True, "retention_baseline": str(PARENT),
              "mutable_prefixes": list(PREFIXES), "steps": STEPS, "batches": batches(), "boundaries": list(BOUNDARIES),
              "confirmation_cases": confirmation_cases(), "confirmation_seeds": list(CONFIRMATION_SEEDS),
              "learning_rate_schedule": {"constant": RATE}, "margin": .1, "guarded_policy": paired.POLICY,
              "teacher_policy": teacher_policy.POLICY, "teacher_identities": teacher_policy.identities(base, PARENT),
              "adapted_source_sha256": ADAPTED_SOURCE_SHA256, "capture_source_sha256": compact_cache.CAPTURE_SOURCE_SHA256,
              "evaluate_source_sha256": compact_cache.EVALUATE_SOURCE_SHA256, "protection_source_sha256": paired.PROTECTED_SOURCE_SHA256,
              "task_storage_policy": compact_cache.POLICY, "scratch": str(SCRATCH), "scratch_is_provider_persistent": False,
              "best_step_zero_semantics": "No eligible trained boundary; never promote unchanged V8 as a newly trained model",
              "input_sha256": {**base["input_sha256"], **evidence, **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in earlier}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    SCRATCH.mkdir()
    output.mkdir()
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "training_started": False, "confirmation_evaluated": False, "real_benchmarks_evaluated": False})
    status(output, "registered")


def save_endpoint(payload, path):
    if path.parent != OUTPUT or path.exists() or path.is_symlink():
        raise FileExistsError("Keep every fresh full raw endpoint without replacement")
    gradients, snapshot = composite.share(payload["gradients"], payload["snapshot"], len(payload["batch"]["tasks"]))
    shared = {**payload, "gradients": gradients, "snapshot": snapshot}
    target = SCRATCH / path.name
    if shutil.disk_usage(SCRATCH).free - storage.storage_bytes(shared) - storage.METADATA_ALLOWANCE < 50 * 1024**3:
        raise OSError("Preserve50 GiB scratch including every pending endpoint byte")
    digest = storage.save(shared, target)
    path.symlink_to(target)
    write_json(path.with_suffix(".scratch-write.json"), {"path": str(path), "target": str(target), "sha256": digest,
               "receipt_sha256": sha256(target.with_suffix(".write.json")), "scratch_is_provider_persistent": False})
    return digest


def audit_endpoint(payload, path):
    target, receipt = SCRATCH / path.name, path.with_suffix(".scratch-write.json")
    if (not path.is_symlink() or path.resolve() != target or read(receipt) != {"path": str(path), "target": str(target),
            "sha256": sha256(path), "receipt_sha256": sha256(target.with_suffix(".write.json")),
            "scratch_is_provider_persistent": False}):
        raise ValueError("Preserve exact endpoint bytes and declared scratch placement")
    composite.check_alias(payload)
    return {str(receipt): sha256(receipt), **storage.audit(payload, target)}


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
register_node = next(node for node in ast.parse(ADAPTED_SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "register")
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, ast.get_source_segment(ADAPTED_SOURCE, register_node),
                              'def register(output):\n    return register_trial(output)')
for old, new in {
    'import preflight_single_model_structured_teacher as preflight': 'import preflight_single_model_top_block as preflight',
    'OUTPUT = ROOT / "structured_teacher_v20"':
        'OUTPUT = ROOT / "top_block_v24"\nSCRATCH = Path("/mnt/matrixformer_scratch/top_block_v24_raw")',
    'DATA_SEED, GRADIENT_SEED = 381100901, 382100901': 'DATA_SEED, GRADIENT_SEED = 491100901, 492100901',
    'CONFIRMATION_SEEDS = (391100901, 392100901)': 'CONFIRMATION_SEEDS = (501100901, 502100901)',
    'config["learning_rate_schedule"] != {"constant": 2e-5}': 'config["learning_rate_schedule"] != {"constant": RATE}',
    'config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256':
        'config["capture_source_sha256"] != compact_cache.CAPTURE_SOURCE_SHA256\n'
        '            or config["evaluate_source_sha256"] != compact_cache.EVALUATE_SOURCE_SHA256\n'
        '            or config["task_storage_policy"] != compact_cache.POLICY or config["scratch"] != str(SCRATCH)\n'
        '            or config["scratch_is_provider_persistent"] is not False',
    'config["mutable_prefix"] != "decoder."': 'config["mutable_prefixes"] != list(PREFIXES)',
    'view = decoder.DecoderParameterView(model)': 'view = top_block.TopBlockParameterView(model)',
    '"decoder_only_training": True': '"decoder_only_training": False, "top_block_training": True',
    'frozen = decoder.frozen_state_equal(parent_state, model.state_dict(), {name for name, parameter in model.named_parameters() if name.startswith("decoder.")})':
        'frozen = preflight.frozen_inventory(parent_state, model.state_dict(), {name for name, parameter in model.named_parameters() if name.startswith(PREFIXES)})',
    'row[0].startswith("decoder.")': 'row[0].startswith(PREFIXES)',
    'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):':
        'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]), compact_cache.runtime():',
    'with decoder.decoder_capture_coordinates(model, view):': 'with top_block.capture_coordinates(model, view):',
    'entries, tasks, before = replicated.capture_batch(': 'entries, tasks, before = compact_cache.capture_batch(',
    'lambda current: original.evaluate_training(model, criterion, tasks)':
        'lambda current: compact_cache.evaluate_training(model, criterion, tasks)',
    'original.save_tensor({"snapshot": error.snapshot,': 'save_endpoint({"snapshot": error.snapshot,',
    'record["raw_endpoint_sha256"] = original.save_tensor(payload, output / "raw_final_update.pt")':
        'record["raw_endpoint_sha256"] = save_endpoint(payload, output / "raw_final_update.pt")',
    'wait_for_gpu(output, True, False)\n    reference =': 'wait_for_gpu(output, True, False)\n    require_capacity()\n    reference =',
    '"protecting_decoder_update"': '"protecting_top_block_update"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart a started top-block trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
