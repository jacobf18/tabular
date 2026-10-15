"""Continue the unqualified V22 student using prospectively registered compact evidence."""

import hashlib

import train_single_matrixformer_teacher_continuation as baseline
import train_single_matrixformer_targeted_decoder as schedule
import single_model_compact_training_cache as compact_cache
import single_model_compressed_endpoint as compressed_endpoint
import single_model_composite_gradient_storage as composite_storage
import single_model_replicated_decoder_batch as replicated
import single_model_paired_protection as paired
import single_model_structured_teacher as teacher_policy
import preflight_single_model_compressed_endpoint as storage_preflight
from single_model_registered_adaptation import replace_once, replace_assignment


RAW_GRADIENT_STORAGE = composite_storage.POLICY
PARENT_SHA256 = "8ab2782a88659e6a71bb7121c8c74302e473911a78ac9a851b3e6376feba1d39"
WRITE_BUDGET = 17 * 1024**3 // 4
TRIAL_POLICY = ("Initialize one ordinary single-checkpoint network from the independently audited V22 recovered "
    "selected boundary64, which remains unqualified after failed fresh confirmation. Attempt96 fresh28-task "
    "decoder-only mixed-prior updates with fresh AdamW moments, unchanged constant2e-5 composite proposals, "
    "paired0.1 protection,2x norm bound and exact finite-point rollback. Preserve .25 legacy/.25 structured/"
    ".5 ratings mass, MAR/skip emphasis, original component/shape/regime schedule, observed-only normalization "
    "and Huber/NLL/CDF/query objectives. Original V8 remains the legacy teacher and retention baseline; original "
    "general teaches structured imputation and original specialist teaches ratings. Freeze every nondecoder "
    "tensor. Select only at0/16/32/64/96 with unchanged22-guard V8 retention and strict minimax improvement; "
    "step zero is ineligible. This tests more mixed-prior training, not new teachers or an inference ensemble. "
    "Store new tasks using separately declared compact replay capsules, preserving complete live-batch payloads "
    "for finite acceptance. Save all protected terminal gradients and unchanged composite prefix through the "
    "tested RAM-backed lossless endpoint writer. Use unused data/backward/confirmation streams, not consumed "
    "V22 confirmation. No changed tolerances, real-data tuning, automatic retry or promotion on development.")
TRIAL_QUALIFICATION = ("Independently regenerate and digest-check every one of2688 new complete task/teacher payloads "
    "from its specification and saved four-vector rating state, audit every attempted update and acceptance, "
    "all scheduled development grids, frozen backbone and the complete terminal optimizer/gradient evidence. "
    "Independently reconstruct/replay the endpoint and sole selected152-case development boundary without "
    "reselection. Only then confirm once on284 unused reserved synthetic cases with unchanged original V8/"
    "general retention gates; independently audit all1136 confirmation grids before publishing a best alias. "
    "Only a qualified candidate receives the complete frozen tabular, four-dataset recommender and LLM campaign "
    "and all required comparisons. No fallback, hidden regressions, aggregate-only success, or claim that task "
    "regeneration replays all earlier student gradients. Additive audit/qualification implementations must "
    "preserve this registration and these gates; training alone cannot promote any checkpoint.")


def constant_rate(step):
    if type(step) is not int or not 1 <= step <= STEPS:
        raise ValueError("Require a registered attempted batch from1 through96")
    return 2e-5


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED)
            for batch in schedule.batches()[:STEPS]]


def require_history():
    parent_pipeline = ROOT / "teacher_continuation_v22_cached_recovery_v1_pipeline"
    parent_audit = ROOT / "teacher_continuation_v22_cached_recovery_v1_audit" / "audit.json"
    compact_probe = ROOT / "teacher_continuation_v22_compact_replay_probe_v1"
    compact_audit = ROOT / "teacher_continuation_v22_compact_replay_probe_v1_audit" / "audit.json"
    storage_audit = storage_preflight.OUTPUT / "audit.json"
    for directory in (parent_pipeline, compact_probe, storage_preflight.OUTPUT):
        terminal = read(directory / "status.json")
        if terminal["state"] != "complete" or Path(f'/proc/{terminal["pid"]}').exists():
            raise ValueError("Require idle completed parent qualification and both storage/replay feasibility checks")
    result, audit = read(parent_pipeline / "result.json"), read(parent_audit)
    probe, capsules, storage = read(compact_probe / "result.json"), read(compact_audit), read(storage_audit)
    if (result["complete"] is not True or result["gate"]["passed"] is not False
            or result["gate"]["confirmation_evaluated"] is not True
            or result["campaign"] is not None or result["review"] is not None
            or not audit["complete"] or audit["best"]["step"] != 64 or audit["accepted_updates"] != 59
            or audit["artifact_sha256"].get(str(PARENT)) != PARENT_SHA256 or sha256(PARENT) != PARENT_SHA256
            or not probe["complete"] or probe["tasks"] != 1792 or not probe["all_episode_and_teacher_bytes_match"]
            or probe["seed_only_regeneration"] is not False or not capsules["complete"] or capsules["tasks"] != 1792
            or capsules["student_autograd_replayed"] is not False
            or not storage["complete"] or not storage["actual_filesystem_roundtrip"]
            or storage["training_performed"] is not False):
        raise ValueError("Bind unqualified V22 and the actual successful exact replay/storage preflights")
    verify_hashes({str(PARENT): PARENT_SHA256})
    verify_hashes(storage["artifact_sha256"])
    verify_hashes(read(storage_preflight.OUTPUT / "config.json")["source_sha256"])
    paths = [parent_pipeline / name for name in ("config.json", "registration.json", "status.json", "result.json")]
    paths.extend([parent_audit, compact_probe / "result.json", compact_audit, storage_audit,
                  storage_preflight.OUTPUT / "config.json"])
    return {str(path): sha256(path) for path in paths}


def save_endpoint(payload, path):
    complete, snapshot = composite_storage.share(payload["gradients"], payload["snapshot"],
                                                  payload["record"]["composite_tasks"])
    shared = {**payload, "gradients": complete, "snapshot": snapshot}
    return compressed_endpoint.save(shared, path)


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "teacher_continuation_v22"': 'OUTPUT = ROOT / "compact_continuation_v23"',
    'INITIAL_RUN = ROOT / "structured_teacher_v20"': 'INITIAL_RUN = ROOT / "teacher_continuation_v22_cached_recovery_v1"',
    'STEPS, BOUNDARIES = 64, (0, 10, 25, 50, 64)': 'STEPS, BOUNDARIES = 96, (0, 16, 32, 64, 96)',
    'DATA_SEED, GRADIENT_SEED = 441100901, 442100901': 'DATA_SEED, GRADIENT_SEED = 461100901, 462100901',
    'CONFIRMATION_SEEDS = (451100901, 452100901)': 'CONFIRMATION_SEEDS = (471100901, 472100901)',
    'shutil.disk_usage(ROOT).free < 31 * 1024**3':
        'shutil.disk_usage(ROOT).free - WRITE_BUDGET < compressed_endpoint.MINIMUM_FREE',
    'Reserve all64 doubled task batches,84-gradient decoder endpoint, validation and18 GiB free space':
        'Reserve4.25 GiB for compact tasks, checkpoints, validation and compressed endpoint above18 GiB; later qualification needs a separate capacity check',
    'len(fresh) != 3868': 'len(fresh) != 5660',
    'Reserve all3584 new data/backward seeds': 'Reserve all5376 new data/backward seeds',
    'REPO / "tests/test_single_model_teacher_continuation.py"':
        'REPO / "tests/test_single_model_compact_continuation.py", REPO / "tests/test_single_model_compact_training_cache.py",\n'
        '               REPO / "scripts/audit_single_model_compact_continuation.py",\n'
        '               REPO / "tests/test_single_model_compact_continuation_audit.py"',
    'config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256':
        'config["capture_source_sha256"] != compact_cache.CAPTURE_SOURCE_SHA256\n'
        '            or config["evaluate_source_sha256"] != compact_cache.EVALUATE_SOURCE_SHA256\n'
        '            or config["task_storage_policy"] != compact_cache.POLICY\n'
        '            or config["endpoint_storage_policy"] != compressed_endpoint.POLICY\n'
        '            or config["training_write_budget_bytes"] != WRITE_BUDGET',
    '"capture_source_sha256": replicated.CAPTURE_SOURCE_SHA256,':
        '"capture_source_sha256": compact_cache.CAPTURE_SOURCE_SHA256,\n'
        '              "evaluate_source_sha256": compact_cache.EVALUATE_SOURCE_SHA256,\n'
        '              "task_storage_policy": compact_cache.POLICY, "endpoint_storage_policy": compressed_endpoint.POLICY,\n'
        '              "training_write_budget_bytes": WRITE_BUDGET,',
    'never promotion of the unqualified V20 training input': 'never promotion of the unqualified V22 training input',
    '"parent_training_updates": 56': '"parent_training_updates": 115',
    'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):':
        'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]), compact_cache.runtime():',
    'entries, tasks, before = replicated.capture_batch(': 'entries, tasks, before = compact_cache.capture_batch(',
    'lambda current: original.evaluate_training(model, criterion, tasks)':
        'lambda current: compact_cache.evaluate_training(model, criterion, tasks)',
    'shared_storage.save({"snapshot": error.snapshot, "record": error.record, "gradients": shared_gradients(gradients[:len(entries)], error.snapshot),':
        'save_endpoint({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)],',
    'payload = {"snapshot": snapshot, "record": record, "gradients": shared_gradients(gradients[:len(entries)], snapshot),':
        'payload = {"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)],',
    'record["raw_endpoint_sha256"] = shared_storage.save(payload, output / "raw_final_update.pt")':
        'record["raw_endpoint_sha256"] = save_endpoint(payload, output / "raw_final_update.pt")',
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
                raise ValueError("Never restart a begun compact continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
