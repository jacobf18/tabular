"""Test exact synthetic regeneration with compact saved rating-normalization state."""

import probe_single_model_regenerated_training as baseline
import single_model_compact_rating_replay as compact
from single_model_registered_adaptation import install


POLICY_COMPACT = ("A distinct non-promotable diagnostic following the preserved failed seed-only regeneration "
    "probe. Regenerate all64 original batches/1792 episodes and teacher targets. For each of128 ratings "
    "contexts replay only its four original saved normalization vectors after cropping and before the "
    "unchanged arithmetic; never restore matrix source/target/mask grids. Legacy/structured imputation "
    "uses the unchanged sampler without replayed state. Require the same exact whole-task tensor dtype, "
    "shape, stride, bytes, complete metadata/routing and teacher-target equality as the failed probe. "
    "Retain all NaN payloads and signed zero. Record compact-state bytes and typed content digests, not "
    "new full task caches. Stop at the first mismatch and preserve it, without retry or tolerance search. "
    "No update, model, student-gradient replay, confirmation, benchmark, prior-file edit, cache deletion "
    "or claim that seed-only regeneration succeeded. This is feasibility evidence for a separately "
    "registered future training-storage procedure, not authorization to change past audits.")


def require_seed_only_failure():
    baseline.verify()
    current = baseline.read(baseline.OUTPUT / "status.json")
    mismatch = baseline.read(baseline.OUTPUT / "mismatch.json")
    rows = [baseline.json.loads(line) for line in (baseline.OUTPUT / "tasks.jsonl").read_text().splitlines()]
    if (current["state"] != "failed" or current["error"] != "ValueError('Regenerated tensor differs exactly: episode.source')"
            or baseline.Path(f'/proc/{current["pid"]}').exists() or mismatch["step"] != 1
            or mismatch["task"]["index"] != 13 or mismatch["task"]["name"] != "recsys_draw_0"
            or mismatch["task"]["specification"]["seed"] != 441102201
            or [row["index"] for row in rows] != list(range(13))
            or any(row["step"] != 1 or not row["exact_episode_and_teacher"] for row in rows)
            or (baseline.OUTPUT / "result.json").exists()):
        raise ValueError("Preserve the exact stopped seed-only failure after13 successful imputation tasks")
    paths = [baseline.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "mismatch.json",
                                               "first_generated_mismatch.pt", "tasks.jsonl")]
    hashes = {str(path): baseline.sha256(path) for path in paths}
    if hashes[str(baseline.OUTPUT / "first_generated_mismatch.pt")] != mismatch["generated_sha256"]:
        raise ValueError("Preserve the actual failed generated artifact")
    return hashes


install(baseline, globals(), {
    'OUTPUT = training.ROOT / "teacher_continuation_v22_regeneration_probe_v1"':
        'OUTPUT = training.ROOT / "teacher_continuation_v22_compact_replay_probe_v1"',
    '    hashes = terminal_evidence()': '    hashes = {**terminal_evidence(), **require_seed_only_failure()}',
    'REPO / "tests/test_single_model_regenerated_training.py"':
        'REPO / "tests/test_single_model_compact_training.py", REPO / "tests/test_single_model_compact_rating_replay.py"',
    '"teacher_identities": config["teacher_identities"], "input_sha256": hashes,':
        '"teacher_identities": config["teacher_identities"], "input_sha256": hashes,\n'
        '        "compact_policy": compact.POLICY, "adapted_source_sha256": ADAPTED_SOURCE_SHA256,\n'
        '        "rating_sampler_source_sha256": compact.ADAPTED_SOURCE_SHA256,',
    'or config["teacher_identities"] != original["teacher_identities"]):':
        'or config["teacher_identities"] != original["teacher_identities"]\n'
        '            or config["compact_policy"] != compact.POLICY or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256\n'
        '            or config["rating_sampler_source_sha256"] != compact.ADAPTED_SOURCE_SHA256):',
    '    count, roles = 0, {"initialization": 0, "general": 0, "specialist": 0}':
        '    count, roles = 0, {"initialization": 0, "general": 0, "specialist": 0}\n'
        '    rating_contexts, scaling_bytes = 0, 0',
    '                episode = sampler.sample(step=1, **task["specification"])':
        '                saved_scaling = cached["episode"]["scaling"]\n'
        '                if (saved_scaling is not None) != (role == "specialist"):\n'
        '                    raise ValueError("Replay compact normalization only for ratings")\n'
        '                with compact.runtime(saved_scaling) as replayed:\n'
        '                    episode = sampler.sample(step=1, **task["specification"])\n'
        '                rating_contexts += replayed["calls"]\n'
        '                scaling_bytes += replayed["vector_bytes"]',
    '"exact_episode_and_teacher": True, "content_sha256": digest})':
        '"exact_episode_and_teacher": True, "content_sha256": digest,\n'
        '                    "rating_state_replayed": replayed["calls"] == 1, "scaling_bytes": replayed["vector_bytes"],\n'
        '                    "scaling_content_sha256": content_digest(saved_scaling)})',
    '    if count != 1792 or roles != {"initialization": 128, "general": 1536, "specialist": 128}:':
        '    if count != 1792 or rating_contexts != 128 or roles != {"initialization": 128, "general": 1536, "specialist": 128}:',
    '"all_episode_and_teacher_bytes_match": True, "records_sha256": sha256(OUTPUT / "tasks.jsonl"),':
        '"all_episode_and_teacher_bytes_match": True, "records_sha256": sha256(OUTPUT / "tasks.jsonl"),\n'
        '        "rating_contexts_replayed": rating_contexts, "normalization_vector_bytes": scaling_bytes,\n'
        '        "seed_only_regeneration": False, "prior_seed_only_failure_preserved": True,',
}, {"POLICY": POLICY_COMPACT})


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
