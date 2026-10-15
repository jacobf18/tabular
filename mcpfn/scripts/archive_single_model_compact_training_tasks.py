"""Losslessly preserve V22 task caches after complete independent compact-replay verification."""

import inspect

import archive_single_model_completed_task_groups as template
import audit_single_model_compact_training as compact_audit
import probe_single_model_compact_training as probe
from single_model_registered_adaptation import install


SOURCE = probe.training.original_training.OUTPUT / "training_batches"


def validate_finished(status, result, audit):
    if (status["state"] != "complete" or Path(f'/proc/{status["pid"]}').exists()
            or result["complete"] is not True or result["tasks"] != 1792
            or result["rating_contexts_replayed"] != 128 or result["all_episode_and_teacher_bytes_match"] is not True
            or result["seed_only_regeneration"] is not False or result["prior_seed_only_failure_preserved"] is not True
            or result["cached_artifacts_deleted"] is not False or result["training_performed"] is not False
            or not audit["complete"] or audit["tasks"] != 1792 or audit["rating_contexts"] != 128
            or audit["normalization_vector_bytes"] != result["normalization_vector_bytes"]
            or audit["old_artifacts_deleted"] is not False or audit["full_matrix_grids_in_capsules"] is not False
            or audit["student_autograd_replayed"] is not False or audit["future_training_registered"] is not False):
        raise ValueError("Require the stopped complete compact diagnostic and independent capsule audit")


def terminal_evidence():
    probe.verify()
    inputs = probe.baseline.terminal_evidence()
    paths = [probe.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "result.json")]
    paths.extend(compact_audit.OUTPUT / name for name in ("config.json", "audit.json", "capsules.pt"))
    result = read(probe.OUTPUT / "result.json")
    audit = read(compact_audit.OUTPUT / "audit.json")
    validate_finished(read(probe.OUTPUT / "status.json"), result, audit)
    if (sha256(compact_audit.OUTPUT / "config.json") != audit["config_sha256"]
            or sha256(compact_audit.OUTPUT / "capsules.pt") != audit["capsules_sha256"]):
        raise ValueError("Preserve the independently verified complete compact capsule file")
    verify_hashes(audit["artifact_sha256"])
    paths.extend(Path(path) for path in inputs)
    return paths, read(probe.pipeline.AUDIT / "audit.json")


def task_plan():
    paths, audit = terminal_evidence()
    records = tree_inventory(SOURCE, audit["artifact_sha256"])
    if (len(records) != 1856 or sum(Path(path).suffix == ".pt" for path in records) != 1792
            or sum(Path(path).name == "manifest.json" for path in records) != 64):
        raise ValueError("Preserve every original V22 task and manifest despite its failed original coordinator")
    verify_hashes(records)
    return partition(records), paths


install(template, globals(), {
    'OUTPUT = ROOT / "completed_task_archive_v1"': 'OUTPUT = ROOT / "compact_training_task_archive_v1"',
    inspect.getsource(template.plan): 'def plan():\n    return task_plan()\n',
    'REPO / "tests/test_single_model_completed_task_groups.py"]':
        'Path(template.__file__).resolve(), Path(compact_audit.__file__).resolve(), Path(probe.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_compact_training_archives.py"]',
    'Lossless archival of five completed audited task trees':
        'Lossless archival of all original V22 tasks after negative qualification and complete compact replay',
}, {"RUNS": (("teacher_continuation_v22", 64, 1368, "teacher_continuation_v22_cached_recovery_v1_pipeline"),)})


if __name__ == "__main__":
    main()
