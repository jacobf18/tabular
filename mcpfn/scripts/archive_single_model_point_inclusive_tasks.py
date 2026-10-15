"""Preserve V21's fully recovered, unselected training tasks in lossless bounded archives."""

import hashlib

import archive_single_model_structured_teacher_tasks as baseline
import archive_single_model_completed_task_groups as template
from single_model_registered_adaptation import replace_once, replace_assignment


def terminal_evidence():
    paths = [ROOT / name for name in (
        "point_inclusive_v21/status.json", "point_inclusive_v21_pipeline_entry_recovery_v1/status.json",
        "point_inclusive_v21_pipeline_entry_recovery_v1/result.json", "point_inclusive_v21_audit/audit.json",
        "point_inclusive_v21_endpoint_replay/audit.json", "point_inclusive_v21_confirmation/gate.json",
        "point_inclusive_v21_teacher_replay/audit.json", "point_inclusive_v21_pipeline_entry_recovery_v1/recovery_audit.json",
        "point_inclusive_v21_pipeline/status.json")]
    trainer, recovery, result, audit, replay, gate, teachers, corrected, original = map(read, paths)
    if (trainer["state"] != "training_complete" or recovery["state"] != "complete"
            or any(Path(f'/proc/{current["pid"]}').exists() for current in (trainer, recovery, original))
            or not result["complete"] or result["gate"] != gate or gate["passed"]
            or gate["confirmation_evaluated"] or gate["selected_step"] != 0
            or result["campaign"] is not None or result["review"] is not None
            or not audit["complete_training_audit"] or audit["through_step"] != 64 or audit["accepted_updates"] != 63
            or audit["best"]["step"] != 0 or not replay["complete"] or replay["endpoint_grids"] != 56
            or not teachers["complete"] or teachers["full_grids"] != 1792 or not teachers["full_grid_bitwise_equal"]
            or teachers["role_counts"] != {"initialization": 128, "general": 1536, "specialist": 128}
            or not corrected["complete"] or not corrected["original_failure_preserved"]
            or corrected["training_restarted"] or corrected["scientific_configuration_changed"]
            or corrected["teacher_predictions_repeated"] or original["state"] != "failed"):
        raise ValueError("Require terminal fully audited V21 nonselection with the original CLI failure preserved")
    for report in (audit, replay, teachers):
        verify_hashes(report["artifact_sha256"])
    return paths, audit


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "structured_teacher_task_archive_v1"': 'OUTPUT = ROOT / "point_inclusive_task_archive_v1"',
    'REPO / "tests/test_single_model_structured_teacher_archives.py"':
        'Path(baseline.__file__).resolve(), REPO / "tests/test_single_model_point_inclusive_archives.py"',
    'fully audited terminal V20 task tree': 'fully audited recovered terminal V21 task tree',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "RUNS",
    (("point_inclusive_v21", 64, 1216, "point_inclusive_v21_pipeline_entry_recovery_v1"),))
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
