"""Preserve the terminal V20 task evidence in bounded, lossless read-only archives."""

import archive_single_model_completed_task_groups as template
from single_model_registered_adaptation import install


def terminal_evidence():
    paths = [ROOT / name for name in (
        "structured_teacher_v20/status.json", "structured_teacher_v20_pipeline/status.json",
        "structured_teacher_v20_pipeline/result.json", "structured_teacher_v20_audit/audit.json",
        "structured_teacher_v20_endpoint_replay/audit.json", "structured_teacher_v20_confirmation/audit.json",
        "structured_teacher_v20_confirmation/gate.json", "structured_teacher_v20_teacher_replay/audit.json")]
    trainer, pipeline, result, audit, replay, confirmation, gate, teachers = map(read, paths)
    if (trainer["state"] != "training_complete" or pipeline["state"] != "complete"
            or any(Path(f'/proc/{current["pid"]}').exists() for current in (trainer, pipeline))
            or not result["complete"] or result["gate"]["passed"]
            or not result["gate"]["confirmation_evaluated"] or result["campaign"] is not None
            or result["review"] is not None or not audit["complete_training_audit"]
            or audit["through_step"] != 64 or not replay["complete"] or replay["endpoint_grids"] != 56
            or not confirmation["complete"] or confirmation["prediction_grids"] != 1136 or gate["passed"]
            or not teachers["complete"] or teachers["full_grids"] != 1792
            or not teachers["full_grid_bitwise_equal"]
            or teachers["role_counts"] != {"initialization": 128, "general": 1536, "specialist": 128}):
        raise ValueError("Require terminal fully audited rejected V20 and complete teacher replay")
    for report in (audit, replay, confirmation, teachers):
        verify_hashes(report["artifact_sha256"])
    return paths, audit


install(template, globals(), {
    'OUTPUT = ROOT / "completed_task_archive_v1"': 'OUTPUT = ROOT / "structured_teacher_task_archive_v1"',
    'groups, inputs = [], []': 'inputs, _audit = terminal_evidence()\n    groups = []',
    'audit["training_episodes"] != steps * 14': 'audit["training_episodes"] != steps * 28',
    'replay["endpoint_grids"] != 28': 'replay["endpoint_grids"] != 56',
    'len(records) != steps * 15': 'len(records) != steps * 29',
    'sum(Path(path).suffix == ".pt" for path in records) != steps * 14':
        'sum(Path(path).suffix == ".pt" for path in records) != steps * 28',
    'REPO / "tests/test_single_model_completed_task_groups.py"]':
        'Path(template.__file__).resolve(), REPO / "tests/test_single_model_structured_teacher_archives.py"]',
    'Lossless archival of five completed audited task trees': 'Lossless archival of the fully audited terminal V20 task tree',
}, {"RUNS": (("structured_teacher_v20", 64, 1216, "structured_teacher_v20_pipeline"),)})


if __name__ == "__main__":
    main()
