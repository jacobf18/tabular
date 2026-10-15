"""Move two idle archive images onto root while retaining every original byte and path."""

import relocate_single_model_mounted_archives as template
from single_model_registered_adaptation import install


POLICY_TEXT = (template.POLICY.replace("three", "two")
               + " Use the newly completed supplemental18-checkpoint capacity inventory. Its20 GiB target need "
                 "not be reached when the full inventory is exhausted: independently require actual room for "
                 "both image copies plus the unchanged6 GiB reserve before registering this relocation.")
install(template, globals(), {
    'CAPACITY = REPO / "runs/single_model_goal_20261005/legacy_snapshot_capacity_v1"':
        'CAPACITY = REPO / "runs/single_model_goal_20261005/continuation_snapshot_capacity_v1"',
    'OUTPUT = REPO / "runs/single_model_goal_20261005/mounted_archive_relocation_v1"':
        'OUTPUT = REPO / "runs/single_model_goal_20261005/continuation_archive_relocation_v1"',
    'not report.get("capacity_target_reached")':
        '(report.get("archived_checkpoints") != 18 or report.get("candidate_checkpoints") != 18)',
    'all three images': 'both images',
    'Only the three fixed terminal archive images may move': 'Only the two fixed terminal archive images may move',
    'REPO / "tests/test_single_model_mounted_archive_relocation.py"]':
        'Path(template.__file__).resolve(), REPO / "tests/test_single_model_continuation_evidence.py"]',
}, {"NAMES": ("paired_endpoint_archive_v1", "structured_teacher_endpoint_archive_v1"), "POLICY": POLICY_TEXT})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
