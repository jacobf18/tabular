from copy import deepcopy

import pytest

import archive_single_model_compact_training_tasks as tasks
import archive_single_model_compact_training_evidence as evidence


def finished():
    return [{"state": "complete", "pid": 999999999},
            {"complete": True, "tasks": 1792, "rating_contexts_replayed": 128,
             "all_episode_and_teacher_bytes_match": True, "seed_only_regeneration": False,
             "prior_seed_only_failure_preserved": True, "cached_artifacts_deleted": False,
             "training_performed": False, "normalization_vector_bytes": 2048000},
            {"complete": True, "tasks": 1792, "rating_contexts": 128, "normalization_vector_bytes": 2048000,
             "old_artifacts_deleted": False, "full_matrix_grids_in_capsules": False,
             "student_autograd_replayed": False, "future_training_registered": False}]


def test_requires_complete_stopped_independently_audited_capsules():
    tasks.validate_finished(*finished())


@pytest.mark.parametrize("index,key,value", [(0, "state", "regenerating"), (0, "pid", 1),
    (1, "tasks", 1791), (1, "rating_contexts_replayed", 127), (1, "all_episode_and_teacher_bytes_match", False),
    (1, "seed_only_regeneration", True), (1, "prior_seed_only_failure_preserved", False),
    (1, "cached_artifacts_deleted", True), (1, "training_performed", True),
    (2, "complete", False), (2, "tasks", 100), (2, "normalization_vector_bytes", 0),
    (2, "old_artifacts_deleted", True), (2, "full_matrix_grids_in_capsules", True),
    (2, "student_autograd_replayed", True), (2, "future_training_registered", True)])
def test_rejects_live_partial_or_changed_diagnostic(index, key, value):
    records = deepcopy(finished())
    records[index][key] = value
    with pytest.raises(ValueError, match="stopped complete"):
        tasks.validate_finished(*records)


def test_archive_scope_is_exact_and_preserves_write_floor():
    assert tasks.SOURCE.parent.name == "teacher_continuation_v22"
    assert tasks.OUTPUT.name == "compact_training_task_archive_v1"
    assert evidence.OUTPUT.name == "compact_training_evidence_archive_v1"
    assert tasks.MINIMUM_FREE == evidence.MINIMUM_FREE == 18 * 1024**3
    assert tasks.MAXIMUM_GROUP == 384 * 1024**2
    assert tasks.RUNS == (("teacher_continuation_v22", 64, 1368, "teacher_continuation_v22_cached_recovery_v1_pipeline"),)
    assert '"canonical_artifacts_checked": 2' in evidence.ADAPTED_SOURCE
    assert '"preserved_root_links_checked": 0' in evidence.ADAPTED_SOURCE
