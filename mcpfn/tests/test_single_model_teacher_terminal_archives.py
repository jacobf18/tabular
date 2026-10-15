import pytest

import archive_single_model_paired_tasks as tasks
import archive_single_model_terminal_checkpoints as checkpoints


def test_only_v19_task_tree_changes_in_scoped_adapter():
    assert tasks.SOURCE.parent.name == "paired_decoder_v19"
    assert tasks.template.SOURCE.parent.name == "replicated_decoder_v18"
    assert tasks.MINIMUM_FREE == tasks.template.MINIMUM_FREE == 18 * 1024**3
    assert 'len(records) != 1856' in tasks.ADAPTED_SOURCE
    assert '1792' in tasks.ADAPTED_SOURCE and '1136' in tasks.ADAPTED_SOURCE


@pytest.mark.parametrize("state,pid", [("training", 999999999), ("training_complete", 1)])
def test_never_archive_live_or_unfinished_checkpoint(monkeypatch, state, pid):
    monkeypatch.setattr(checkpoints, "read", lambda path: {"state": state, "pid": pid})
    with pytest.raises(ValueError, match="immutable completed"):
        checkpoints.require_terminal()


def test_only_terminal_optimizer_checkpoints_not_selected_aliases():
    assert len(checkpoints.RUNS) == 10
    assert "structured_teacher_v20" not in checkpoints.RUNS
    source = __import__("inspect").getsource(checkpoints.checkpoint_inventory)
    assert '"latest_checkpoint.pth"' in source
    assert '"selected_checkpoint.pth"' not in source and '"best_validation_checkpoint.pth"' not in source
    assert checkpoints.MINIMUM_FREE == 18 * 1024**3
