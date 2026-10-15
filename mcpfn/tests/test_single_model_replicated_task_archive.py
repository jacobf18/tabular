import pytest

from archive_single_model_replicated_tasks import tree_inventory


def test_complete_audited_nested_tree_is_preserved(tmp_path):
    source = tmp_path / "tasks"
    batch = source / "step_001"
    batch.mkdir(parents=True)
    task = batch / "task_00.pt"
    task.write_bytes(b"scientific evidence")
    audited = {str(task): "recorded-hash", str(tmp_path / "unrelated.pt"): "other"}
    assert tree_inventory(source, audited) == {str(task): "recorded-hash"}
    assert task.read_bytes() == b"scientific evidence"


@pytest.mark.parametrize("mutation", ["extra", "missing", "linked_file", "linked_tree", "empty"])
def test_incomplete_or_unaccounted_trees_cannot_be_replaced(tmp_path, mutation):
    source = tmp_path / "tasks"
    source.mkdir()
    task = source / "task.pt"
    task.write_bytes(b"original")
    audited = {str(task): "hash"}
    if mutation == "extra":
        (source / "unknown").write_text("preserve me")
    elif mutation == "missing":
        audited[str(source / "absent.pt")] = "missing"
    elif mutation == "linked_file":
        (source / "link.pt").symlink_to(task)
    elif mutation == "linked_tree":
        linked = tmp_path / "linked"
        linked.symlink_to(source, target_is_directory=True)
        source = linked
    else:
        task.unlink()
        audited.clear()
    with pytest.raises(ValueError):
        tree_inventory(source, audited)
