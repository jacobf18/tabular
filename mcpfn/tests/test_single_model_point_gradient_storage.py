import pytest
import torch

import single_model_composite_gradient_storage as composite
import single_model_point_gradient_storage as storage
import single_model_shared_tensor_storage as accounting


def payload():
    gradients = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    original = {"task_gradients": gradients.clone(), "optimizer_before": {"state": {}}}
    complete, snapshot = storage.share(gradients, original, 2)
    assert snapshot is not original and snapshot["optimizer_before"] is original["optimizer_before"]
    assert original["task_gradients"].data_ptr() != complete.data_ptr()
    return {"gradients": complete, "snapshot": snapshot, "record": {"composite_tasks": 2, "protected_tasks": 6}}


def test_roundtrip_preserves_complete_bytes_and_counts_one_storage(tmp_path):
    current = payload()
    path = tmp_path / "endpoint.pt"
    torch.save(current, path)
    restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    storage.check_alias(restored)
    assert storage.byte_equal(current["gradients"], restored["gradients"])
    assert accounting.storage_bytes(restored) == 24 * 4
    with pytest.raises(ValueError):
        composite.check_alias(restored)


@pytest.mark.parametrize("corruption", ["copy", "prefix", "missing_points", "count", "dtype", "empty"])
def test_audit_rejects_lost_full_proposal_or_false_alias(corruption):
    current = payload()
    if corruption == "copy":
        current["snapshot"]["task_gradients"] = current["snapshot"]["task_gradients"].clone()
    elif corruption == "prefix":
        current["snapshot"]["task_gradients"] = current["gradients"][:2]
    elif corruption == "missing_points":
        current["gradients"] = current["gradients"][:2]
    elif corruption == "count":
        current["record"]["protected_tasks"] = 5
    elif corruption == "empty":
        current["gradients"] = torch.empty((6, 0))
        current["snapshot"]["task_gradients"] = current["gradients"]
    else:
        current["snapshot"]["task_gradients"] = current["gradients"].double()
    with pytest.raises(ValueError):
        storage.check_alias(current)


def test_share_rejects_prefix_and_signed_zero_substitution():
    gradients = torch.ones((6, 4))
    with pytest.raises(ValueError):
        storage.share(gradients, {"task_gradients": gradients[:2].clone()}, 2)
    gradients[0, 0] = 0.
    changed = gradients.clone()
    changed[0, 0] = -0.
    with pytest.raises(ValueError):
        storage.share(gradients, {"task_gradients": changed}, 2)


@pytest.mark.parametrize("count", [0, -1, True, 2., 6])
def test_share_rejects_invalid_task_count(count):
    gradients = torch.zeros((6, 4))
    with pytest.raises(ValueError):
        storage.share(gradients, {"task_gradients": gradients.clone()}, count)
