from copy import deepcopy

import pytest
import torch

import single_model_composite_gradient_storage as storage
import single_model_shared_tensor_storage as accounting
from preflight_single_model_point_inclusive import shared_gradients as incompatible_full_alias
import train_single_matrixformer_family_projection as composite


def test_actual_composite_adam_snapshot_has_only_proposal_rows():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    gradients = torch.arange(1, 253, dtype=torch.float32).reshape(84, 3) / 1000
    weights = [1. / 28] * 28
    record, snapshot = composite.update_model(model, optimizer, gradients[:28], weights, False, 2e-5, raw=True)
    assert snapshot["task_gradients"].shape == (28, 3)
    with pytest.raises(ValueError, match="exact complete original"):
        incompatible_full_alias(gradients, snapshot)
    original_proposal = snapshot["task_gradients"]
    preserved = deepcopy(snapshot)
    complete, shared = storage.share(gradients, snapshot, 28)
    assert snapshot["task_gradients"] is original_proposal
    assert storage.byte_equal(complete, gradients)
    assert storage.byte_equal(shared["task_gradients"], preserved["task_gradients"])
    assert shared["task_gradients"].shape == (28, 3)
    assert shared["record"] == record
    assert shared["optimizer_before"] is snapshot["optimizer_before"]
    assert shared["optimizer_after"] is snapshot["optimizer_after"]
    storage.check_alias({"gradients": complete, "snapshot": shared,
                         "record": {"composite_tasks": 28, "protected_tasks": 84}})


def payload():
    gradients = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    complete, snapshot = storage.share(gradients, {"task_gradients": gradients[:2].clone()}, 2)
    return {"gradients": complete, "snapshot": snapshot, "record": {"composite_tasks": 2, "protected_tasks": 6}}


def test_byte_equality_handles_scalar_model_buffers():
    assert storage.byte_equal(torch.tensor(3.0), torch.tensor(3.0))
    assert not storage.byte_equal(torch.tensor(-0.0), torch.tensor(0.0))
    assert not storage.byte_equal(torch.tensor(3.0), torch.tensor(3))


def test_roundtrip_preserves_all_rows_and_counts_one_storage(tmp_path):
    current = payload()
    path = tmp_path / "endpoint.pt"
    torch.save(current, path)
    restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    storage.check_alias(restored)
    assert storage.byte_equal(current["gradients"], restored["gradients"])
    assert storage.byte_equal(current["snapshot"]["task_gradients"], restored["snapshot"]["task_gradients"])
    assert accounting.storage_bytes(restored) == 24 * 4


@pytest.mark.parametrize("corruption", ["copy", "all_rows", "missing_points", "count", "dtype"])
def test_audit_rejects_false_alias_or_lost_original_layout(corruption):
    current = payload()
    if corruption == "copy":
        current["snapshot"]["task_gradients"] = current["snapshot"]["task_gradients"].clone()
    elif corruption == "all_rows":
        current["snapshot"]["task_gradients"] = current["gradients"]
    elif corruption == "missing_points":
        current["gradients"] = current["gradients"][:2]
    elif corruption == "count":
        current["record"]["protected_tasks"] = 5
    else:
        current["snapshot"]["task_gradients"] = current["snapshot"]["task_gradients"].double()
    with pytest.raises(ValueError, match="exact shorter proposal"):
        storage.check_alias(current)


def test_share_rejects_different_prefix_bytes_including_signed_zero():
    gradients = torch.ones((6, 4), dtype=torch.float32)
    proposal = gradients[:2].clone()
    proposal[0, 0] = 2
    with pytest.raises(ValueError, match="different bytes"):
        storage.share(gradients, {"task_gradients": proposal}, 2)
    gradients[0, 0] = 0.0
    proposal[0, 0] = -0.0
    with pytest.raises(ValueError, match="different bytes"):
        storage.share(gradients, {"task_gradients": proposal}, 2)


@pytest.mark.parametrize("count", [0, -1, True, 2.0, 6])
def test_share_rejects_undefined_proposal_layout(count):
    gradients = torch.zeros((6, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="shorter CPU composite prefix"):
        storage.share(gradients, {"task_gradients": gradients[:2].clone()}, count)
