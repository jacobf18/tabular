import pytest

import archive_single_model_continuation_capacity as capacity


def test_preserves_inventory_and_space_floors():
    assert capacity.NAMES == capacity.baseline.NAMES
    assert capacity.OUTPUT != capacity.baseline.OUTPUT
    assert capacity.OUTPUT.name == "continuation_snapshot_capacity_v1"
    assert capacity.MINIMUM_ROOT_FREE == 3 * 1024**3
    assert capacity.TARGET_ROOT_FREE == 20 * 1024**3
    assert "spectral_optimizer_20260915" not in capacity.NAMES
    assert 'inputs.update(require_unused_inventory())' in capacity.ADAPTED_SOURCE


@pytest.mark.parametrize("complete,count,preserved,digest,valid", [
    (True, 0, True, "expected", True),
    (False, 0, True, "expected", False),
    (True, 1, True, "expected", False),
    (True, 0, False, "expected", False),
    (True, 0, True, "changed", False),
])
def test_requires_original_unused_inventory(monkeypatch, complete, count, preserved, digest, valid):
    previous = capacity.baseline
    monkeypatch.setattr(previous.gradients.relocation, "terminal", lambda path: None)
    monkeypatch.setattr(previous, "read", lambda path: {
        "complete": complete, "archived_checkpoints": count,
        "all_original_bytes_and_paths_preserved": preserved, "config_sha256": "expected"})
    monkeypatch.setattr(previous, "sha256", lambda path: digest)
    if valid:
        assert len(capacity.require_unused_inventory()) == 3
    else:
        with pytest.raises(ValueError, match="unused supplemental"):
            capacity.require_unused_inventory()
