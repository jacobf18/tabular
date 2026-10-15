from copy import deepcopy
from types import SimpleNamespace

import pytest

import relocate_single_model_v23_confirmation_bank as relocation


def test_scope_is_only_consumed_failed_bank_and_declares_local_storage():
    assert relocation.OWNER.name == "compact_continuation_v23_confirmation"
    assert relocation.DESTINATION.name == "synthetic_bank.pt"
    assert relocation.DESTINATION.is_relative_to(relocation.scratch.SCRATCH)
    assert "not provider-persistent storage or an external backup" in relocation.POLICY
    assert "V24 training continues untouched" in relocation.POLICY


def test_complete_bank_inventory_is_required():
    episode = {name: None for name in ("source", "target", "mask", "metadata", "scaling")}
    bank = [deepcopy(episode) for index in range(284)]
    relocation.check_bank(bank)
    for value in (bank[:-1], {}, [None] * 284):
        with pytest.raises(ValueError, match="complete284-case"):
            relocation.check_bank(value)
    bank[-1].pop("mask")
    with pytest.raises(ValueError, match="complete284-case"):
        relocation.check_bank(bank)


@pytest.mark.parametrize("path", [relocation.ROOT, "/", relocation.scratch.SCRATCH])
def test_bank_copy_budget_never_spends_reserves(monkeypatch, path):
    total = 807711191
    capacity = {relocation.ROOT: 18 * 1024**3, "/": 6 * 1024**3,
                relocation.scratch.SCRATCH: 50 * 1024**3 + total + 1024**2}
    monkeypatch.setattr(relocation.shutil, "disk_usage", lambda target: SimpleNamespace(free=capacity[target]))
    relocation.require_capacity(total)
    capacity[path] -= 1
    with pytest.raises(OSError, match="reserves"):
        relocation.require_capacity(total)
