import builtins
import inspect

import pytest

import relocate_single_model_anchor_retention_capacity as relocation
import test_single_model_full_network_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_original_unselected_tail_identity_checks_are_preserved(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, "unselected_tail", mutation)


@pytest.mark.parametrize("count", [4, 6])
def test_old_vectors_remain_hash_and_mmap_identical(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, "relocation", relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


def test_exact_scope_excludes_selected_models_new_probe_and_failed_v28():
    assert set(relocation.OWNERS) == {"full_network_cap90_v36", "full_network_preflight_v1", "point_only_preflight_v1"}
    assert relocation.OWNERS["full_network_cap90_v36"] == {"kind": "unselected_tail", "run": "full_network_cap90_v36",
        "step": 64, "updates": 61, "selected": 0, "grids": 1368, "view_flag": "full_network_training"}
    assert relocation.OWNERS["full_network_preflight_v1"]["parameters"] == 77998496
    assert relocation.OWNERS["point_only_preflight_v1"]["parameters"] == 27766670
    assert relocation.TRAINING_ADMISSION_BYTES == 21743271936
    assert 'len(records) != 3' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index("copying.copy_and_verify(row)") < relocation.MAIN_SOURCE.index("copying.publish(row)")
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify


def test_new_helpers_have_complete_runtime_globals_without_older_closures():
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert function.__closure__ is None, name
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
    for name in ("inventory_original", "checkpoint_identity", "require_idle", "copy_and_verify", "main"):
        assert getattr(relocation, name).__globals__ is vars(relocation)
