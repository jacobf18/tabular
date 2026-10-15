"""Never relocate selected or unverified checkpoints while restoring training capacity."""

import builtins
from copy import deepcopy
import inspect

import pytest

import relocate_single_model_v27_v29_terminal_tails as relocation
from recover_single_model_legacy_strata_replay import code_globals


def identity():
    current = {"state": "training_complete", "step": 64, "accepted_updates": 63, "best": {"step": 25, "score": 1.06}}
    payload = {"step": 64, "attempted_batches": 64, "gradient_updates_performed": 63, "best_step": 25,
               "best_score": 1.06, "config_sha256": "digest", "independent_confirmation_passed": False,
               "preflight_only": False, "top_two_block_training": True, "model": {"weights": "preserved"},
               "optimizer": {"state": "preserved"}}
    return payload, current, (64, 63, 25)


def test_exact_nonselected_optimizer_state_can_be_preserved_without_mutation():
    payload, current, expected = identity()
    before = deepcopy((payload, current))
    relocation.checkpoint_identity(payload, current, expected, "digest")
    assert (payload, current) == before


@pytest.mark.parametrize("mutation", ["selected", "training", "updates", "step", "attempts", "score", "config",
                                     "qualified", "preflight", "scope", "model", "optimizer"])
def test_reject_selected_unfinished_or_changed_state(mutation):
    payload, current, expected = identity()
    if mutation == "selected":
        payload["step"] = 25
    elif mutation == "training":
        current["state"] = "training"
    else:
        field, value = {"updates": ("gradient_updates_performed", 62), "step": ("step", 63),
                        "attempts": ("attempted_batches", 63), "score": ("best_score", 1.0),
                        "config": ("config_sha256", "changed"), "qualified": ("independent_confirmation_passed", True),
                        "preflight": ("preflight_only", True), "scope": ("top_two_block_training", False),
                        "model": ("model", {}), "optimizer": ("optimizer", {})}[mutation]
        payload[field] = value
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, expected, "digest")


def test_fixed_scope_reuses_copy_fsync_hash_mmap_and_atomic_publication():
    assert relocation.OWNERS == {"general_imputation_teacher_v27": (64, 63, 10), "legacy_strata_v29": (64, 63, 25)}
    assert 'copying.copy_and_verify(row)' in relocation.MAIN_SOURCE
    assert 'copying.publish(row)' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index('copying.copy_and_verify(row)') < relocation.MAIN_SOURCE.index('copying.publish(row)')
    assert 'verify_hashes(inputs)' in relocation.MAIN_SOURCE
    assert '"checkpoint_free_after": shutil.disk_usage(ROOT).free' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 2' in relocation.MAIN_SOURCE
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
