import builtins
import inspect

import pytest

import relocate_single_model_v30_terminal_tail as relocation
import test_single_model_v27_v29_tail_relocation as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_nonselected_step64_with_no_selection_is_valid():
    payload, current, expected = inherited.identity()
    payload.update(gradient_updates_performed=64, best_step=0)
    current.update(accepted_updates=64, best={"step": 0, "score": payload["best_score"]})
    relocation.checkpoint_identity(payload, current, (64, 64, 0), "digest")
    payload["independent_confirmation_passed"] = True
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, (64, 64, 0), "digest")


def test_one_file_scope_preserves_original_copy_and_atomic_publication():
    assert relocation.OWNERS == {"legacy_heavy_v30": (64, 64, 0)}
    assert '"canonical_artifacts_checked": 1' in relocation.MAIN_SOURCE
    assert 'len(records) != 1' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index('copying.copy_and_verify(row)') < relocation.MAIN_SOURCE.index('copying.publish(row)')
    assert 'verify_hashes(inputs)' in relocation.MAIN_SOURCE
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
