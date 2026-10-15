import builtins
from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_top_three_capacity as relocation
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("view_flag", ["top_block_training", "top_two_block_training"])
@pytest.mark.parametrize("mutation", [None, "selected", "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer"])
def test_tail_identity_preserves_each_original_scope_and_rejects_substitutions(view_flag, mutation):
    current = {"state": "training_complete", "step": 64, "accepted_updates": 61, "best": {"step": 10, "score": 1.06}}
    payload = {"step": 64, "attempted_batches": 64, "gradient_updates_performed": 61, "best_step": 10,
               "best_score": 1.06, "config_sha256": "digest", "independent_confirmation_passed": False,
               "preflight_only": False, view_flag: True, "model": {"weights": "preserved"}, "optimizer": {"state": "preserved"}}
    expected = {"kind": "tail", "step": 64, "updates": 61, "selected": 10, "view_flag": view_flag}
    before = deepcopy((payload, current))
    if mutation == "selected":
        expected["selected"] = expected["step"]
    elif mutation == "live":
        current["state"] = "training"
    elif mutation:
        key, value = {"updates": ("gradient_updates_performed", 60), "attempts": ("attempted_batches", 63),
                      "config": ("config_sha256", "changed"), "qualified": ("independent_confirmation_passed", True),
                      "preflight": ("preflight_only", True), "scope": (view_flag, False), "optimizer": ("optimizer", {})}[mutation]
        payload[key] = value
    if mutation:
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, "digest")
    else:
        relocation.checkpoint_identity(payload, current, expected, "digest")
        assert (payload, current) == before


@pytest.mark.parametrize("count", [4, 6])
def test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count):
    directory, destination = tmp_path / "owner", tmp_path / "scratch/candidate.pt"
    directory.mkdir()
    source = directory / "candidate_parameters.pt"
    torch.save(torch.arange(count, dtype=torch.float32), source)
    relocation.write_json(directory / "status.json", {"state": "complete"})
    relocation.write_json(directory / "config.json", {})
    monkeypatch.setattr(relocation, "OWNERS", {"owner": {"kind": "vector", "parameters": count, "frozen": 1}})
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size, "sha256": relocation.sha256(source)}
    relocation.copy_and_verify(row)
    assert not source.is_symlink()
    relocation.copying.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    assert relocation.sha256(source) == relocation.sha256(destination) == row["sha256"]
    torch.testing.assert_close(torch.load(source, weights_only=True, mmap=True), torch.arange(count, dtype=torch.float32))
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(torch.ones(count + 1), {"state": "complete"}, relocation.OWNERS["owner"], "digest")


def test_fixed_four_artifact_scope_keeps_recent_probe_selected_models_and_sources():
    assert set(relocation.OWNERS) == {"point_only_v32", "top_block_continuation_v25", "top_two_blocks_preflight_v1", "top_block_preflight_v1"}
    assert relocation.OWNERS["point_only_v32"]["selected"] == 10
    assert relocation.OWNERS["top_block_continuation_v25"]["selected"] == 32
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify
    assert 'len(records) != 4' in relocation.MAIN_SOURCE and '"canonical_artifacts_checked": 4' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index('copying.copy_and_verify(row)') < relocation.MAIN_SOURCE.index('copying.publish(row)')
    assert 'verify_hashes(inputs)' in relocation.MAIN_SOURCE
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
