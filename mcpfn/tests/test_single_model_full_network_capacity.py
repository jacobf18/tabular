import builtins
from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_full_network_capacity as relocation
import test_single_model_top_three_capacity as inherited
from recover_single_model_legacy_strata_replay import code_globals


def fixture(tmp_path, monkeypatch, kind="duplicate_selected_tail"):
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    directory = tmp_path / "owner"
    directory.mkdir()
    selected = 64 if kind == "duplicate_selected_tail" else 0
    expected = {"kind": kind, "run": "owner", "step": 64, "updates": 64, "selected": selected,
                "view_flag": "top_three_block_training"}
    current = {"state": "training_complete", "step": 64, "accepted_updates": 64, "best": {"step": selected, "score": 1.}}
    relocation.write_json(directory / "config.json", {})
    digest = relocation.sha256(directory / "config.json")
    payload = {"step": 64, "attempted_batches": 64, "gradient_updates_performed": 64, "best_step": selected,
               "best_score": 1., "config_sha256": digest, "independent_confirmation_passed": False,
               "preflight_only": False, "top_three_block_training": True,
               "model": {"weights": torch.tensor([0., 1.])}, "optimizer": {"state": "preserved"}}
    relocation.write_json(directory / "status.json", current)
    if selected:
        torch.save({key: deepcopy(value) for key, value in payload.items() if key != "optimizer"}, directory / "selected_checkpoint.pth")
    return payload, current, expected, digest


@pytest.mark.parametrize("kind", ["duplicate_selected_tail", "unselected_tail"])
@pytest.mark.parametrize("mutation", [None, "live", "updates", "attempts", "config", "qualified", "preflight", "scope", "optimizer", "selection"])
def test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, kind, mutation):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch, kind)
    if mutation == "live":
        current["state"] = "training"
    elif mutation:
        key, value = {"updates": ("gradient_updates_performed", 63), "attempts": ("attempted_batches", 63),
                      "config": ("config_sha256", "changed"), "qualified": ("independent_confirmation_passed", True),
                      "preflight": ("preflight_only", True), "scope": ("top_three_block_training", False),
                      "optimizer": ("optimizer", {}), "selection": ("best_step", 10)}[mutation]
        payload[key] = value
    if mutation:
        with pytest.raises(ValueError):
            relocation.checkpoint_identity(payload, current, expected, digest)
    else:
        relocation.checkpoint_identity(payload, current, expected, digest)


@pytest.mark.parametrize("mutation", ["weights", "signed_zero", "dtype", "shape", "keys", "nonfinite", "optimizer", "step", "config", "link"])
def test_duplicate_tail_cannot_replace_or_change_the_retained_selected_model(tmp_path, monkeypatch, mutation):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch)
    path = tmp_path / "owner/selected_checkpoint.pth"
    selected = torch.load(path, weights_only=False)
    if mutation == "weights":
        selected["model"]["weights"].add_(1.)
    elif mutation == "signed_zero":
        selected["model"]["weights"][0] = -0.
    elif mutation == "dtype":
        selected["model"]["weights"] = selected["model"]["weights"].double()
    elif mutation == "shape":
        selected["model"]["weights"] = torch.ones(3)
    elif mutation == "keys":
        selected["model"]["other"] = torch.ones(1)
    elif mutation == "nonfinite":
        payload["model"]["weights"].fill_(float("nan"))
        selected["model"]["weights"].fill_(float("nan"))
    elif mutation == "optimizer":
        selected["optimizer"] = {"state": "not model-only"}
    elif mutation == "step":
        selected["step"] = 63
    elif mutation == "config":
        selected["config_sha256"] = "changed"
    if mutation == "link":
        destination = tmp_path / "moved_selection.pth"
        path.rename(destination)
        path.symlink_to(destination)
    else:
        torch.save(selected, path)
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, expected, digest)


def test_copy_publish_recheck_preserves_durable_selected_file_and_all_serialized_tail_bytes(tmp_path, monkeypatch):
    payload, current, expected, digest = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(relocation, "OWNERS", {"owner": expected})
    source, selected = tmp_path / "owner/latest_checkpoint.pth", tmp_path / "owner/selected_checkpoint.pth"
    torch.save(payload, source)
    before = selected.stat()
    selected_hash = relocation.sha256(selected)
    row = {"source": str(source), "destination": str(tmp_path / "scratch/latest_checkpoint.pth"),
           "bytes": source.stat().st_size, "sha256": relocation.sha256(source)}
    relocation.copy_and_verify(row)
    assert not source.is_symlink()
    relocation.copying.publish(row)
    assert source.is_symlink() and not selected.is_symlink()
    assert (selected.stat().st_dev, selected.stat().st_ino) == (before.st_dev, before.st_ino)
    assert relocation.sha256(selected) == selected_hash
    assert relocation.sha256(source) == row["sha256"]
    restored = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    relocation.checkpoint_identity(restored, current, expected, digest)
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)


@pytest.mark.parametrize("count", [4, 6])
def test_old_vector_retains_hash_shape_and_mmap_access(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited, "relocation", relocation)
    inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


def test_exact_four_artifacts_keep_selected_files_and_recent_full_probe_out_of_scope():
    assert set(relocation.OWNERS) == {"top_three_blocks_v33", "top_two_blocks_v26", "top_block_v24", "top_three_blocks_preflight_v1"}
    assert relocation.OWNERS["top_two_blocks_v26"]["recovered"] is True
    assert relocation.OWNERS["top_three_blocks_v33"]["selected"] == 0
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify
    assert relocation.TRAINING_ADMISSION_BYTES == 21743271936
    assert 'len(records) != 4' in relocation.MAIN_SOURCE and '"canonical_artifacts_checked": 4' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index('copying.copy_and_verify(row)') < relocation.MAIN_SOURCE.index('copying.publish(row)')
    assert 'verify_hashes(inputs)' in relocation.MAIN_SOURCE
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
