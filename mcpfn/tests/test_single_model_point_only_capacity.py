import builtins
import inspect

import pytest
import torch

import relocate_single_model_point_only_capacity as relocation
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize("mutation", [None, "dtype", "shape", "count", "nan", "mapping"])
def test_vector_validation_retains_exact_finite_parameter_inventory(mutation):
    payload = torch.ones(4)
    if mutation == "dtype":
        payload = payload.double()
    elif mutation == "shape":
        payload = payload.reshape(2, 2)
    elif mutation == "count":
        payload = payload[:3]
    elif mutation == "nan":
        payload[0] = float("nan")
    elif mutation == "mapping":
        payload = {"model": payload}
    if mutation:
        with pytest.raises(ValueError):
            relocation.validate_vector(payload, 4)
    else:
        relocation.validate_vector(payload, 4)


def test_copy_hash_mmap_and_atomic_publication_preserve_vector_bytes(tmp_path, monkeypatch):
    source_directory, destination = tmp_path / "owner", tmp_path / "scratch" / "candidate.pt"
    source_directory.mkdir()
    source = source_directory / "candidate_parameters.pt"
    torch.save(torch.arange(4, dtype=torch.float32), source)
    relocation.write_json(source_directory / "status.json", {"state": "complete"})
    relocation.write_json(source_directory / "config.json", {})
    monkeypatch.setattr(relocation, "OWNERS", {"owner": "candidate_vector"})
    validate = relocation.validate_vector
    monkeypatch.setattr(relocation, "validate_vector", lambda payload: validate(payload, 4))
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
           "sha256": relocation.sha256(source)}
    relocation.copy_and_verify(row)
    assert source.is_file() and not source.is_symlink()
    relocation.copying.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    assert relocation.sha256(source) == relocation.sha256(destination) == row["sha256"]
    torch.testing.assert_close(torch.load(source, weights_only=True, mmap=True), torch.arange(4, dtype=torch.float32))
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)


def test_fixed_seven_file_scope_preserves_selected_models_and_runtime_dependencies():
    assert len(relocation.OWNERS) == 7 and relocation.OWNERS["legacy_point_v31"] == (64, 63, 50)
    assert "point_only_preflight_v1" not in relocation.OWNERS
    assert relocation.copying is not relocation.previous.copying
    assert relocation.previous.copying.copy_and_verify is not relocation.copy_and_verify
    assert 'len(records) != 7' in relocation.MAIN_SOURCE
    assert '"canonical_artifacts_checked": 7' in relocation.MAIN_SOURCE
    assert relocation.MAIN_SOURCE.index('copying.copy_and_verify(row)') < relocation.MAIN_SOURCE.index('copying.publish(row)')
    assert 'verify_hashes(inputs)' in relocation.MAIN_SOURCE
    for name, function in vars(relocation).items():
        if inspect.isfunction(function) and function.__globals__ is vars(relocation):
            assert not code_globals(function.__code__) - vars(relocation).keys() - vars(builtins).keys(), name
