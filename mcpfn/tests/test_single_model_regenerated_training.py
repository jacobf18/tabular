from copy import deepcopy
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import probe_single_model_regenerated_training as probe


def payload():
    return {"source": torch.tensor([[1.0, float("nan")], [0.0, -0.0]]),
            "mask": torch.tensor([[True, False], [False, True]]),
            "metadata": {"seed": 123, "shape": [2, 2], "scale": 1.0, "flag": False}, "scaling": None}


def test_clone_keeps_nan_payload_signed_zero_metadata_and_typed_digest():
    original = payload()
    current = deepcopy(original)
    probe.require_equal(current, original, "test")
    assert probe.content_digest(current) == probe.content_digest(original)
    reordered = {key: current[key] for key in reversed(current)}
    probe.require_equal(reordered, original, "test")
    assert probe.content_digest(reordered) == probe.content_digest(original)


@pytest.mark.parametrize("mutation", ["zero_sign", "tensor_value", "dtype", "shape", "stride", "metadata", "extra", "list_type", "bool_type"])
def test_rejects_any_exact_tensor_or_metadata_change(mutation):
    original, current = payload(), payload()
    if mutation == "zero_sign":
        current["source"][1, 1] = 0.0
    elif mutation == "tensor_value":
        current["source"][0, 0] += 1e-6
    elif mutation == "dtype":
        current["source"] = current["source"].double()
    elif mutation == "shape":
        current["source"] = current["source"].flatten()
    elif mutation == "stride":
        current["source"] = current["source"].t().contiguous().t()
    elif mutation == "metadata":
        current["metadata"]["seed"] += 1
    elif mutation == "extra":
        current["extra"] = None
    elif mutation == "list_type":
        current["metadata"]["shape"] = (2, 2)
    else:
        current["metadata"]["flag"] = 0
    with pytest.raises(ValueError):
        probe.require_equal(current, original, "test")
    assert probe.content_digest(current) != probe.content_digest(original)


def test_dtype_and_scalar_boundaries_are_part_of_digest():
    assert probe.content_digest({"values": [1, 23]}) != probe.content_digest({"values": [12, 3]})
    assert probe.content_digest([True]) != probe.content_digest([1])
    assert probe.content_digest((1, 2)) != probe.content_digest([1, 2])
    assert probe.content_digest(torch.ones(2)) != probe.content_digest(torch.ones(2, dtype=torch.float64))
    with pytest.raises(TypeError):
        probe.content_digest({1: "value"})
    with pytest.raises(TypeError):
        probe.require_equal(object(), object(), "unknown")


def test_scalar_tensor_digest_and_shape():
    current = torch.tensor(-0.0)
    probe.require_equal(current.clone(), current, "scalar")
    assert probe.content_digest(current.clone()) == probe.content_digest(current)
    assert probe.content_digest(current.reshape(1)) != probe.content_digest(current)


def test_probe_regenerates_all_original_specs_without_student_updates_or_rng_replay():
    source = inspect.getsource(probe.run)
    assert 'for batch in config["batches"]:' in source
    assert 'sampler.sample(step=1, **task["specification"])' in source
    assert 'backend.teacher_target(' in source
    assert 'require_equal(generated, cached, "complete_task")' in source
    assert 'first_generated_mismatch.pt' in source
    assert 'optimizer.step(' not in source and 'backward_episode(' not in source
    assert 'restore_episode(' not in source and 'restore_rng(' not in source


def test_real_cli_help():
    result = subprocess.run([sys.executable, str(Path(probe.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
