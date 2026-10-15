import json

import pytest
import torch

import train_single_matrixformer_finite_guarded as training


def test_fixed_fifty_batches_preserve_original_mixture_except_fresh_seeds():
    batches = training.batches()
    originals = training.original.training_batches()[:50]
    assert len(batches) == 50 and batches != originals
    assert len(training.alignment.fixed.seeds_in([batches, training.confirmation_cases()])) == 1684
    for batch, original in zip(batches, originals):
        assert len(batch["tasks"]) == 14
        assert sum(task["weight"] for task in batch["tasks"]) == pytest.approx(1.)
        for task, previous in zip(batch["tasks"], original["tasks"]):
            task["gradient_seed"] = previous["gradient_seed"]
            specification = task["specification"].get("structured", task["specification"])
            old = previous["specification"].get("structured", previous["specification"])
            specification["seed"] = old["seed"]
    assert batches == originals


def test_confirmation_is_reserved_complete_and_json_stable():
    cases = training.confirmation_cases()
    assert len(cases) == 284
    assert json.loads(json.dumps(cases)) == cases
    assert not training.alignment.fixed.seeds_in(cases).intersection(training.alignment.fixed.seeds_in(training.batches()))


def test_storage_estimate_counts_nested_tensor_payloads():
    payload = {"model": torch.zeros(10), "nested": [torch.zeros(5, dtype=torch.float64), {"step": 10}], "label": "endpoint"}
    assert training.tensor_bytes(payload) == 80
