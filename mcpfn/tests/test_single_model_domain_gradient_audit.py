from copy import deepcopy
import json
import os

import pytest
import torch

import audit_single_model_domain_gradients as audit


def paired_example():
    specification = deepcopy(audit.probe.specifications()[48])
    payload, metadata, teachers = {}, {}, {}
    for domain in ("imputation", "recsys"):
        request = specification["imputation" if domain == "imputation" else "rating"]
        shape = (1, 3, 2)
        if domain == "recsys":
            request["shape"] = [3, 2]
        metadata[domain] = {"seed": request["seed"], "gradient_data": "synthetic", "domain": domain}
        teachers[domain] = {"target": "initialization" if domain == "imputation" else "specialist"}
        payload[domain] = torch.ones(5)
        payload[f"{domain}_teacher"] = torch.ones(shape)
        payload[f"{domain}_episode"] = {"metadata": deepcopy(metadata[domain]), "source": torch.zeros(shape),
                                       "target": torch.ones(shape), "mask": torch.ones(shape, dtype=torch.bool)}
    record = {**specification, "metadata": metadata, "teachers": teachers, "losses": {"imputation": 1., "recsys": .2}}
    return payload, record, specification


def test_complete_pair_passes_without_cuda_or_an_optimizer():
    audit.verify_payload(*paired_example(), 5)


def test_inventory_cannot_drop_or_reorder_original_parameters():
    model = torch.nn.Linear(3, 2)
    inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
    audit.verify_inventory(inventory, model)
    with pytest.raises(ValueError, match="complete ordered"):
        audit.verify_inventory(inventory[:1], model)
    with pytest.raises(ValueError, match="complete ordered"):
        audit.verify_inventory(list(reversed(inventory)), model)


@pytest.mark.parametrize("change", ["gradient", "teacher", "seed", "source", "query", "rating_shape", "specification"])
def test_changed_pair_is_rejected(change):
    payload, record, specification = paired_example()
    if change == "gradient":
        payload["imputation"][0] = float("nan")
    elif change == "teacher":
        record["teachers"]["imputation"]["target"] = "specialist"
    elif change == "seed":
        record["metadata"]["imputation"]["seed"] += 1
    elif change == "source":
        payload["recsys_episode"]["source"] = torch.zeros(1, 2, 3)
    elif change == "query":
        payload["imputation_episode"]["mask"].zero_()
    elif change == "rating_shape":
        specification = deepcopy(specification)
        specification["rating"]["shape"] = [6, 1]
        record["rating"] = specification["rating"]
    elif change == "specification":
        record["index"] += 1
    with pytest.raises(ValueError):
        audit.verify_payload(payload, record, specification, 5)


def test_wait_requires_a_specific_live_handle(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"state": "measuring_gradients", "pid": os.getpid()}))
    assert not audit.ready(tmp_path)
    path.write_text(json.dumps({"state": "complete"}))
    assert audit.ready(tmp_path)
    path.write_text(json.dumps({"state": "failed", "pid": os.getpid()}))
    with pytest.raises(ValueError, match="never restart"):
        audit.ready(tmp_path)


def test_wait_does_not_treat_a_stale_status_as_live(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "measuring_gradients", "pid": 99999}))
    def missing(pid, signal):
        raise ProcessLookupError
    monkeypatch.setattr(audit.os, "kill", missing)
    with pytest.raises(RuntimeError, match="handle disappeared"):
        audit.ready(tmp_path)
