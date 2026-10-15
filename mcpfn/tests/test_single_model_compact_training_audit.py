from copy import deepcopy

import pytest
import torch

import audit_single_model_compact_training as audit


def task(domain):
    return {"index": 13, "name": "example", "weight": .25, "specification": {"domain": domain, "seed": 123, "shape": [3, 2]}}


def scaling():
    return {"means": torch.ones(2), "stds": torch.ones(2), "row_offsets": torch.zeros(3), "row_scales": torch.ones(3)}


def test_capsules_contain_only_specification_digest_and_independent_small_vectors():
    saved = scaling()
    result = audit.make_capsule(1, task("recsys"), "digest", saved)
    assert set(result) == {"step", "task", "content_sha256", "scaling"}
    assert set(result["scaling"]) == set(audit.probe.compact.NAMES)
    assert sum(value.numel() * value.element_size() for value in result["scaling"].values()) == 40
    result["scaling"]["means"][0] = 9.0
    assert saved["means"][0] == 1.0
    assert audit.make_capsule(1, task("imputation"), "digest", None)["scaling"] is None


@pytest.mark.parametrize("domain,state", [("recsys", None), ("imputation", scaling())])
def test_wrong_domain_cannot_smuggle_compact_state(domain, state):
    with pytest.raises(ValueError, match="Only rating"):
        audit.make_capsule(1, task(domain), "digest", state)


def test_matrix_payload_cannot_be_exported_as_compact_state():
    saved = scaling()
    saved["source"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="four normalization vectors"):
        audit.make_capsule(1, task("recsys"), "digest", saved)


def test_independent_record_recheck_detects_content_and_metadata_corruption(tmp_path):
    path = tmp_path / "original.pt"
    cached = {"episode": {"source": torch.ones(1, 3, 2), "scaling": scaling()}, "teacher": torch.zeros(1, 3, 2)}
    torch.save(cached, path)
    expected = {"step": 1, "index": 13, "name": "example", "role": "specialist", "artifact": str(path),
        "artifact_sha256": audit.sha256(path), "exact_episode_and_teacher": True,
        "content_sha256": audit.probe.content_digest(cached), "rating_state_replayed": True, "scaling_bytes": 40,
        "scaling_content_sha256": audit.probe.content_digest(cached["episode"]["scaling"])}
    capsule, size = audit.check_record(expected, 1, task("recsys"), path, cached, "specialist")
    assert size == 40 and capsule["content_sha256"] == expected["content_sha256"]
    for field, value in (("content_sha256", "wrong"), ("scaling_bytes", 0), ("rating_state_replayed", False),
                         ("index", 12), ("artifact_sha256", "wrong")):
        changed = deepcopy(expected)
        changed[field] = value
        with pytest.raises(ValueError):
            audit.check_record(changed, 1, task("recsys"), path, cached, "specialist")
