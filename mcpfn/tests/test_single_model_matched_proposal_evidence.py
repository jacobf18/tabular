from copy import deepcopy

import pytest
import torch

import single_model_matched_proposal_evidence as evidence
import single_model_paired_protection as paired
import single_model_point_inclusive_proposal as inclusive


def example(method_name, accepted=True):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    gradients = torch.ones((6, 2))
    reference = {"gradients": gradients, "snapshot": {"before": model.weight.detach().flatten().clone()}}
    metrics = {f'legacy_draw_{draw}/{kind}': 1. for draw in (0, 1) for kind in ("rmse", "common_nrmse")}
    candidates = []

    def evaluate(current):
        candidates.append(current.weight.detach().cpu().flatten().clone())
        return {name: (.9 if accepted else 1.1) for name in metrics}

    method = paired if method_name == "composite" else inclusive
    record, snapshot = method.protected_update(model, optimizer, gradients, [.5, .5], .01, metrics, evaluate)
    record.update(step=1, accepted_updates=int(record["accepted"]))
    identity = {"path": "original.pt", "sha256": "original-hash"}
    return method, record, snapshot, reference, identity, candidates[0], optimizer.state_dict()


@pytest.mark.parametrize("method_name", ["composite", "point_inclusive"])
@pytest.mark.parametrize("accepted", [False, True])
def test_lossless_compaction_and_complete_independent_audit(method_name, accepted, tmp_path):
    method, record, snapshot, reference, identity, candidate, actual = example(method_name, accepted)
    compact = evidence.compact(snapshot, record, reference, identity, method_name, candidate, actual)
    assert set(compact["snapshot"]) == evidence.SNAPSHOT_FIELDS - evidence.SHARED_FIELDS
    assert not evidence.SHARED_FIELDS.intersection(compact["snapshot"])
    path = tmp_path / "compact.pt"
    torch.save(compact, path)
    loaded = torch.load(path, weights_only=True)
    expanded = evidence.expand(loaded, reference, identity, candidate)
    for field in ("before", "after", "task_gradients", "combined_gradient", "applied_gradient", "tentative"):
        assert evidence.identical(expanded["snapshot"][field], snapshot[field])
    method.check_record(expanded["record"], [.5, .5], record["point_before"], 6)
    audit = method.check_endpoint(expanded, record, [2], .02, .003, actual)
    assert audit["accepted"] == accepted and audit["gradient_vectors"] == 6


@pytest.mark.parametrize("field", ["before", "task_gradients", "after"])
def test_changed_external_reference_cannot_replace_actual_tensor(field):
    method, record, snapshot, reference, identity, candidate, actual = example("point_inclusive")
    altered = deepcopy(snapshot)
    altered[field].flatten()[0] += .001
    with pytest.raises(ValueError):
        evidence.compact(altered, record, reference, identity, "point_inclusive", candidate, actual)


def test_bitwise_comparison_preserves_dtype_and_signed_zero():
    assert not evidence.identical(torch.tensor([0.]), torch.tensor([-0.]))
    assert not evidence.identical(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert evidence.identical(torch.tensor([-0.]), torch.tensor([-0.]))


def test_unknown_snapshot_fields_and_changed_optimizer_moments_are_rejected():
    method, record, snapshot, reference, identity, candidate, actual = example("composite")
    with pytest.raises(ValueError, match="unknown"):
        evidence.compact({**snapshot, "extra": torch.ones(2)}, record, reference, identity, "composite", candidate, actual)
    changed = deepcopy(actual)
    next(iter(changed["state"].values()))["exp_avg"].add_(1.)
    with pytest.raises(ValueError, match="moments"):
        evidence.compact(snapshot, record, reference, identity, "composite", candidate, changed)


@pytest.mark.parametrize("field,value", [("reference", {}), ("after_source", "initialization"), ("final_optimizer_bitwise_verified", False)])
def test_compact_reference_tampering_is_rejected(field, value):
    method, record, snapshot, reference, identity, candidate, actual = example("point_inclusive")
    compact = evidence.compact(snapshot, record, reference, identity, "point_inclusive", candidate, actual)
    compact[field] = value
    with pytest.raises(ValueError, match="complete compact"):
        evidence.expand(compact, reference, identity, candidate)


def test_storage_bound_counts_every_control_and_prediction_role():
    expected = 3 * (6 * 11227016 * 4 + 2 * 11461696 + 3 * 4 * 1024**2) + 16 * 1024**2
    assert evidence.storage_allowance(11227016, 11461696) == expected == 931641280
    for parameters, prediction_bytes, cases in ((0, 1, 3), (1, -1, 3), (1, 1, 2)):
        with pytest.raises(ValueError):
            evidence.storage_allowance(parameters, prediction_bytes, cases)
