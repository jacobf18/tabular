from copy import deepcopy

import pytest
import torch

import probe_single_model_batch_gradients as probe


def records():
    result = []
    for family in probe.gradients.mixture.FAMILIES:
        for component in probe.gradients.mixture.COMPONENTS:
            result.append({"index": len(result), "group": "structured", "metadata": {
                "imputation": {"domain": "imputation", "gradient_data": "synthetic", "family": family, "component": component},
                "recsys": {"domain": "recsys", "gradient_data": "synthetic"}}})
    for index in range(16):
        result.append({"index": len(result), "group": "legacy", "metadata": {
            "imputation": {"domain": "imputation", "gradient_data": "synthetic"},
            "recsys": {"domain": "recsys", "gradient_data": "synthetic"}}})
    return result


def load(record):
    return {"imputation": torch.tensor([.1, .2] if record["group"] == "structured" else [.3, -.1]),
            "recsys": torch.tensor([-.2, .1])}


def optimizer():
    return {"lr": 5e-7, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .02}


def test_complete_weights_keep_half_legacy_half_structured_and_equal_domains():
    cases = records()
    probe.validate_records(cases)
    for domain in ("imputation", "recsys"):
        assert sum(probe.weights(record)[domain] for record in cases) == pytest.approx(1.)
    for group in ("legacy", "structured"):
        assert sum(probe.weights(record)["imputation"] for record in cases if record["group"] == group) == pytest.approx(.5)
    for family in probe.gradients.mixture.FAMILIES:
        assert sum(probe.weights(record)["imputation"] for record in cases
                   if record["metadata"]["imputation"].get("family") == family) == pytest.approx(.5 / 12)


@pytest.mark.parametrize("kind", ["missing", "order", "stratum", "family", "component", "domain", "not_synthetic"])
def test_original_bank_coverage_and_identities_are_mandatory(kind):
    cases = records()
    if kind == "missing":
        cases.pop()
    elif kind == "order":
        cases.reverse()
    elif kind == "stratum":
        cases[0]["group"] = "legacy"
    elif kind == "family":
        cases[0]["metadata"]["imputation"]["family"] = "unregistered"
    elif kind == "component":
        cases[0]["metadata"]["imputation"]["component"] = cases[1]["metadata"]["imputation"]["component"]
    elif kind == "domain":
        cases[0]["metadata"]["recsys"]["domain"] = "imputation"
    else:
        cases[0]["metadata"]["imputation"]["gradient_data"] = "real"
    with pytest.raises(ValueError):
        probe.validate_records(cases)


def test_linear_accumulation_agrees_when_no_clipping_is_active():
    methods = probe.accumulate(records(), load, [1, 1])
    expected = .5 * (.5 * torch.tensor([.1, .2]) + .5 * torch.tensor([.3, -.1])) + .5 * torch.tensor([-.2, .1])
    for method in methods.values():
        torch.testing.assert_close(method, expected, rtol=1e-6, atol=1e-7)


def test_before_and_after_accumulation_clipping_are_distinct_and_bounded():
    def large(record):
        return {"imputation": torch.tensor([-10., 1.] if record["group"] == "structured" else [1., 1.]),
                "recsys": torch.zeros(2)}

    methods = probe.accumulate(records(), large, [2])
    assert not torch.allclose(methods[probe.METHODS[0]], methods[probe.METHODS[1]])
    assert all(float(vector.norm()) <= .5 for vector in methods.values())
    expected = .5 * probe.adam.clipped(torch.tensor([-5., .5]), [2], .25)
    expected += .5 * probe.adam.clipped(torch.tensor([.5, .5]), [2], .25)
    torch.testing.assert_close(methods["mean_of_clipped_examples"], expected)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan"])
def test_no_gradient_coordinate_may_be_changed_or_omitted(kind):
    def invalid(record):
        payload = load(record)
        if kind == "shape":
            payload["recsys"] = torch.ones(3)
        elif kind == "dtype":
            payload["recsys"] = payload["recsys"].double()
        else:
            payload["recsys"][0] = torch.nan
        return payload

    with pytest.raises(ValueError):
        probe.accumulate(records(), invalid, [2])


def test_every_vector_is_measured_for_every_direction_without_modifying_parameters():
    parameter = torch.tensor([1., -2.])
    original = parameter.clone()
    cases = records()
    methods = probe.directions(parameter, probe.accumulate(cases, load, [2]), optimizer())
    assert torch.equal(parameter, original)
    rows = probe.measure(cases, load, methods)
    summary = probe.summarize(rows)
    assert len(rows) == 768 and len(summary) == 84
    assert sum(row["cases"] for row in summary) == 768
    assert {row["group"] for row in summary} == {"legacy", "recsys", *probe.gradients.mixture.FAMILIES}
    assert all(row["linearized_loss_delta"] == -row["dot_with_descent_direction"] for row in rows)
    text = probe.markdown(summary)
    assert "not NRMSE/RMSE/MAE improvement" in text
    assert "No candidate checkpoint is produced" in text


@pytest.mark.parametrize("kind", ["missing", "duplicate", "group", "nonfinite"])
def test_summary_retains_complete_directional_scope(kind):
    cases = records()
    methods = probe.directions(torch.zeros(2), probe.accumulate(cases, load, [2]), optimizer())
    rows = probe.measure(cases, load, methods)
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "group":
        rows[0]["group"] = "unregistered"
    else:
        rows[0]["dot_with_descent_direction"] = float("nan")
    with pytest.raises(ValueError):
        probe.summarize(rows)


def test_zero_gradients_do_not_invent_descent_or_cosines():
    cases = records()
    empty = lambda record: {"imputation": torch.zeros(2), "recsys": torch.zeros(2)}
    methods = probe.directions(torch.ones(2), probe.accumulate(cases, empty, [2]), optimizer())
    rows = probe.measure(cases, empty, methods)
    assert all(row["cosine"] is None and row["linearized_loss_delta"] == 0. and not row["adverse"] for row in rows)
    assert not any(row["group_mean_adverse"] for row in probe.summarize(rows))
