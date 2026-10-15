from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import json

import pytest
import torch

import single_model_anchor_retention as retention
from recsys_evaluation_common import sha256
from single_model_distillation import Episode
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import assert_gradients_match, example


def metadata(component=6, **changes):
    return {"gradient_data": "synthetic", "domain": "imputation", "seed": 123,
            "component": component, "prior": "anchor_latent_mcar", **changes}


@pytest.mark.parametrize("component", range(8))
def test_legacy_eligibility_uses_both_original_native_priors_only(component):
    current = metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert retention.eligible(current) == (component in (6, 7))


@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
@pytest.mark.parametrize("family", ("mcar", "mar", "mnar_latent_factor", "mnar_soft_polarization"))
def test_structured_eligibility_never_uses_missingness_family_or_error(component, family):
    assert retention.eligible(metadata(component, structured=True, family=family, error=1e9)) == (component == "anchor_latent_mcar")


@pytest.mark.parametrize("changes", [dict(gradient_data="real"), dict(seed=True), dict(seed=-1),
    dict(domain="unknown"), dict(structured=1), dict(component=True), dict(component=8),
    dict(prior="wrong"), dict(structured=True, component="new_prior"), dict(domain="recsys", structured=True)])
def test_malformed_or_real_task_fails_closed(changes):
    with pytest.raises(ValueError):
        retention.eligible(metadata(**changes))


def test_ratings_and_spectral_tasks_have_no_retention_term():
    assert not retention.eligible(metadata(domain="recsys"))
    assert not retention.eligible(metadata(0))


def episode():
    target = torch.tensor([[[0., 1.], [2., 3.], [4., 5.]]])
    mask = torch.tensor([[[True, False], [True, True], [False, True]]])
    return SimpleNamespace(target=target, source=target.masked_fill(mask, float("nan")),
                           mask=mask, metadata=metadata())


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("weight", [.25, 1.])
def test_retention_loss_gradient_mask_and_detachment_are_exact(monkeypatch, matched, weight):
    current = episode()
    student = torch.nn.Parameter(current.target + .3)
    frozen = (student.detach().clone() if matched else current.target - .4).requires_grad_()
    teacher = (current.target + .1).requires_grad_()
    criterion = SimpleNamespace(mean=lambda logits: logits[:, 0])
    captured = []

    def stream(model, source, target, mask, callback, *, auxiliary_targets, **options):
        selected = auxiliary_targets[mask]
        order = torch.tensor([3, 0, 2, 1])
        selected = selected[order]
        truth = target.reshape(-1)[selected]
        logits = model.reshape(-1)[selected, None]
        result = callback(logits, truth, selected)
        captured.append(result.detach())
        return result.mean()

    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "streamed_training_loss", stream)
    monkeypatch.setattr(retention.column.balanced, "balanced_loss",
        lambda criterion, logits, truth, auxiliary, **options: (criterion.mean(logits) - auxiliary.detach()).square())
    coefficients, offset, _ = retention.query_budget_majorizer(current.target, current.mask, teacher, seed=128)
    expected = ((student - teacher.detach()).square() + coefficients * (student - current.target).square()
                + 10. * (student - frozen.detach()).square())[current.mask].mean() + offset
    expected_gradient = torch.autograd.grad(weight * expected, student, retain_graph=True)[0]
    value = retention._backward_episode(student, criterion, current, teacher, weight,
        query_seed_offset=5, retention_target=frozen)
    assert value == pytest.approx(float(expected.detach()), rel=1e-6)
    torch.testing.assert_close(student.grad, expected_gradient)
    assert not student.grad[~current.mask].any()
    assert frozen.grad is None and teacher.grad is None and len(captured) == 1


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite", "empty", "ineligible"])
def test_invalid_reference_never_drops_queries(mutation):
    current = episode()
    prediction = current.target.clone()
    if mutation == "shape":
        prediction = prediction[:, :1]
    elif mutation == "dtype":
        prediction = prediction.long()
    elif mutation == "nonfinite":
        prediction[0, 0, 0] = float("nan")
    elif mutation == "empty":
        current.mask.zero_()
    elif mutation == "ineligible":
        current.metadata = metadata(0)
    with pytest.raises(ValueError):
        retention.validate_target(current, prediction)


def runtime_fixture(tmp_path, monkeypatch):
    reference = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    checkpoint = tmp_path / "reference.pt"
    torch.save({"model": reference.state_dict()}, checkpoint)
    identity = {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint)}
    monkeypatch.setattr(retention, "REFERENCE_SHA256", identity["sha256"])
    backend = SimpleNamespace(backward_episode=partial(retention.query.backward_episode, query_seed_offset=5))
    return reference, identity, backend


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_keeps_original_teacher_saves_all_tasks_and_restores_callback(tmp_path, monkeypatch, failure):
    reference, identity, backend = runtime_fixture(tmp_path, monkeypatch)
    original, current = backend.backward_episode, episode()
    student = deepcopy(reference).requires_grad_(True)
    teacher = current.target + .5
    expected = current.target + .2
    calls = []
    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "streamed_point_predictions", lambda model, source, criterion, **options: expected)

    def backward(model, criterion, actual, target, weight, **options):
        assert model is student and target is teacher
        assert torch.equal(options["retention_target"], expected)
        calls.append(True)
        if failure:
            raise RuntimeError("diagnostic interruption")
        return 3.

    monkeypatch.setattr(retention, "_backward_episode", backward)
    directory = tmp_path / "retention"
    if failure:
        with pytest.raises(RuntimeError, match="interruption"):
            with retention.runtime(backend, directory, reference, identity, query_seed_offset=5):
                backend.backward_episode(student, None, current, teacher, 1.)
    else:
        with retention.runtime(backend, directory, reference, identity, query_seed_offset=5):
            assert backend.backward_episode(student, None, current, teacher, 1.) == 3.
            with pytest.raises(ValueError, match="reuse"):
                backend.backward_episode(student, None, current, teacher, 1.)
    assert backend.backward_episode is original and calls == [True]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] is not failure and manifest["inference_ensemble"] is False
    assert len(manifest["records"]) == 1 and manifest["records"][0]["eligible"] is True
    cached = torch.load(manifest["records"][0]["artifact"], weights_only=True)
    torch.testing.assert_close(cached["source"], current.source, equal_nan=True)
    assert torch.equal(cached["mask"], current.mask) and torch.equal(cached["prediction"], expected)
    assert cached["identity"] == identity and manifest["records"][0]["sha256"] == sha256(directory / "reference_123.pt")


def test_reference_identity_failure_does_not_create_artifacts_or_change_callback(tmp_path, monkeypatch):
    reference, identity, backend = runtime_fixture(tmp_path, monkeypatch)
    original = backend.backward_episode
    identity["sha256"] = "wrong"
    directory = tmp_path / "retention"
    with pytest.raises(ValueError):
        with retention.runtime(backend, directory, reference, identity, query_seed_offset=5):
            pytest.fail("must not enter")
    assert backend.backward_episode is original and not directory.exists()


@pytest.mark.parametrize("transposed", [False, True])
def test_actual_streamed_retention_matches_dense_full_model_gradients(monkeypatch, transposed):
    dense, source, truth, mask = example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    model = deepcopy(dense)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth, requires_grad=True)
    frozen = torch.randn_like(truth, requires_grad=True)
    weights, offset, _ = retention.query_budget_majorizer(truth, mask, teacher, seed=128)
    logits = dense(source, decode_mask=mask)
    means = criterion.mean(logits.float())
    losses = retention.column.balanced.balanced_loss(criterion, logits, truth[mask], teacher[mask],
        domain="imputation", probability_scales=retention.column.balanced.PROBABILITY_SCALES)
    expected = (losses + weights[mask] * (means - truth[mask]).square()
                + 10. * (means - frozen.detach()[mask]).square()).mean() + offset
    monkeypatch.setattr(retention, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
        "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    current = Episode(source, truth, mask, metadata())
    actual = retention._backward_episode(model, criterion, current, teacher, .5,
        query_seed_offset=5, retention_target=frozen)
    (.5 * expected).backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    assert_gradients_match(dense, model)
    assert frozen.grad is None and teacher.grad is None


@pytest.mark.parametrize("domain,component", [("imputation", 0), ("recsys", 6)])
def test_nonanchor_runtime_uses_exact_original_callback_without_reference_inference(tmp_path, monkeypatch, domain, component):
    calls = []

    def original(*arguments, **keywords):
        calls.append((arguments, keywords))
        return 7.

    monkeypatch.setattr(retention.query, "backward_episode", original)
    reference, identity, backend = runtime_fixture(tmp_path, monkeypatch)
    saved = backend.backward_episode
    current = episode()
    current.metadata = metadata(component, domain=domain)
    monkeypatch.setattr(retention, "streamed_point_predictions", lambda *args, **kwargs: pytest.fail("No retention inference"))
    with retention.runtime(backend, tmp_path / "retention", reference, identity, query_seed_offset=5):
        assert backend.backward_episode(None, None, current, current.target, .5) == 7.
    assert backend.backward_episode is saved and len(calls) == 1
    assert calls[0][0] == (None, None, current, current.target, .5)
    assert calls[0][1] == {"query_seed_offset": 5}
    manifest = json.loads((tmp_path / "retention/manifest.json").read_text())
    assert manifest["complete"] is True and len(manifest["records"]) == 1
    assert manifest["records"][0]["eligible"] is False and "artifact" not in manifest["records"][0]
