from contextlib import nullcontext
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

import single_model_native_column_mean as retention
import single_model_general_native_retention as predecessor
import test_single_model_anchor_retention as originals
import test_single_model_column_anchor_retention as columns
import test_single_model_cohort_anchor_retention as cohorts
import test_single_model_stronger_mean_distillation as mean_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize("component", range(8))
def test_every_native_component_uses_same_column_formula_and_general_reference(component):
    episode = originals.episode()
    episode.metadata = originals.metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    episode.target, episode.mask = columns.column_example()
    with retention.stronger.training_objective():
        weights = retention.teacher_mean_weights(episode)
    assert retention.reference_role(episode.metadata) == "general"
    assert not weights.requires_grad
    torch.testing.assert_close(weights, predecessor.column_coefficients(episode.target, episode.mask).reshape(-1), rtol=0, atol=0)
    episode.metadata.update(seed=999999, error=1e10, failed=True)
    with retention.stronger.training_objective():
        torch.testing.assert_close(retention.teacher_mean_weights(episode), weights, rtol=0, atol=0)


@pytest.mark.parametrize("family", retention.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_every_structured_component_remains_unchanged(family, component):
    episode = originals.episode()
    episode.metadata = originals.metadata(component, structured=True, family=family)
    assert retention.teacher_mean_weights(episode) is None
    assert retention.reference_role(episode.metadata) == "v8"
    assert retention.retention_weighting(episode.metadata) == predecessor.retention_weighting(episode.metadata)


def test_rating_objective_is_not_reweighted():
    episode = originals.episode()
    episode.metadata = originals.metadata(domain="recsys")
    assert retention.teacher_mean_weights(episode) is None
    assert retention.reference_role(episode.metadata) is None


def test_reference_math_and_runtime_are_exact_predecessors():
    assert retention.SPECIFICATION == predecessor.SPECIFICATION
    assert retention.REFERENCE_SHA256S == predecessor.REFERENCE_SHA256S
    assert retention.RUNTIME_SOURCE == predecessor.RUNTIME_SOURCE
    assert retention.BACKWARD_SOURCE_SHA256 != predecessor.BACKWARD_SOURCE_SHA256
    for name in ("_backward_episode", "runtime", "column_coefficients", "independent_penalty", "teacher_mean_weights"):
        function = inspect.unwrap(getattr(retention, name))
        assert function.__globals__ is vars(retention) and function.__closure__ is None
        assert unresolved_globals(function) == []


def test_native_requires_active_mean20_without_mutating_callback():
    original = retention.stronger.original.balanced_loss
    with pytest.raises(ValueError, match="active mean20"):
        retention.teacher_mean_weights(originals.episode())
    assert retention.stronger.original.balanced_loss is original


@pytest.mark.parametrize("coefficient", [.01, 1., 7.])
def test_only_native_mean_weight_changes_and_detaches_teacher(coefficient):
    criterion, logits, truth, teacher = mean_tests.fixture()
    scales = dict(retention.column.balanced.PROBABILITY_SCALES)
    before = retention.stronger.balanced_loss(criterion, logits, truth, teacher, domain="imputation", probability_scales=scales)
    point = criterion.mean(logits)
    weights = torch.full_like(point, coefficient)
    actual = retention.reweight_teacher_mean(before, point, teacher, weights)
    base = retention.stronger.BASE_LOSS(criterion, logits, truth, teacher, domain="imputation", probability_scales=scales)
    huber = functional.huber_loss(point, teacher.detach(), delta=1., reduction="none")
    expected = base - 2. * huber + 20. * weights * huber
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0], rtol=1e-5, atol=2e-6)
    assert torch.autograd.grad(actual.sum(), teacher, allow_unused=True)[0] is None
    if coefficient == 1.:
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["shape", "gradient", "zero", "negative", "nan"])
def test_invalid_mean_coefficients_fail_closed(mutation):
    values = torch.ones(3)
    weights = torch.ones(3)
    if mutation == "shape":
        weights = weights[:2]
    elif mutation == "gradient":
        weights.requires_grad_()
    else:
        weights[0] = {"zero": 0., "negative": -1., "nan": float("nan")}[mutation]
    with pytest.raises(ValueError, match="coefficients"):
        retention.reweight_teacher_mean(values, values, values, weights)


def independently_weighted_mean(point, teacher, truth, mask):
    terms = []
    for column in range(truth.shape[-1]):
        selected = mask[0, :, column]
        count = int(selected.sum())
        if count:
            variance = truth.detach()[0, selected, column].double().var(unbiased=False)
            shrunk = ((count * variance + 1.) / (count + 1.)).clamp_min(1e-6)
            losses = functional.huber_loss(point[0, selected, column], teacher.detach()[0, selected, column], delta=1., reduction="none")
            terms.append(losses.mean() / shrunk)
    return 20. * torch.stack(terms).mean()


@pytest.mark.parametrize("gradient_scale", [1., .125])
@pytest.mark.parametrize("matched", [False, True])
def test_actual_native_callback_matches_independent_column_sum_and_gradients(monkeypatch, gradient_scale, matched):
    episode = originals.episode()
    episode.target, episode.mask = columns.column_example()
    episode.source = episode.target.masked_fill(episode.mask, float("nan"))
    student = torch.nn.Parameter(torch.nan_to_num(episode.target) + .3)
    teacher = (student.detach().clone() if matched else torch.nan_to_num(episode.target) - 2.).requires_grad_()
    reference = (torch.nan_to_num(episode.target) - .4).requires_grad_()
    criterion = SimpleNamespace(mean=lambda logits: logits[:, 0])

    def base(current, logits, truth, target, **options):
        return (logits[:, 0] - truth).square() + 2. * functional.huber_loss(logits[:, 0], target.detach(), delta=1., reduction="none")

    def stream(model, source, target, mask, callback, *, auxiliary_targets, **options):
        selected = auxiliary_targets[mask].flip(0)
        return callback(model.reshape(-1)[selected, None], target.reshape(-1)[selected], selected).mean()

    monkeypatch.setattr(retention.stronger, "BASE_LOSS", base)
    monkeypatch.setattr(retention.column.balanced, "balanced_loss", base)
    monkeypatch.setattr(retention, "compute_context", lambda device: nullcontext())
    monkeypatch.setattr(retention, "streamed_training_loss", stream)
    coefficients, offset, _ = retention.query_budget_majorizer(episode.target, episode.mask, teacher, seed=128)
    expected = ((1. + coefficients[episode.mask]) * (student[episode.mask] - episode.target[episode.mask]).square()).mean() + offset
    expected = expected + independently_weighted_mean(student, teacher, episode.target, episode.mask)
    expected = expected + 10. * (retention.column_coefficients(episode.target, episode.mask)[episode.mask]
        * (student[episode.mask] - reference.detach()[episode.mask]).square()).mean()
    expected_gradient = torch.autograd.grad(gradient_scale * expected, student)[0]
    with retention.stronger.training_objective():
        actual = retention._backward_episode(student, criterion, episode, teacher, gradient_scale,
            query_seed_offset=5, retention_target=reference)
    assert actual == pytest.approx(float(expected.detach()), rel=2e-6, abs=2e-6)
    torch.testing.assert_close(student.grad, expected_gradient, rtol=2e-6, atol=2e-6)
    assert teacher.grad is reference.grad is None


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", "mar", "mar_block_neural", "mnar_two_phase_subset"])
def test_real_streamed_model_matches_dense_loss_and_every_parameter(monkeypatch, transposed, cohort):
    dense, source, truth, mask = columns.example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    model = deepcopy(dense)
    criterion = columns.FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    metadata = originals.metadata() if cohort == "native" else originals.metadata("low_rank_mcar", structured=True, family=cohort)
    teacher = torch.randn_like(truth, requires_grad=True)
    reference = torch.randn_like(truth, requires_grad=True)
    coefficients, offset, _ = retention.query_budget_majorizer(truth, mask, teacher, seed=128)
    monkeypatch.setattr(retention, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
        "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    with retention.stronger.training_objective():
        logits = dense(source, decode_mask=mask)
        means = criterion.mean(logits.float())
        losses = retention.stronger.balanced_loss(criterion, logits, truth[mask], teacher[mask],
            domain="imputation", probability_scales=retention.column.balanced.PROBABILITY_SCALES)
        weighted = (losses + coefficients[mask] * (means - truth[mask]).square()
            + 10. * retention.retention_weighting(metadata)["multiplier"] * retention.column_coefficients(truth, mask)[mask]
            * (means - reference.detach()[mask]).square()).mean() + offset
        if cohort == "native":
            grid = torch.zeros_like(truth).masked_scatter(mask, means)
            weighted = weighted - 20. * functional.huber_loss(means, teacher.detach()[mask], delta=1., reduction="mean")
            weighted = weighted + independently_weighted_mean(grid, teacher, truth, mask)
        actual = retention._backward_episode(model, criterion, columns.Episode(source, truth, mask, metadata),
            teacher, 1., query_seed_offset=5, retention_target=reference)
    weighted.backward()
    assert actual == pytest.approx(float(weighted.detach()), rel=2e-6, abs=2e-6)
    for (name, expected), (actual_name, parameter) in zip(dense.named_parameters(), model.named_parameters()):
        assert name == actual_name and expected.grad is not None and parameter.grad is not None
        scale = float(expected.grad.abs().max())
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-4, atol=max(3e-7, 2e-5 * scale), msg=name)
        assert float((parameter.grad - expected.grad).norm()) <= 1e-5 * float(expected.grad.norm()) + 3e-7, name
    assert reference.grad is teacher.grad is None


@pytest.fixture
def runtime_fixture(monkeypatch):
    monkeypatch.setattr(cohorts, "retention", retention)
    references, identities, backend, calls = cohorts.runtime_fixture.__wrapped__(monkeypatch)
    references["general"] = references.pop("v29")
    return references, identities, backend, calls


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_preserves_exact_rating_callback_and_restores_objective(tmp_path, runtime_fixture, failure):
    references, identities, backend, calls = runtime_fixture
    callback = backend.backward_episode
    balanced = retention.stronger.original.balanced_loss
    with pytest.raises(RuntimeError) if failure else nullcontext():
        with retention.stronger.training_objective(), retention.runtime(backend, tmp_path / "references", references, identities, query_seed_offset=5):
            for index, metadata in enumerate((originals.metadata(), originals.metadata("low_rank_mcar", structured=True, family="mar"), originals.metadata(domain="recsys"))):
                episode = originals.episode()
                episode.metadata = {**metadata, "seed": 1000 + index}
                assert backend.backward_episode(torch.nn.Linear(2, 1), None, episode, episode.target, .25) == (7. if index == 2 else 3.)
            if failure:
                raise RuntimeError("synthetic callback failure")
    assert backend.backward_episode is callback
    assert retention.stronger.original.balanced_loss is balanced
    assert [entry[1] for entry in calls if entry[0] == "predict"] == ["general", "v8"]
    manifest = json.loads((tmp_path / "references/manifest.json").read_text())
    assert manifest["complete"] is not failure
    assert [record["reference_role"] for record in manifest["records"]] == ["general", "v8", None]
