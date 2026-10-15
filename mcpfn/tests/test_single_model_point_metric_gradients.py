from types import SimpleNamespace

import numpy as np
import pytest
import torch

import single_model_point_metric_gradients as metrics
from single_model_distillation import Episode, episode_metrics


def imputation_fixture(dtype=torch.float64):
    truth = torch.tensor([[[1., 2., 7.], [3., 4., 7.], [5., 6., 7.], [8., 8., 7.]]], dtype=dtype)
    mask = torch.tensor([[[True, True, False], [True, False, True], [True, True, True], [False, True, True]]])
    prediction = truth + torch.tensor([[[.2, -.3, .4], [.5, .1, -.1], [-.7, -.8, .6], [.9, .7, -.2]]], dtype=dtype)
    episode = Episode(truth.masked_fill(mask, torch.nan), truth, mask, {"domain": "imputation", "gradient_data": "synthetic", "seed": 10})
    return prediction, episode


def reference_imputation(prediction, episode):
    residual = prediction[episode.mask] - episode.target[episode.mask]
    result = {"rmse": torch.linalg.vector_norm(residual) / np.sqrt(residual.numel())}
    columns = []
    for column in range(prediction.shape[-1]):
        selected = episode.mask[0, :, column]
        truth = episode.target[0, selected, column]
        if truth.numel():
            scale = float(np.std(truth.detach().cpu().numpy()))
            if scale >= 1e-3:
                error = prediction[0, selected, column] - truth
                columns.append(torch.linalg.vector_norm(error) / (np.sqrt(error.numel()) * scale))
    result["common_nrmse"] = torch.stack(columns).mean() if columns else None
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_imputation_values_match_original_dtype_std_and_derivatives_match_autograd(dtype):
    prediction, episode = imputation_fixture(dtype)
    actual = metrics.point_metric_terms(prediction, episode)
    trainable = prediction.double().requires_grad_()
    expected = reference_imputation(trainable, episode)
    for name, value in expected.items():
        gradient = torch.autograd.grad(value, trainable, retain_graph=True)[0]
        assert actual[name]["value"] == pytest.approx(float(value.detach()), rel=1e-12, abs=1e-12)
        torch.testing.assert_close(actual[name]["gradient"], gradient, rtol=1e-12, atol=1e-12)
    assert actual["common_nrmse"]["defined_units"] == 2
    assert (actual["common_nrmse"]["gradient"][:, :, 2] == 0).all()


def rating_fixture():
    truth = torch.tensor([[[1.2, -.4, .6], [.4, -.1, .9], [0., 0., 0.]]], dtype=torch.float32)
    prediction = torch.tensor([[[-8., .3, 9.], [.8, .6, -.2], [2., 2., 2.]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True, True], [True, False, False], [False, False, False]]])
    scaling = SimpleNamespace(means=torch.tensor([2.1, 2.4, 2.7]), stds=torch.tensor([.7, 1.1, .9]),
                              row_offsets=torch.tensor([.2, -.3, .1]), row_scales=torch.tensor([1.2, .8, 1.4]))
    episode = Episode(truth.masked_fill(mask, torch.nan), truth, mask, {"domain": "recsys", "gradient_data": "synthetic", "seed": 20}, scaling)
    return prediction, episode


def reference_ratings(prediction, episode):
    rating = episode.ratings
    scale = rating.stds[None, None, :] * rating.row_scales[None, :, None]
    location = rating.means[None, None, :] + rating.row_offsets[None, :, None]
    residual = (prediction * scale + location).clamp(1., 5.) - (episode.target.double() * scale + location)
    errors = [residual[0, row][episode.mask[0, row]] for row in range(prediction.shape[1]) if episode.mask[0, row].any()]
    return {"rmse": torch.stack([torch.linalg.vector_norm(error) / np.sqrt(error.numel()) for error in errors]).mean(),
            "mae": torch.stack([error.abs().mean() for error in errors]).mean()}


def test_ratings_preserve_equal_user_weight_inverse_scaling_and_clipping():
    prediction, episode = rating_fixture()
    actual = metrics.point_metric_terms(prediction, episode)
    frozen = episode_metrics(prediction, episode)
    trainable = prediction.clone().requires_grad_()
    reference = reference_ratings(trainable, episode)
    for name, value in reference.items():
        gradient = torch.autograd.grad(value, trainable, retain_graph=True)[0]
        assert actual[name]["value"] == pytest.approx(frozen[name], rel=1e-12, abs=1e-12)
        assert actual[name]["value"] == pytest.approx(float(value.detach()), rel=1e-12, abs=1e-12)
        torch.testing.assert_close(actual[name]["gradient"], gradient, rtol=1e-12, atol=1e-12)
        assert actual[name]["defined_units"] == 2
        assert actual[name]["gradient"][0, 0, 0] == actual[name]["gradient"][0, 0, 2] == 0
        assert (actual[name]["gradient"][~episode.mask] == 0).all()


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
def test_linear_surrogate_has_exact_anchor_value_and_metric_gradient(domain):
    prediction, episode = imputation_fixture() if domain == "imputation" else rating_fixture()
    for record in metrics.point_metric_terms(prediction, episode).values():
        coefficients, offset = metrics.linearization(prediction, episode.mask, record)
        trainable = prediction.clone().requires_grad_()
        objective = (trainable[episode.mask] * coefficients[episode.mask]).mean() + offset
        objective.backward()
        assert float(objective.detach()) == pytest.approx(record["value"], rel=1e-12, abs=1e-12)
        torch.testing.assert_close(trainable.grad, record["gradient"], rtol=1e-12, atol=1e-12)


def test_zero_residual_has_zero_subgradient_and_undefined_columns_stay_null():
    prediction, episode = imputation_fixture()
    records = metrics.point_metric_terms(episode.target.clone(), episode)
    for record in records.values():
        assert record["value"] == 0 and (record["gradient"] == 0).all()
    episode.target.fill_(1.)
    record = metrics.point_metric_terms(prediction, episode)
    assert record["common_nrmse"] is None
    with pytest.raises(ValueError, match="undefined"):
        metrics.linearization(prediction, episode.mask, record["common_nrmse"])


def test_unscored_missing_predictions_do_not_poison_derivatives():
    prediction, episode = imputation_fixture()
    original = metrics.point_metric_terms(prediction, episode)
    prediction[~episode.mask] = torch.nan
    result = metrics.point_metric_terms(prediction, episode)
    for name in result:
        assert result[name]["value"] == original[name]["value"]
        assert torch.equal(result[name]["gradient"], original[name]["gradient"])


@pytest.mark.parametrize("kind", ["real", "unknown", "missing_scaling", "nonfinite", "mask", "shape"])
def test_invalid_metric_data_is_rejected(kind):
    prediction, episode = imputation_fixture()
    if kind == "real":
        episode.metadata["gradient_data"] = "real"
    elif kind == "unknown":
        episode.metadata["domain"] = "other"
    elif kind == "missing_scaling":
        episode.metadata["domain"] = "recsys"
    elif kind == "nonfinite":
        prediction[0, 0, 0] = torch.nan
    elif kind == "mask":
        episode.mask.fill_(False)
    else:
        prediction = prediction[:, :-1]
    with pytest.raises(ValueError):
        metrics.point_metric_terms(prediction, episode)


@pytest.mark.parametrize("name", ["stds", "row_scales"])
def test_invalid_rating_scale_is_not_refitted(name):
    prediction, episode = rating_fixture()
    getattr(episode.ratings, name)[0] = 0.
    with pytest.raises(ValueError, match="positive"):
        metrics.point_metric_terms(prediction, episode)


def test_linearization_cannot_change_the_scored_mask():
    prediction, episode = imputation_fixture()
    record = metrics.point_metric_terms(prediction, episode)["rmse"]
    mask = episode.mask.clone()
    mask[0, 0, 0], mask[0, 0, 2] = False, True
    with pytest.raises(ValueError):
        metrics.linearization(prediction, mask, record)


def test_zero_derivative_does_not_allow_a_different_query_mask():
    prediction, episode = imputation_fixture()
    prediction = episode.target.clone()
    record = metrics.point_metric_terms(prediction, episode)["rmse"]
    mask = episode.mask.clone()
    mask[0, 0, 0], mask[0, 0, 2] = False, True
    assert not record["gradient"].any()
    with pytest.raises(ValueError):
        metrics.linearization(prediction, mask, record)


def test_exact_rating_clamp_boundaries_use_the_stated_pytorch_subgradient():
    truth = torch.tensor([[[2., 4.]]], dtype=torch.float64)
    prediction = torch.tensor([[[1., 5.]]], dtype=torch.float64)
    scaling = SimpleNamespace(means=torch.zeros(2), stds=torch.ones(2), row_offsets=torch.zeros(1), row_scales=torch.ones(1))
    episode = Episode(truth, truth, torch.ones_like(truth, dtype=torch.bool), {"domain": "recsys", "gradient_data": "synthetic"}, scaling)
    for record in metrics.point_metric_terms(prediction, episode).values():
        torch.testing.assert_close(record["gradient"], torch.tensor([[[-.5, .5]]], dtype=torch.float64), rtol=0., atol=0.)


def test_rating_zero_residual_has_finite_zero_subgradients():
    prediction, episode = rating_fixture()
    records = metrics.point_metric_terms(episode.target.double(), episode)
    for record in records.values():
        assert record["value"] == 0. and not record["gradient"].any()


def test_common_column_threshold_uses_the_original_target_dtype_std():
    truth = torch.tensor([[[0., 0., 4.], [.002, .0019, 4.]]], dtype=torch.float32)
    prediction = truth + .25
    episode = Episode(truth, truth, torch.ones_like(truth, dtype=torch.bool), {"domain": "imputation", "gradient_data": "synthetic"})
    record = metrics.point_metric_terms(prediction, episode)["common_nrmse"]
    assert record["defined_units"] == 1
    error = prediction[0, :, 0].double().numpy() - truth[0, :, 0].double().numpy()
    expected = float(np.sqrt(np.mean(error ** 2)) / np.std(truth[0, :, 0].numpy()))
    assert record["value"] == pytest.approx(expected, rel=1e-12)
    assert (record["gradient"][:, :, 1:] == 0).all()
