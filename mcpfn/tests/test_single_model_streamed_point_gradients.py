"""Check exact-forward anchoring and parameter derivatives on small original models."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from single_model_distillation import Episode
from single_model_streamed_point_gradients import streamed_point_metric_losses, streamed_point_predictions
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_single_model_point_metric_gradients import reference_imputation, reference_ratings
from test_streamed_training import assert_gradients_match, example


OPTIONS = {"attention_batch_size": 2, "cell_batch_size": 13,
           "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5}


def fixture(domain, dropout=0.):
    model, source, target, mask = example((1, 9, 7), dropout)
    ratings = None
    if domain == "recsys":
        ratings = SimpleNamespace(means=torch.full((7,), 3.), stds=torch.linspace(.8, 1.2, 7),
                                  row_offsets=torch.linspace(-.2, .2, 9), row_scales=torch.linspace(.8, 1.2, 9))
    episode = Episode(source, target, mask, {"domain": domain, "gradient_data": "synthetic", "seed": 10}, ratings)
    return model, FullSupportBarDistribution(torch.linspace(-5, 5, 21)), episode


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_parameter_gradients_match_direct_autograd_on_same_streamed_forward(domain, checkpoint):
    model, criterion, episode = fixture(domain)
    expected_model = deepcopy(model)
    prediction, records, losses = streamed_point_metric_losses(model, criterion, episode, use_checkpoint=checkpoint, **OPTIONS)
    expected_prediction = streamed_point_predictions(expected_model, criterion, episode.source, episode.mask,
                                                     use_checkpoint=checkpoint, **OPTIONS)
    torch.testing.assert_close(prediction, expected_prediction, rtol=0, atol=0)
    expected = (reference_imputation if domain == "imputation" else reference_ratings)(expected_prediction.double(), episode)
    for name, loss in losses.items():
        assert float(loss.detach()) == records[name]["value"]
        torch.testing.assert_close(loss, expected[name], rtol=1e-12, atol=1e-12)
        loss.backward(retain_graph=True)
        expected[name].backward(retain_graph=True)
        assert_gradients_match(expected_model, model)
        model.zero_grad(set_to_none=True)
        expected_model.zero_grad(set_to_none=True)
    assert (prediction[~episode.mask] == 0).all()


def test_dense_prediction_and_gradient_parity():
    model, criterion, episode = fixture("imputation")
    reference = deepcopy(model)
    actual = streamed_point_predictions(model, criterion, episode.source, episode.mask, **OPTIONS)
    expected = criterion.mean(reference(episode.source, decode_mask=episode.mask))
    torch.testing.assert_close(actual[episode.mask], expected, rtol=1e-5, atol=1e-6)
    actual[episode.mask].square().mean().backward()
    expected.square().mean().backward()
    assert_gradients_match(reference, model)


def test_checkpoint_preserves_dropout_forward_and_gradients():
    model, criterion, episode = fixture("imputation", dropout=.2)
    reference = deepcopy(model)
    torch.manual_seed(17)
    expected = streamed_point_predictions(reference, criterion, episode.source, episode.mask, use_checkpoint=False, **OPTIONS)
    torch.manual_seed(17)
    actual = streamed_point_predictions(model, criterion, episode.source, episode.mask, use_checkpoint=True, **OPTIONS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[episode.mask].square().mean().backward()
    expected[episode.mask].square().mean().backward()
    assert_gradients_match(reference, model)


def test_undefined_nrmse_is_not_a_zero_gradient_objective():
    model, criterion, episode = fixture("imputation")
    episode.target.fill_(2.)
    _, records, losses = streamed_point_metric_losses(model, criterion, episode, **OPTIONS)
    assert records["common_nrmse"] is None and losses["common_nrmse"] is None
    assert losses["rmse"] is not None


@pytest.mark.parametrize("option", list(OPTIONS))
def test_reject_nonpositive_microbatches(option):
    model, criterion, episode = fixture("imputation")
    with pytest.raises(ValueError, match="positive"):
        streamed_point_predictions(model, criterion, episode.source, episode.mask, **{**OPTIONS, option: 0})


@pytest.mark.parametrize("mode", ["empty", "dtype", "shape"])
def test_reject_invalid_mask(mode):
    model, criterion, episode = fixture("imputation")
    mask = {"empty": torch.zeros_like(episode.mask), "dtype": episode.mask.float(), "shape": episode.mask[:, :-1]}[mode]
    with pytest.raises(ValueError, match="boolean query mask"):
        streamed_point_predictions(model, criterion, episode.source, mask, **OPTIONS)
