from copy import deepcopy

import pytest
import torch

import preflight_single_model_decoder_training as preflight
from single_model_decoder_training import DecoderParameterView, decoder_capture_coordinates, frozen_state_equal
from single_model_distillation import distillation_loss
from single_model_streamed_point_gradients import streamed_point_metric_losses
from tabimpute.streamed_training import streamed_training_loss
from test_single_model_streamed_point_gradients import OPTIONS, fixture


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_frozen_backbone_preserves_point_forward_and_decoder_gradients(domain, checkpoint):
    reference, criterion, episode = fixture(domain)
    model = deepcopy(reference)
    before = deepcopy(model.state_dict())
    view = DecoderParameterView(model)
    expected, _, reference_losses = streamed_point_metric_losses(reference, criterion, episode, use_checkpoint=checkpoint, **OPTIONS)
    prediction, _, losses = streamed_point_metric_losses(model, criterion, episode, use_checkpoint=checkpoint, **OPTIONS)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    for metric in losses:
        losses[metric].backward(retain_graph=True)
        reference_losses[metric].backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if name.startswith("decoder."):
                torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
            else:
                assert parameter.grad is None and not parameter.requires_grad
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    result = frozen_state_equal(before, model.state_dict(), {name for name, parameter in view.named_parameters()})
    assert result["backbone_and_buffers_bitwise_unchanged"] and not result["changed_decoder_tensors"]


def test_parameter_capture_adapter_is_local_and_restored_on_failure():
    model, _, _ = fixture("imputation")
    view = DecoderParameterView(model)
    original = preflight.training.alignment.preflight.flatten
    with pytest.raises(RuntimeError):
        with decoder_capture_coordinates(model, view):
            assert torch.equal(preflight.training.alignment.preflight.flatten(model), original(view))
            assert torch.equal(preflight.training.alignment.preflight.flatten(view), original(view))
            raise RuntimeError("test")
    assert preflight.training.alignment.preflight.flatten is original


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
def test_original_composite_loss_and_decoder_gradients_are_preserved(domain):
    reference, criterion, episode = fixture(domain)
    model = deepcopy(reference)
    DecoderParameterView(model)
    weight = .1 if domain == "recsys" else 1.

    def objective(logits, truth, teacher):
        return distillation_loss(criterion, logits, truth, teacher, nll_weight=weight)

    expected = streamed_training_loss(reference, episode.source, episode.target, episode.mask, objective,
                                     auxiliary_targets=episode.target, **OPTIONS)
    actual = streamed_training_loss(model, episode.source, episode.target, episode.mask, objective,
                                   auxiliary_targets=episode.target, **OPTIONS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward()
    actual.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_frozen_parameter_or_buffer_change_is_rejected():
    initial = {"encoder.weight": torch.ones(2), "decoder.weight": torch.ones(2), "buffer": torch.zeros(1)}
    for name in ("encoder.weight", "buffer"):
        current = deepcopy(initial)
        current[name].add_(1)
        with pytest.raises(ValueError, match="Frozen"):
            frozen_state_equal(initial, current, {"decoder.weight"})


def test_preflight_keeps_all_families_weights_and_fresh_seed_pairs():
    batch = preflight.batch()
    assert batch["step"] == 1 and len(batch["tasks"]) == 14
    assert [task["name"] for task in batch["tasks"]] == ["legacy", *preflight.training.alignment.mixture.FAMILIES, "recsys"]
    assert sum(task["weight"] for task in batch["tasks"]) == pytest.approx(1.)
    seeds = preflight.training.alignment.fixed.seeds_in(batch)
    assert len(seeds) == 28
    assert batch["tasks"][0]["specification"]["shape"] == [2000, 1000]
    assert preflight.RATE == 2e-6
