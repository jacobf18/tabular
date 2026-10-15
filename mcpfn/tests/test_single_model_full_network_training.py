from copy import deepcopy

import pytest
import torch

import single_model_full_network_training as training
from single_model_distillation import distillation_loss
from single_model_streamed_point_gradients import streamed_point_metric_losses
from tabimpute.model import model_new_stable
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_training import streamed_training_loss
from test_single_model_top_two_block_training import fixture
from test_single_model_streamed_point_gradients import OPTIONS


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dropout", [0., .2])
def test_point_predictions_and_every_parameter_gradient_equal_ordinary_autograd(domain, checkpoint, dropout):
    reference, criterion, episode = fixture(domain, dropout=dropout)
    model, before = deepcopy(reference), deepcopy(reference.state_dict())
    model.requires_grad_(False)
    view = training.FullNetworkParameterView(model)
    assert view.block_indices == (0, 1, 2)
    assert tuple(model.state_dict()) == tuple(before)
    assert tuple(view.state_dict()) == tuple(before)
    assert all(parameter.requires_grad for parameter in model.parameters())
    torch.manual_seed(981)
    expected, _, reference_losses = streamed_point_metric_losses(reference, criterion, episode,
        use_checkpoint=checkpoint, **OPTIONS)
    torch.manual_seed(981)
    actual, _, losses = streamed_point_metric_losses(model, criterion, episode, use_checkpoint=checkpoint, **OPTIONS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = dict(reference.named_parameters())
    for metric in losses:
        torch.testing.assert_close(losses[metric], reference_losses[metric], rtol=0, atol=0)
        reference_losses[metric].backward(retain_graph=True)
        losses[metric].backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=0, atol=0)
        with training.capture_coordinates(model, view):
            actual_gradient = training.original.coordinates.flatten(model, gradients=True)
            expected_gradient = training.original.coordinates.flatten(reference, gradients=True)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    result = training.frozen_state_equal(before, model, view)
    assert result["unselected_parameters_and_all_buffers_bitwise_unchanged"]
    assert not result["changed_trainable_tensors"]


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_original_composite_loss_and_all_gradients_are_unchanged(domain, checkpoint):
    reference, criterion, episode = fixture(domain)
    model = deepcopy(reference)
    training.FullNetworkParameterView(model)
    weight = .1 if domain == "recsys" else 1.

    def objective(logits, truth, teacher):
        return distillation_loss(criterion, logits, truth, teacher, nll_weight=weight)

    expected = streamed_training_loss(reference, episode.source, episode.target, episode.mask, objective,
        auxiliary_targets=episode.target, use_checkpoint=checkpoint, **OPTIONS)
    actual = streamed_training_loss(model, episode.source, episode.target, episode.mask, objective,
        auxiliary_targets=episode.target, use_checkpoint=checkpoint, **OPTIONS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward()
    actual.backward()
    parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=0, atol=0)


def test_update_changes_encoder_all_blocks_and_decoder_and_saves_ordinary_state(tmp_path):
    model, criterion, episode = fixture()
    ordinary, before = deepcopy(model), deepcopy(model.state_dict())
    view = training.FullNetworkParameterView(model)
    _, _, losses = streamed_point_metric_losses(model, criterion, episode, **OPTIONS)
    losses["rmse"].backward()
    torch.optim.AdamW(view.parameters(), lr=1e-4).step()
    result = training.frozen_state_equal(before, model, view)
    required = ("feature_encoder.", *(f"transformer_blocks.{index}." for index in view.block_indices), "decoder.")
    for prefix in required:
        assert any(name.startswith(prefix) for name in result["changed_trainable_tensors"])
    assert result["frozen_tensors"] == len(list(model.named_buffers()))
    path = tmp_path / "ordinary.pth"
    torch.save({"model": model.state_dict()}, path)
    ordinary.load_state_dict(torch.load(path, weights_only=True, mmap=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(episode.source), ordinary(episode.source), rtol=0, atol=0)


def test_capture_preserves_exact_all_parameter_coordinates_and_restores_on_failure():
    model, _, _ = fixture()
    view = training.FullNetworkParameterView(model)
    original = training.original.coordinates.flatten
    with pytest.raises(RuntimeError, match="stop"):
        with training.capture_coordinates(model, view):
            assert torch.equal(training.original.coordinates.flatten(model), original(model))
            assert torch.equal(training.original.coordinates.flatten(view), original(model))
            raise RuntimeError("stop")
    assert training.original.coordinates.flatten is original


@pytest.mark.parametrize("signed_zero", [False, True])
def test_every_buffer_is_still_bitwise_frozen(signed_zero):
    model, _, _ = fixture()
    view = training.FullNetworkParameterView(model)
    name = next(name for name, buffer in model.named_buffers() if buffer.is_floating_point())
    current = model.state_dict()
    if signed_zero:
        current[name].reshape(-1)[0] = 0.
    before = deepcopy(current)
    current[name].reshape(-1)[0] = -0. if signed_zero else current[name].reshape(-1)[0] + 1.
    with pytest.raises(ValueError, match="Frozen parameter or buffer"):
        training.frozen_state_equal(before, model, view)


@pytest.mark.parametrize("mutation", ["frozen_parameter", "replace_encoder", "replace_block", "replace_decoder", "prefixes", "indices"])
def test_stale_or_incomplete_optimizer_coordinates_are_rejected(mutation):
    model, _, _ = fixture()
    view = training.FullNetworkParameterView(model)
    if mutation == "frozen_parameter":
        next(model.feature_encoder.parameters()).requires_grad_(False)
    elif mutation == "replace_encoder":
        model.feature_encoder = deepcopy(model.feature_encoder)
    elif mutation == "replace_block":
        model.transformer_blocks[0] = deepcopy(model.transformer_blocks[0])
    elif mutation == "replace_decoder":
        model.decoder = deepcopy(model.decoder)
    elif mutation == "prefixes":
        view.prefixes = ("decoder.",)
    else:
        view.block_indices = (1, 2)
    with pytest.raises(ValueError, match="Only"):
        training.check_layout(model, view)


@pytest.mark.parametrize("mutation", ["no_encoder", "no_decoder", "aliased_blocks", "extra_parameter", "wrong_container"])
def test_invalid_architecture_fails_before_gradient_flags_change(mutation):
    model, _, _ = fixture()
    if mutation == "no_encoder":
        model.feature_encoder = None
    elif mutation == "no_decoder":
        model.decoder = None
    elif mutation == "aliased_blocks":
        model.transformer_blocks[-1] = model.transformer_blocks[0]
    elif mutation == "extra_parameter":
        model.extra_parameter = torch.nn.Parameter(torch.ones(1))
    else:
        model.transformer_blocks = torch.nn.Sequential(*model.transformer_blocks)
    model.requires_grad_(False)
    before = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError):
        training.FullNetworkParameterView(model)
    assert before == [parameter.requires_grad for parameter in model.parameters()]


def test_original_model_has_238_trainable_tensors_and_16_frozen_buffers(monkeypatch):
    original_schedule = model_new_stable._build_drop_path_schedule

    def cpu_schedule(num_layers, drop_path_rate):
        with torch.device("cpu"):
            return original_schedule(num_layers, drop_path_rate)

    monkeypatch.setattr(model_new_stable, "_build_drop_path_schedule", cpu_schedule)
    with torch.device("meta"):
        model = TabImputeModelStable(embedding_size=768, num_attention_heads=16, mlp_hidden_size=1536,
            num_layers=8, num_outputs=5000, num_cls=12, rope_fraction=.5, attention_dropout=0.,
            ffn_dropout=0., drop_path_rate=0., embedding_dropout=0., use_absolute_positional_embeddings=False,
            positional_damping_factor=.1, residual_scale_init=1., rms_norm_eps=1e-6, rope_base=10000., use_rope=True)
    names = tuple(model.state_dict())
    view = training.FullNetworkParameterView(model)
    assert view.block_indices == tuple(range(8))
    assert tuple(model.state_dict()) == tuple(view.state_dict()) == names
    assert len(view.mutable_names) == 238 and len(names) == 254
    assert sum(parameter.numel() for parameter in view.parameters()) == 77998496
    assert not any(not parameter.requires_grad for parameter in model.parameters())
    assert set(names) - set(view.mutable_names) == {name for name, buffer in model.named_buffers()}
    assert len(list(model.named_buffers())) == 16
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
