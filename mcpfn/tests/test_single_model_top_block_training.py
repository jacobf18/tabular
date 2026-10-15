from copy import deepcopy

import pytest
import torch

import single_model_top_block_training as training
from single_model_distillation import distillation_loss
from single_model_streamed_point_gradients import streamed_point_metric_losses
from tabimpute.streamed_training import streamed_training_loss
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model import model_new_stable
from test_single_model_streamed_point_gradients import OPTIONS, fixture


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dropout", [0., .2])
def test_point_forward_and_selected_gradients_match_full_autograd(domain, checkpoint, dropout):
    reference, criterion, episode = fixture(domain, dropout=dropout)
    model = deepcopy(reference)
    before = deepcopy(model.state_dict())
    view = training.TopBlockParameterView(model)
    assert view.block_index == 1
    assert any(name.startswith("transformer_blocks.1.") for name in view.mutable_names)
    assert any(name.startswith("decoder.") for name in view.mutable_names)
    assert tuple(model.state_dict()) == tuple(before)
    torch.manual_seed(991)
    expected, _, reference_losses = streamed_point_metric_losses(reference, criterion, episode,
                                                                use_checkpoint=checkpoint, **OPTIONS)
    torch.manual_seed(991)
    actual, _, losses = streamed_point_metric_losses(model, criterion, episode, use_checkpoint=checkpoint, **OPTIONS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for metric in losses:
        reference_losses[metric].backward(retain_graph=True)
        losses[metric].backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if name in view.mutable_names:
                torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
            else:
                assert parameter.grad is None and not parameter.requires_grad
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    result = training.frozen_state_equal(before, model, view)
    assert result["unselected_parameters_and_all_buffers_bitwise_unchanged"]
    assert not result["changed_trainable_tensors"]


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_original_composite_loss_and_selected_gradients_are_preserved(domain, checkpoint):
    reference, criterion, episode = fixture(domain)
    model = deepcopy(reference)
    view = training.TopBlockParameterView(model)
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
    for name, parameter in model.named_parameters():
        if name in view.mutable_names:
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_optimizer_changes_only_selected_original_tensors_and_ordinary_model_roundtrips(tmp_path):
    model, criterion, episode = fixture("imputation")
    before = deepcopy(model.state_dict())
    ordinary = deepcopy(model)
    view = training.TopBlockParameterView(model)
    _, _, losses = streamed_point_metric_losses(model, criterion, episode, **OPTIONS)
    losses["rmse"].backward()
    torch.optim.AdamW(view.parameters(), lr=1e-4).step()
    result = training.frozen_state_equal(before, model, view)
    assert any(name.startswith("transformer_blocks.1.") for name in result["changed_trainable_tensors"])
    assert any(name.startswith("decoder.") for name in result["changed_trainable_tensors"])
    path = tmp_path / "ordinary_model.pth"
    torch.save({"model": model.state_dict()}, path)
    ordinary.load_state_dict(torch.load(path, weights_only=True, mmap=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(episode.source), ordinary(episode.source), rtol=0, atol=0)


def test_coordinate_capture_preserves_order_and_restores_on_error():
    model, _, _ = fixture("imputation")
    view = training.TopBlockParameterView(model)
    original = training.coordinates.flatten
    with pytest.raises(RuntimeError):
        with training.capture_coordinates(model, view):
            assert torch.equal(training.coordinates.flatten(model), original(view))
            assert torch.equal(training.coordinates.flatten(view), original(view))
            raise RuntimeError("stop")
    assert training.coordinates.flatten is original


@pytest.mark.parametrize("target", ["feature_encoder", "earlier_block", "buffer", "signed_zero"])
def test_frozen_parameters_and_buffers_are_checked_bitwise(target):
    model, _, _ = fixture("imputation")
    view = training.TopBlockParameterView(model)
    current = model.state_dict()
    if target == "feature_encoder":
        name = next(name for name, parameter in model.named_parameters() if name.startswith("feature_encoder."))
    elif target == "earlier_block":
        name = next(name for name, parameter in model.named_parameters() if name.startswith("transformer_blocks.0."))
    else:
        name = next(name for name, buffer in model.named_buffers() if buffer.is_floating_point())
    if target == "signed_zero":
        current[name].reshape(-1)[0] = 0.
    before = deepcopy(model.state_dict())
    current[name].reshape(-1)[0] = -0. if target == "signed_zero" else current[name].reshape(-1)[0] + 1.
    with pytest.raises(ValueError, match="Frozen parameter or buffer"):
        training.frozen_state_equal(before, model, view)


def test_refuses_hidden_trainable_parameters_or_replaced_parameter_identity():
    model, _, _ = fixture("imputation")
    view = training.TopBlockParameterView(model)
    next(model.feature_encoder.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="Only the original"):
        training.check_layout(model, view)
    model.feature_encoder.requires_grad_(False)
    model.transformer_blocks[-1] = deepcopy(model.transformer_blocks[-1])
    with pytest.raises(ValueError, match="Only the original"):
        training.check_layout(model, view)


@pytest.mark.parametrize("mutation", ["no_decoder", "one_block", "wrong_container"])
def test_invalid_architecture_is_rejected_without_changing_existing_gradient_flags(mutation):
    model, _, _ = fixture("imputation")
    if mutation == "no_decoder":
        model.decoder = None
    elif mutation == "one_block":
        model.transformer_blocks = torch.nn.ModuleList([model.transformer_blocks[0]])
    else:
        model.transformer_blocks = torch.nn.Sequential(*model.transformer_blocks)
    before = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError, match="original ordered"):
        training.TopBlockParameterView(model)
    assert before == [parameter.requires_grad for parameter in model.parameters()]


def test_full_original_architecture_has_exact_restricted_inventory_without_allocating_weights(monkeypatch):
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
    original_names = tuple(model.state_dict())
    view = training.TopBlockParameterView(model)
    assert view.block_index == 7
    assert tuple(model.state_dict()) == original_names
    assert sum(parameter.numel() for parameter in model.parameters()) == 77998496
    assert sum(parameter.numel() for parameter in view.parameters()) == 19496843
    assert sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad) == 58501653
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert all(name.startswith(("transformer_blocks.7.", "decoder.")) for name in view.mutable_names)
