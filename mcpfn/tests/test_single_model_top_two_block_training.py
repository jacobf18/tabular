from copy import deepcopy

import pytest
import torch

import single_model_top_two_block_training as training
from single_model_distillation import distillation_loss
from single_model_streamed_point_gradients import streamed_point_metric_losses
from tabimpute.model import model_new_stable
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_training import streamed_training_loss
from test_single_model_streamed_point_gradients import OPTIONS, fixture as episode_fixture


def fixture(domain="imputation", dropout=0., blocks=3):
    _, criterion, episode = episode_fixture(domain, dropout=dropout)
    model = TabImputeModelStable(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32,
        num_layers=blocks, num_outputs=20, num_cls=3, rope_fraction=.5, attention_dropout=dropout,
        ffn_dropout=dropout, drop_path_rate=dropout, embedding_dropout=dropout).train()
    return model, criterion, episode


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dropout", [0., .2])
def test_point_forward_and_selected_gradients_equal_full_autograd(domain, checkpoint, dropout):
    reference, criterion, episode = fixture(domain, dropout=dropout)
    model = deepcopy(reference)
    before = deepcopy(model.state_dict())
    view = training.TopTwoBlockParameterView(model)
    assert view.block_indices == (1, 2)
    assert tuple(model.state_dict()) == tuple(before)
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
            if name in view.mutable_names:
                torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=0, atol=0)
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
    view = training.TopTwoBlockParameterView(model)
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
        if name in view.mutable_names:
            torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_update_changes_both_blocks_and_decoder_and_roundtrips_as_ordinary_model(tmp_path):
    model, criterion, episode = fixture()
    ordinary, before = deepcopy(model), deepcopy(model.state_dict())
    view = training.TopTwoBlockParameterView(model)
    _, _, losses = streamed_point_metric_losses(model, criterion, episode, **OPTIONS)
    losses["rmse"].backward()
    torch.optim.AdamW(view.parameters(), lr=1e-4).step()
    result = training.frozen_state_equal(before, model, view)
    for prefix in view.prefixes:
        assert any(name.startswith(prefix) for name in result["changed_trainable_tensors"])
    path = tmp_path / "ordinary.pth"
    torch.save({"model": model.state_dict()}, path)
    ordinary.load_state_dict(torch.load(path, weights_only=True, mmap=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(episode.source), ordinary(episode.source), rtol=0, atol=0)


def test_capture_retains_coordinate_order_and_restores_on_error():
    model, _, _ = fixture()
    view = training.TopTwoBlockParameterView(model)
    original = training.original.coordinates.flatten
    with pytest.raises(RuntimeError, match="stop"):
        with training.capture_coordinates(model, view):
            assert torch.equal(training.original.coordinates.flatten(model), original(view))
            assert torch.equal(training.original.coordinates.flatten(view), original(view))
            raise RuntimeError("stop")
    assert training.original.coordinates.flatten is original


@pytest.mark.parametrize("target", ["encoder", "earlier_block", "selected_buffer", "signed_zero"])
def test_unselected_parameters_and_all_buffers_are_preserved_bitwise(target):
    model, _, _ = fixture()
    view = training.TopTwoBlockParameterView(model)
    if target == "encoder":
        name = next(name for name, parameter in model.named_parameters() if name.startswith("feature_encoder."))
    elif target == "earlier_block":
        name = next(name for name, parameter in model.named_parameters() if name.startswith("transformer_blocks.0."))
    else:
        name = next(name for name, buffer in model.named_buffers()
                    if name.startswith("transformer_blocks.1.") and buffer.is_floating_point())
    current = model.state_dict()
    if target == "signed_zero":
        current[name].reshape(-1)[0] = 0.
    before = deepcopy(model.state_dict())
    current[name].reshape(-1)[0] = -0. if target == "signed_zero" else current[name].reshape(-1)[0] + 1.
    with pytest.raises(ValueError, match="Frozen parameter or buffer"):
        training.frozen_state_equal(before, model, view)


@pytest.mark.parametrize("mutation", ["hidden_trainable", "replace_penultimate", "replace_final", "prefixes", "indices"])
def test_layout_rejects_hidden_training_or_stale_coordinates(mutation):
    model, _, _ = fixture()
    view = training.TopTwoBlockParameterView(model)
    if mutation == "hidden_trainable":
        next(model.feature_encoder.parameters()).requires_grad_(True)
    elif mutation.startswith("replace_"):
        index = -2 if mutation == "replace_penultimate" else -1
        model.transformer_blocks[index] = deepcopy(model.transformer_blocks[index])
    elif mutation == "prefixes":
        view.prefixes = ("decoder.",)
    else:
        view.block_indices = (0, 1)
    with pytest.raises(ValueError, match="Only"):
        training.check_layout(model, view)


@pytest.mark.parametrize("mutation", ["two_blocks", "no_decoder", "wrong_container", "aliased_blocks"])
def test_invalid_architecture_is_rejected_before_changing_gradient_flags(mutation):
    model, _, _ = fixture()
    if mutation == "two_blocks":
        model.transformer_blocks = torch.nn.ModuleList(list(model.transformer_blocks)[:2])
    elif mutation == "no_decoder":
        model.decoder = None
    elif mutation == "wrong_container":
        model.transformer_blocks = torch.nn.Sequential(*model.transformer_blocks)
    else:
        model.transformer_blocks[-1] = model.transformer_blocks[-2]
    before = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError):
        training.TopTwoBlockParameterView(model)
    assert before == [parameter.requires_grad for parameter in model.parameters()]


def test_full_original_architecture_has_exact_scope_without_allocating_weights(monkeypatch):
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
    view = training.TopTwoBlockParameterView(model)
    assert view.block_indices == (6, 7)
    assert view.prefixes == ("transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
    assert tuple(model.state_dict()) == names
    assert len(view.mutable_names) == 62
    assert sum(parameter.numel() for parameter in model.parameters()) == 77998496
    assert sum(parameter.numel() for parameter in view.parameters()) == 27766670
    assert sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad) == 50231826
    assert len(set(model.state_dict()) - set(view.mutable_names)) == 192
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
