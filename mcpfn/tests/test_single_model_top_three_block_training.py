from copy import deepcopy

import pytest
import torch

import single_model_top_three_block_training as training
from single_model_distillation import distillation_loss
from single_model_streamed_point_gradients import streamed_point_metric_losses
from tabimpute.model import model_new_stable
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_training import streamed_training_loss
from test_single_model_top_two_block_training import fixture as original_fixture
from test_single_model_streamed_point_gradients import OPTIONS


def fixture(domain="imputation", dropout=0.):
    return original_fixture(domain, dropout=dropout, blocks=4)


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dropout", [0., .2])
def test_point_forward_and_selected_gradients_equal_full_autograd(domain, checkpoint, dropout):
    reference, criterion, episode = fixture(domain, dropout)
    model, before = deepcopy(reference), deepcopy(reference.state_dict())
    view = training.TopThreeBlockParameterView(model)
    assert view.block_indices == (1, 2, 3)
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
    assert training.frozen_state_equal(before, model, view)["unselected_parameters_and_all_buffers_bitwise_unchanged"]


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_original_composite_loss_and_selected_gradients_are_preserved(domain, checkpoint):
    reference, criterion, episode = fixture(domain)
    model = deepcopy(reference)
    view = training.TopThreeBlockParameterView(model)
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


def test_update_changes_all_selected_blocks_and_roundtrips_as_ordinary_model(tmp_path):
    model, criterion, episode = fixture()
    ordinary, before = deepcopy(model), deepcopy(model.state_dict())
    view = training.TopThreeBlockParameterView(model)
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


def test_capture_keeps_exact_coordinates_and_restores_on_exception():
    model, _, _ = fixture()
    view = training.TopThreeBlockParameterView(model)
    original = training.original.coordinates.flatten
    with pytest.raises(RuntimeError, match="stop"):
        with training.capture_coordinates(model, view):
            assert torch.equal(training.original.coordinates.flatten(model), original(view))
            assert torch.equal(training.original.coordinates.flatten(view), original(view))
            raise RuntimeError("stop")
    assert training.original.coordinates.flatten is original


@pytest.mark.parametrize("target", ["encoder", "earlier_block", "selected_buffer", "signed_zero"])
def test_unselected_parameters_and_all_selected_prefix_buffers_remain_bitwise_frozen(target):
    model, _, _ = fixture()
    view = training.TopThreeBlockParameterView(model)
    if target in {"encoder", "earlier_block"}:
        prefix = "feature_encoder." if target == "encoder" else "transformer_blocks.0."
        name = next(name for name, parameter in model.named_parameters() if name.startswith(prefix))
    else:
        name = next(name for name, buffer in model.named_buffers()
                    if name.startswith("transformer_blocks.1.") and buffer.is_floating_point())
    current = model.state_dict()
    if target == "signed_zero":
        current[name].reshape(-1)[0] = 0.
    before = deepcopy(current)
    current[name].reshape(-1)[0] = -0. if target == "signed_zero" else current[name].reshape(-1)[0] + 1.
    with pytest.raises(ValueError, match="Frozen parameter or buffer"):
        training.frozen_state_equal(before, model, view)


@pytest.mark.parametrize("mutation", ["hidden_trainable", "replace_third", "replace_penultimate", "replace_final", "prefixes", "indices"])
def test_layout_rejects_hidden_training_and_stale_coordinates(mutation):
    model, _, _ = fixture()
    view = training.TopThreeBlockParameterView(model)
    if mutation == "hidden_trainable":
        next(model.feature_encoder.parameters()).requires_grad_(True)
    elif mutation.startswith("replace_"):
        index = {"replace_third": -3, "replace_penultimate": -2, "replace_final": -1}[mutation]
        model.transformer_blocks[index] = deepcopy(model.transformer_blocks[index])
    elif mutation == "prefixes":
        view.prefixes = ("decoder.",)
    else:
        view.block_indices = (1, 2)
    with pytest.raises(ValueError, match="Only"):
        training.check_layout(model, view)


@pytest.mark.parametrize("mutation", ["three_blocks", "no_decoder", "wrong_container", "aliased_blocks"])
def test_invalid_architecture_is_rejected_before_gradient_flags_change(mutation):
    model, _, _ = fixture()
    if mutation == "three_blocks":
        model.transformer_blocks = torch.nn.ModuleList(list(model.transformer_blocks)[:3])
    elif mutation == "no_decoder":
        model.decoder = None
    elif mutation == "wrong_container":
        model.transformer_blocks = torch.nn.Sequential(*model.transformer_blocks)
    else:
        model.transformer_blocks[-1] = model.transformer_blocks[-3]
    before = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError):
        training.TopThreeBlockParameterView(model)
    assert before == [parameter.requires_grad for parameter in model.parameters()]


def test_full_original_architecture_has_90_mutable_and_164_frozen_tensors(monkeypatch):
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
    view = training.TopThreeBlockParameterView(model)
    assert view.block_indices == (5, 6, 7)
    assert view.prefixes == ("transformer_blocks.5.", "transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
    assert tuple(model.state_dict()) == names and len(view.mutable_names) == 90
    assert sum(parameter.numel() for parameter in model.parameters()) == 77998496
    assert sum(parameter.numel() for parameter in view.parameters()) == 36036497
    assert sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad) == 41961999
    assert len(set(model.state_dict()) - set(view.mutable_names)) == 164
    buffers = [name for name, buffer in model.named_buffers() if name.startswith(view.prefixes)]
    assert len(buffers) == 6 and not set(buffers).intersection(view.mutable_names)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
