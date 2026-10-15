from copy import deepcopy

import pytest
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_training import streamed_training_loss


def example(shape=(2, 9, 7), dropout=0.0):
    torch.manual_seed(318)
    model = TabImputeModelStable(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32,
        num_layers=2, num_outputs=20, num_cls=3, rope_fraction=0.5,
        attention_dropout=dropout, ffn_dropout=dropout, drop_path_rate=dropout,
        embedding_dropout=dropout,
    ).train()
    targets = torch.randn(shape)
    mask = torch.rand(shape) < 0.6
    mask.flatten()[0] = True
    return model, targets.masked_fill(mask, torch.nan), targets, mask


def assert_gradients_match(reference, actual):
    for (name, expected), (actual_name, parameter) in zip(reference.named_parameters(), actual.named_parameters()):
        assert name == actual_name
        assert expected.grad is not None and parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-4, atol=3e-7, msg=name)


@pytest.mark.parametrize("shape", [(2, 9, 7), (1, 1, 1)])
def test_loss_gradients_and_optimizer_step_match_dense(shape):
    reference, values, targets, mask = example(shape)
    actual = deepcopy(reference)
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    expected_loss = distribution(reference(values, decode_mask=mask), targets[mask]).mean()
    actual_loss = streamed_training_loss(
        actual, values, targets, mask, distribution, attention_batch_size=2,
        cell_batch_size=13, encoder_row_batch_size=3,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
    expected_loss.backward()
    actual_loss.backward()
    assert_gradients_match(reference, actual)
    for model in (reference, actual):
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for expected, parameter in zip(reference.parameters(), actual.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-7)


def test_checkpoint_recomputation_preserves_dropout_rng_and_full_sequences():
    reference, values, targets, mask = example(dropout=0.2)
    actual = deepcopy(reference)
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    lengths = {"features": [], "datapoints": []}
    handles = []
    for block in actual.transformer_blocks:
        for axis, attention in (("features", block.self_attention_between_features),
                                ("datapoints", block.self_attention_between_datapoints)):
            handles.append(attention.register_forward_pre_hook(
                lambda module, args, axis=axis: lengths[axis].append(tuple(args[0].shape[:2]))
            ))
    losses = []
    try:
        for model, use_checkpoint in ((reference, False), (actual, True)):
            torch.manual_seed(41)
            loss = streamed_training_loss(
                model, values, targets, mask, distribution, attention_batch_size=2,
                cell_batch_size=13, encoder_row_batch_size=3, use_checkpoint=use_checkpoint,
            )
            loss.backward()
            losses.append(loss.detach())
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    assert_gradients_match(reference, actual)
    assert lengths["features"] and lengths["datapoints"]
    assert all(batch <= 2 and sequence == 10 for batch, sequence in lengths["features"])
    assert all(batch <= 2 and sequence == 12 for batch, sequence in lengths["datapoints"])


def test_rejects_empty_mask_and_reduced_loss():
    model, values, targets, mask = example()
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    with pytest.raises(ValueError, match="nonempty boolean"):
        streamed_training_loss(model, values, targets, torch.zeros_like(mask), distribution)
    with pytest.raises(ValueError, match="one loss per"):
        streamed_training_loss(model, values, targets, mask,
                               lambda logits, truth: distribution(logits, truth).mean())
