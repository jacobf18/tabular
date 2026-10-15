import pytest
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_inference import streamed_point_predictions


def small_model():
    torch.manual_seed(123)
    return TabImputeModelStable(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32,
        num_layers=2, num_outputs=20, num_cls=3, rope_fraction=0.5,
        attention_dropout=0.1, ffn_dropout=0.1, drop_path_rate=0.1,
    ).eval()


@pytest.mark.parametrize("statistic", ["mean", "median"])
@pytest.mark.parametrize("shape", [(1, 9, 7), (2, 5, 11), (1, 1, 1)])
def test_streaming_matches_dense_and_preserves_weights(statistic, shape):
    model = small_model()
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    values = torch.randn(shape)
    values[torch.rand(shape) < 0.5] = torch.nan
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = getattr(distribution, statistic)(model(values))
    actual = streamed_point_predictions(
        model, values, distribution, attention_batch_size=2, cell_batch_size=13,
        encoder_row_batch_size=3, prediction_stat=statistic,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not actual.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])


def test_microbatching_preserves_complete_attention_sequences():
    model = small_model()
    lengths = {"features": [], "datapoints": []}
    handles = []
    for block in model.transformer_blocks:
        for axis, attention in (("features", block.self_attention_between_features),
                                ("datapoints", block.self_attention_between_datapoints)):
            handles.append(attention.register_forward_pre_hook(
                lambda module, args, axis=axis: lengths[axis].append(tuple(args[0].shape[:2]))
            ))
    try:
        streamed_point_predictions(
            model, torch.full((1, 9, 7), torch.nan),
            FullSupportBarDistribution(torch.linspace(-5, 5, 21)),
            attention_batch_size=2, cell_batch_size=13, encoder_row_batch_size=3,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert all(batch <= 2 and sequence == 10 for batch, sequence in lengths["features"])
    assert all(batch <= 2 and sequence == 12 for batch, sequence in lengths["datapoints"])
    assert len(lengths["features"]) > 2 and len(lengths["datapoints"]) > 2


def test_streaming_rejects_training_and_invalid_batches():
    model = small_model()
    values = torch.zeros(1, 2, 2)
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    with pytest.raises(ValueError, match="positive"):
        streamed_point_predictions(model, values, distribution, attention_batch_size=0)
    model.train()
    with pytest.raises(ValueError, match="inference-only"):
        streamed_point_predictions(model, values, distribution)
    model.eval()
    model.set_attention_capture(True)
    with pytest.raises(ValueError, match="capture"):
        streamed_point_predictions(model, values, distribution)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 parity check")
def test_bfloat16_streaming_preserves_strided_projection_rounding():
    model = small_model().cuda()
    values = torch.randn(1, 65, 37, device="cuda")
    values[torch.rand_like(values) < 0.9] = torch.nan
    distribution = FullSupportBarDistribution(torch.linspace(-5, 5, 21, device="cuda"))
    previous_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(values)
        with torch.inference_mode():
            expected = distribution.mean(logits.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = streamed_point_predictions(
                model, values, distribution, attention_batch_size=7,
                cell_batch_size=128, encoder_row_batch_size=9,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-6)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(previous_cudnn)
