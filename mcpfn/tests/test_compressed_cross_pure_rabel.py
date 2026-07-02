import torch

from tabimpute.model.model_compressed_cross_pure_rabel import (
    ColumnCompressionModelPureRaBEL,
    CompressedCrossTabImputeModelPureRaBEL,
    RowCompressionModelPureRaBEL,
)
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL


def _predictor_kwargs() -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 2,
        "num_outputs": 17,
        "num_cls": 2,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "embedding_dropout": 0.0,
        "use_activation_checkpointing": False,
        "rabel_num_kernels": 16,
        "rabel_exponent_embedding_size": 8,
    }


def _compressor_kwargs() -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 2,
        "num_cls": 2,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "use_activation_checkpointing": False,
        "rabel_config": {
            "rabel_num_kernels": 16,
            "rabel_exponent_embedding_size": 8,
        },
    }


def _build_model() -> CompressedCrossTabImputeModelPureRaBEL:
    predictor = TabImputeModelPureRaBEL(**_predictor_kwargs())
    row = RowCompressionModelPureRaBEL(
        **_compressor_kwargs(),
        k_row=2,
    )
    column = ColumnCompressionModelPureRaBEL(
        **_compressor_kwargs(),
        k_col=2,
    )
    return CompressedCrossTabImputeModelPureRaBEL(
        predictor=predictor,
        row_compressor=row,
        column_compressor=column,
        k_row=2,
        k_col=2,
    )


def test_pure_rabel_compressors_have_no_legacy_scalar_mlp():
    model = _build_model()

    assert not hasattr(
        model.row_compressor.scalar_embedder,
        "observed_linear_layer",
    )
    assert not hasattr(
        model.column_compressor.scalar_embedder,
        "observed_linear_layer",
    )


def test_pure_rabel_compressed_model_runs_forward_and_backward():
    model = _build_model()
    x = torch.randn(2, 4, 4)
    x[:, 0, 0] = torch.nan
    logits = model.forward_target_chunk(
        x,
        chunk_rows=2,
        chunk_cols=2,
        target_row_idx=0,
        target_col_idx=0,
    )
    logits.square().mean().backward()

    assert logits.shape == (2, 2, 2, 17)
    assert (
        model.predictor.feature_encoder.rabel_encoder.projection.weight.grad
        is not None
    )
    assert (
        model.row_compressor.scalar_embedder.rabel_encoder.projection.weight.grad
        is not None
    )
    assert (
        model.column_compressor.scalar_embedder.rabel_encoder.projection.weight.grad
        is not None
    )


def test_pure_rabel_compressor_initialization_matches_predictor():
    model = _build_model()
    model.initialize_compressor_embedders_from_predictor()
    model.initialize_compressor_transformers_from_predictor()
    predictor_encoder = model.predictor.feature_encoder

    for compressor in (model.row_compressor, model.column_compressor):
        compressor_encoder = compressor.scalar_embedder
        for key, value in predictor_encoder.rabel_encoder.state_dict().items():
            assert torch.equal(
                value,
                compressor_encoder.rabel_encoder.state_dict()[key],
            )
        assert torch.equal(
            predictor_encoder.mask_token,
            compressor_encoder.mask_token,
        )
        assert torch.equal(
            predictor_encoder.row_cls_embedding.weight,
            compressor.row_cls_embedding.weight,
        )
        assert torch.equal(
            predictor_encoder.col_cls_embedding.weight,
            compressor.col_cls_embedding.weight,
        )
        for predictor_block, compressor_block in zip(
            model.predictor.transformer_blocks,
            compressor.transformer.blocks,
        ):
            for key, value in predictor_block.state_dict().items():
                assert torch.equal(value, compressor_block.state_dict()[key])
