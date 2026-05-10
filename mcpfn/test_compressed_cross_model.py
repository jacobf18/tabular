import json

import numpy as np
import torch

from tabimpute.interface import CompressedCrossImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
)
from tabimpute.model.model_new_stable_memory_efficient import TabImputeModel


def tiny_predictor_config(num_outputs: int = 17) -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 2,
        "num_outputs": num_outputs,
        "num_cls": 2,
        "use_rope": True,
        "rope_base": 10000.0,
        "rope_fraction": 0.5,
        "use_absolute_positional_embeddings": False,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "residual_scale_init": 1.0,
        "rms_norm_eps": 1e-6,
        "embedding_dropout": 0.0,
        "use_activation_checkpointing": False,
    }


def tiny_compressor_kwargs(k_row: int = 2, k_col: int = 3) -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 1,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "residual_scale_init": 1.0,
        "rms_norm_eps": 1e-6,
        "use_rope": True,
        "rope_base": 10000.0,
        "rope_fraction": 0.5,
        "use_activation_checkpointing": False,
        "k_row": k_row,
        "k_col": k_col,
    }


def build_tiny_compressed_model(num_outputs: int = 17) -> CompressedCrossTabImputeModel:
    predictor = TabImputeModel(**tiny_predictor_config(num_outputs=num_outputs))
    cfg = tiny_compressor_kwargs()
    row_compressor = RowCompressionModel(
        **{key: value for key, value in cfg.items() if key != "k_col"}
    )
    column_compressor = ColumnCompressionModel(
        **{key: value for key, value in cfg.items() if key != "k_row"}
    )
    return CompressedCrossTabImputeModel(
        predictor=predictor,
        row_compressor=row_compressor,
        column_compressor=column_compressor,
        k_row=cfg["k_row"],
        k_col=cfg["k_col"],
    )


def test_compressors_return_latent_rows_and_columns():
    cfg = tiny_compressor_kwargs(k_row=2, k_col=3)
    row_compressor = RowCompressionModel(
        **{key: value for key, value in cfg.items() if key != "k_col"}
    )
    column_compressor = ColumnCompressionModel(
        **{key: value for key, value in cfg.items() if key != "k_row"}
    )
    target = torch.randn(2, 4, 5)
    row_source = torch.randn(2, 6, 5)
    col_source = torch.randn(2, 4, 7)
    target[0, 0, 0] = float("nan")

    row_latents = row_compressor(target, row_source)
    col_latents = column_compressor(target, col_source)

    assert row_latents.shape == (2, 2, 5, 32)
    assert col_latents.shape == (2, 4, 3, 32)


def test_forward_target_chunk_logits_shape_with_and_without_context():
    model = build_tiny_compressed_model(num_outputs=17)
    x = torch.randn(2, 6, 6)
    x[:, 0, 0] = float("nan")

    logits = model.forward_target_chunk(
        x,
        chunk_rows=3,
        chunk_cols=3,
        target_row_idx=0,
        target_col_idx=0,
    )
    assert logits.shape == (2, 3, 3, 17)

    single_chunk_logits = model.forward_target_chunk(
        x,
        chunk_rows=6,
        chunk_cols=6,
        target_row_idx=0,
        target_col_idx=0,
    )
    assert single_chunk_logits.shape == (2, 6, 6, 17)


def test_forward_full_matrix_backward_flows_to_all_modules():
    model = build_tiny_compressed_model(num_outputs=17)
    model.train()
    x_target = torch.randn(2, 6, 6)
    x_missing = x_target.clone()
    x_missing[:, ::2, ::2] = float("nan")
    criterion = FullSupportBarDistribution(borders=torch.linspace(-4, 4, 18))

    loss = model.forward_full_matrix(
        x_missing,
        x_target,
        chunk_rows=3,
        chunk_cols=3,
        criterion=criterion,
    )
    loss.backward()

    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.predictor.parameters()
    )
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.row_compressor.parameters()
    )
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.column_compressor.parameters()
    )


def test_compressed_cross_interface_impute_shape(tmp_path):
    config = tiny_predictor_config(num_outputs=5000)
    predictor = TabImputeModel(**config)
    checkpoint_path = tmp_path / "checkpoint_1.pth"
    torch.save({"epoch": 1, "model": predictor.state_dict()}, checkpoint_path)
    with (tmp_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"model": config}, f)

    imputer = CompressedCrossImputePFN(
        predictor_checkpoint_path=str(checkpoint_path),
        predictor_config=config,
        compressor_config={
            **tiny_compressor_kwargs(k_row=1, k_col=1),
            "num_layers": 1,
        },
        chunk_rows=2,
        chunk_cols=2,
        k_row=1,
        k_col=1,
        device="cpu",
        model_family="memory_efficient",
    )
    x = np.random.randn(4, 4).astype(np.float32)
    x[0, 0] = np.nan

    out = imputer.impute(x)

    assert out.shape == x.shape
    assert np.isfinite(out[0, 0])
