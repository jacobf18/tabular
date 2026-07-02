import importlib.util
from pathlib import Path

import torch

from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
)
from tabimpute.model.model_compressed_cross_rabel import (
    ColumnCompressionModelRaBEL,
    CompressedCrossTabImputeModelRaBEL,
    FeatureEncoderRaBEL,
    RaBELValueEncoder,
    RowCompressionModelRaBEL,
    TabImputeModelRaBEL,
)
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model.model_new_stable_rabel import (
    FeatureEncoderPureRaBEL,
    TabImputeModelPureRaBEL,
)


def _predictor_kwargs() -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 1,
        "num_outputs": 17,
        "num_cls": 2,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "embedding_dropout": 0.0,
    }


def _compressor_kwargs() -> dict:
    return {
        "embedding_size": 32,
        "num_attention_heads": 4,
        "mlp_hidden_size": 64,
        "num_layers": 1,
        "num_cls": 2,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "drop_path_rate": 0.0,
        "use_activation_checkpointing": False,
    }


def _build_rabel_model() -> CompressedCrossTabImputeModelRaBEL:
    predictor = TabImputeModelRaBEL(
        **_predictor_kwargs(),
        rabel_num_kernels=16,
        rabel_exponent_embedding_size=8,
    )
    config = {
        "rabel_num_kernels": 16,
        "rabel_exponent_embedding_size": 8,
    }
    row = RowCompressionModelRaBEL(
        **_compressor_kwargs(),
        k_row=2,
        rabel_config=config,
    )
    column = ColumnCompressionModelRaBEL(
        **_compressor_kwargs(),
        k_col=2,
        rabel_config=config,
    )
    return CompressedCrossTabImputeModelRaBEL(
        predictor=predictor,
        row_compressor=row,
        column_compressor=column,
        k_row=2,
        k_col=2,
    )


def _build_stable_model() -> CompressedCrossTabImputeModel:
    predictor = TabImputeModelStable(**_predictor_kwargs())
    row = RowCompressionModel(**_compressor_kwargs(), k_row=2)
    column = ColumnCompressionModel(**_compressor_kwargs(), k_col=2)
    return CompressedCrossTabImputeModel(
        predictor=predictor,
        row_compressor=row,
        column_compressor=column,
        k_row=2,
        k_col=2,
    )


def _load_training_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_compressed_cross_rank10.py"
    )
    spec = importlib.util.spec_from_file_location("compressed_trainer_rabel", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rabel_encoder_has_diverse_finite_scalar_embeddings():
    encoder = RaBELValueEncoder(
        embedding_size=32,
        num_kernels=16,
        exponent_embedding_size=8,
    )
    values = torch.linspace(-9.0, 9.0, 81).reshape(1, 81, 1)
    embeddings = encoder(values).squeeze(0)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    assert embeddings.shape == (81, 32)
    assert torch.isfinite(embeddings).all()
    assert torch.linalg.matrix_rank(centered, tol=1e-4) >= 8


def test_rabel_feature_encoder_handles_missing_values_and_cls_tokens():
    encoder = FeatureEncoderRaBEL(
        embedding_size=32,
        num_cls=2,
        rabel_num_kernels=16,
        rabel_exponent_embedding_size=8,
    )
    x = torch.randn(2, 3, 4)
    x[0, 1, 2] = torch.nan
    output = encoder(x)

    assert output.shape == (2, 5, 6, 32)
    assert torch.isfinite(output).all()


def test_rabel_compressed_model_runs_forward_and_backward():
    model = _build_rabel_model()
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
    assert model.predictor.feature_encoder.rabel_encoder.projection.weight.grad is not None
    assert (
        model.row_compressor.scalar_embedder.rabel_encoder.projection.weight.grad
        is not None
    )
    assert (
        model.column_compressor.scalar_embedder.rabel_encoder.projection.weight.grad
        is not None
    )


def test_old_compressed_checkpoint_can_extend_only_the_encoder(tmp_path):
    stable_model = _build_stable_model()
    rabel_model = _build_rabel_model()
    checkpoint = tmp_path / "stable_compressed.pth"
    torch.save({"model": stable_model.state_dict()}, checkpoint)

    trainer = _load_training_script()
    trainer.load_compressed_model_checkpoint(
        rabel_model,
        checkpoint,
        device="cpu",
        allow_encoder_extension=True,
    )

    for key, value in stable_model.state_dict().items():
        assert torch.equal(value, rabel_model.state_dict()[key])
    predictor_rabel = rabel_model.predictor.feature_encoder.rabel_encoder.state_dict()
    for compressor in (rabel_model.row_compressor, rabel_model.column_compressor):
        compressor_rabel = compressor.scalar_embedder.rabel_encoder.state_dict()
        assert predictor_rabel.keys() == compressor_rabel.keys()
        for key in predictor_rabel:
            assert torch.equal(predictor_rabel[key], compressor_rabel[key])


def test_rabel_optimizer_group_is_disjoint_and_complete():
    model = _build_rabel_model()
    trainer = _load_training_script()
    optimizer = trainer.build_split_optimizer(
        model,
        optimizer_name="adamw",
        predictor_lr=1e-5,
        compressor_lr=2e-5,
        encoder_lr=1e-4,
        weight_decay=0.03,
        optimizer_betas=(0.9, 0.999),
        optimizer_eps=1e-8,
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    grouped_ids = [
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    ]

    assert set(groups) == {"predictor", "compressors", "rabel_encoder"}
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {
        id(param) for param in model.parameters() if param.requires_grad
    }
    rabel_names = {
        name
        for name, param in model.named_parameters()
        if id(param) in {id(item) for item in groups["rabel_encoder"]["params"]}
    }
    assert rabel_names
    assert all(
        ".rabel_encoder." in name or name.endswith(".rabel_scale")
        for name in rabel_names
    )


def test_pure_rabel_replaces_scalar_mlp_and_backpropagates():
    encoder = FeatureEncoderPureRaBEL(
        embedding_size=32,
        num_cls=2,
        rabel_num_kernels=16,
        rabel_exponent_embedding_size=8,
    )
    assert not hasattr(encoder, "observed_linear_layer")

    model = TabImputeModelPureRaBEL(
        **_predictor_kwargs(),
        use_activation_checkpointing=False,
        rabel_num_kernels=16,
        rabel_exponent_embedding_size=8,
    )
    x = torch.randn(2, 4, 5)
    x[:, 0, 0] = torch.nan
    missing = torch.isnan(x)
    logits = model(x, decode_mask=missing)
    logits.square().mean().backward()

    assert logits.shape == (2, 17)
    assert not hasattr(model.feature_encoder, "observed_linear_layer")
    assert model.feature_encoder.rabel_encoder.projection.weight.grad is not None
