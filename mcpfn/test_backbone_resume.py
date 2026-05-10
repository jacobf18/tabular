import torch

from tabimpute.model.model_new_stable import build_best_transfer_model
from tabimpute.model.rfm_head import RFMHead


import json
import os

CKPT = "/root/tabular/mcpfn/src/tabimpute/workdir/dwip_v3r1_t002_ctl-trial_001/checkpoint_17000.pth"
CONFIG_PATH = os.path.join(os.path.dirname(CKPT), "config.json")

# Architectural keys accepted by TabImputeModelStable.__init__
_MODEL_ARCH_KEYS = {
    "embedding_size", "num_attention_heads", "mlp_hidden_size", "num_layers",
    "num_cls", "use_rope", "rope_base", "rope_fraction",
    "use_absolute_positional_embeddings", "positional_damping_factor",
    "attention_dropout", "ffn_dropout", "drop_path_rate",
    "residual_scale_init", "rms_norm_eps", "embedding_dropout",
}


def load_arch_overrides():
    with open(CONFIG_PATH) as f:
        full = json.load(f)
    return {k: v for k, v in full["model"].items() if k in _MODEL_ARCH_KEYS}


def test_backbone_only_resume_into_headless_model():
    model = build_best_transfer_model(num_outputs=None, **load_arch_overrides())  # decoder skipped
    assert model.decoder is None

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict"))
    sd_filtered = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
    dropped = len(sd) - len(sd_filtered)
    print(f"dropped {dropped} decoder.* keys; kept {len(sd_filtered)} backbone keys")
    assert dropped == 6, f"expected 6 bar-head keys, dropped {dropped}"

    res = model.load_state_dict(sd_filtered, strict=False)
    assert not res.missing_keys, f"missing: {res.missing_keys}"
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys}"
    print(f"loaded backbone OK: {len(sd_filtered)} tensors, no missing / unexpected")


def test_forward_returns_embeddings_shape():
    model = build_best_transfer_model(num_outputs=None, **load_arch_overrides()).to("cuda").to(torch.bfloat16)
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict"))
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
    model.load_state_dict(sd, strict=False)
    model.eval()

    B, R, C = 2, 16, 24
    x = torch.randn(B, R, C, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(B, R, C, device="cuda") < 0.4
    x_masked = x.clone()
    x_masked[mask] = float("nan")

    with torch.no_grad():
        feats = model(x_masked, return_embeddings=True)
    assert feats.shape == (B, R, C, 768), f"got {feats.shape}"
    assert torch.isfinite(feats).all()
    print(f"forward (embeddings): shape={tuple(feats.shape)}  OK")


def test_end_to_end_rfm_loss_runs():
    model = build_best_transfer_model(num_outputs=None, **load_arch_overrides()).to("cuda").to(torch.bfloat16)
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict"))
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
    model.load_state_dict(sd, strict=False)
    head = RFMHead(lambda_init=1e-2).to("cuda")

    B, R, C = 2, 16, 24
    target = torch.randn(B, R, C, device="cuda", dtype=torch.bfloat16)
    mask_missing = torch.rand(B, R, C, device="cuda") < 0.4
    x = target.clone()
    x[mask_missing] = float("nan")
    observed = ~mask_missing

    feats = model(x, return_embeddings=True)
    loss, mae = head(feats, target, observed, num_anchors=128)
    assert torch.isfinite(loss) and torch.isfinite(mae)
    loss.backward()
    grads_finite = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert grads_finite, "non-finite grads in backbone"
    assert head.log_lambda.grad is not None and torch.isfinite(head.log_lambda.grad)
    print(f"end_to_end: loss={loss.item():.4e} mae={mae.item():.4e}  OK")


if __name__ == "__main__":
    test_backbone_only_resume_into_headless_model()
    test_forward_returns_embeddings_shape()
    test_end_to_end_rfm_loss_runs()
    print("\nALL BACKBONE-RESUME + RFM SMOKE TESTS PASSED")
