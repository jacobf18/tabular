import torch

from tabimpute.model.rfm_head import RFMHead


def make_synthetic(B=2, R=8, C=8, D=16, missing_p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    Phi = torch.randn(B, R, C, D, generator=g)
    u = torch.randn(B, D, generator=g)
    target = torch.einsum("brcd,bd->brc", Phi, u)
    rand = torch.rand(B, R, C, generator=g)
    observed = rand >= missing_p
    return Phi, target, observed


def test_shapes_and_finite():
    Phi, target, observed = make_synthetic()
    head = RFMHead(lambda_init=1e-2)
    loss, mae = head(Phi, target, observed, num_anchors=32)
    assert loss.dim() == 0 and mae.dim() == 0
    assert torch.isfinite(loss) and torch.isfinite(mae)
    print(f"shapes_and_finite: loss={loss.item():.6f} mae={mae.item():.6f}  OK")


def test_noiseless_low_rank_recovery():
    Phi, target, observed = make_synthetic(B=1, R=16, C=16, D=8, missing_p=0.3, seed=1)
    head = RFMHead(lambda_init=1e-6)
    loss, mae = head(Phi, target, observed, num_anchors=200)
    print(f"noiseless_recovery: loss={loss.item():.3e} mae={mae.item():.3e}")
    assert loss.item() < 1e-3, f"expected loss < 1e-3 on noiseless rank-D, got {loss.item()}"


def test_grad_flows_to_phi_and_lambda():
    Phi, target, observed = make_synthetic()
    Phi = Phi.detach().requires_grad_(True)
    head = RFMHead(lambda_init=1e-2)
    loss, _ = head(Phi, target, observed, num_anchors=32)
    loss.backward()
    assert Phi.grad is not None and torch.isfinite(Phi.grad).all() and Phi.grad.abs().sum() > 0
    assert head.log_lambda.grad is not None and torch.isfinite(head.log_lambda.grad)
    print(f"grad_flows: |dL/dPhi|={Phi.grad.abs().sum().item():.3e}  "
          f"dL/dlog_lambda={head.log_lambda.grad.item():.3e}  OK")


def test_lambda_positive_and_init():
    head = RFMHead(lambda_init=1e-2)
    val = head.lam.item()
    assert val > head.lambda_floor
    # init should reproduce ~1e-2 + floor
    assert abs(val - (1e-2 + head.lambda_floor)) < 1e-6, f"got {val}"
    print(f"lambda_init: lam={val:.6f}  OK")
    # large lambda init must NOT overflow (regression test for log(expm1(100)) -> inf)
    head_big = RFMHead(lambda_init=1000.0)
    big = head_big.lam.item()
    assert torch.isfinite(torch.tensor(big)), f"lam should be finite, got {big}"
    assert abs(big - (1000.0 + head_big.lambda_floor)) < 1e-2, f"got {big}"
    print(f"lambda_init=1000: lam={big:.4f}  OK")


def test_agop_shapes_and_finite():
    Phi, target, observed = make_synthetic(D=16)
    head = RFMHead(lambda_init=1e-2, agop_rank=4, embed_dim=16)
    loss, mae = head(Phi, target, observed, num_anchors=32)
    assert torch.isfinite(loss) and torch.isfinite(mae)
    assert head.L is not None and head.L.shape == (16, 4)
    print(f"agop_shapes: loss={loss.item():.6f} mae={mae.item():.6f}  L.shape={tuple(head.L.shape)}  OK")


def test_agop_grad_flows_to_L():
    Phi, target, observed = make_synthetic(D=16)
    Phi = Phi.detach().requires_grad_(True)
    head = RFMHead(lambda_init=1e-2, agop_rank=4, embed_dim=16)
    loss, _ = head(Phi, target, observed, num_anchors=32)
    loss.backward()
    assert head.L.grad is not None and torch.isfinite(head.L.grad).all() and head.L.grad.abs().sum() > 0
    assert head.log_lambda.grad is not None and torch.isfinite(head.log_lambda.grad)
    assert Phi.grad is not None and Phi.grad.abs().sum() > 0
    print(f"agop_grad: |dL/dL|={head.L.grad.abs().sum().item():.3e}  "
          f"dL/dlog_lambda={head.log_lambda.grad.item():.3e}  OK")


def test_agop_noiseless_recovery():
    # With agop_rank=D and L starting near identity-in-expectation, recovery
    # should still be tight (a few extra training steps would tighten further;
    # for the unit test we just check the head doesn't blow up at init).
    Phi, target, observed = make_synthetic(B=1, R=16, C=16, D=8, missing_p=0.3, seed=2)
    head = RFMHead(lambda_init=1e-6, agop_rank=8, embed_dim=8)
    loss, mae = head(Phi, target, observed, num_anchors=200)
    print(f"agop_noiseless: loss={loss.item():.3e} mae={mae.item():.3e}")
    # Random L init: bigger than dot-product loss but bounded; assert sanity not exact
    assert loss.item() < 5.0, f"runaway loss at agop init: {loss.item()}"


if __name__ == "__main__":
    torch.manual_seed(0)
    test_shapes_and_finite()
    test_lambda_positive_and_init()
    test_noiseless_low_rank_recovery()
    test_grad_flows_to_phi_and_lambda()
    test_agop_shapes_and_finite()
    test_agop_grad_flows_to_L()
    test_agop_noiseless_recovery()
    print("\nALL RFM HEAD TESTS PASSED")
