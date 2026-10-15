import json
from pathlib import Path

import torch

from run_dwip_spectral_geometry import build_commands, dwip_spec
from run_spectral_geometry_experiments import build_commands as geometry_commands
from train_dwip_spectral_geometry import DwipTrainingAdapter
from tabimpute.model.model_new_stable import TabImputeModelStable


def test_configuration_preserves_dwip_and_latest_prior():
    root = Path("/tmp/dwip_spectral_test")
    _, commands, prior, anchor = build_commands(root)
    _, _, geometry, _ = geometry_commands(root)
    assert prior == geometry
    command = commands["dwip1e5"]
    assert json.loads(command[command.index("--model-overrides-json") + 1]) == dwip_spec()["model_config"]
    assert str(anchor) == dwip_spec()["checkpoint_path"]
    assert "--allow-partial-init" not in command
    for flag in ("--svt-mixture-depth-distill-weight", "--svt-mixture-cv-regret-weight"):
        assert command[command.index(flag) + 1] == "0"


def test_adapter_exact_output_gradients_and_no_prior_label_leak():
    cfg = dict(embedding_size=32, num_attention_heads=4, mlp_hidden_size=64,
               num_layers=2, num_outputs=10, num_cls=2, attention_dropout=0.,
               ffn_dropout=0., drop_path_rate=0., embedding_dropout=0.)
    old, new = TabImputeModelStable(**cfg), DwipTrainingAdapter(**cfg)
    new.load_state_dict(old.state_dict(), strict=True)
    source = torch.randn(2, 8, 5)
    source[:, 1, 2] = float("nan")
    mask = source.isnan()
    expected = old(source, decode_mask=mask)
    actual = new(source, decode_mask=mask, prior_index=torch.tensor([123, 456]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for a, b in zip(old.parameters(), new.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
