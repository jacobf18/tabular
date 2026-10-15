import pytest
import torch
from pathlib import Path

from rowcol_fixed_validation import apply_support_mask
from spectral_support_normalization import normalize_remaining_support, apply_shrunk_support_mask


def test_remaining_observations_only_and_affine_roundtrip():
    torch.manual_seed(3)
    source = torch.randn(2, 20, 8)
    target = source.clone()
    keep = torch.rand_like(source) > .4
    x, y, mean, scale = normalize_remaining_support(source, target, keep)
    changed = source.masked_fill(~keep, 1e8)
    x2, _, mean2, scale2 = normalize_remaining_support(changed, target * 200, keep)
    torch.testing.assert_close(x, x2, equal_nan=True)
    torch.testing.assert_close(mean, mean2)
    torch.testing.assert_close(scale, scale2)
    torch.testing.assert_close(y * scale + mean, target)
    assert torch.isnan(x)[~keep].all()


def test_sparse_near_constant_column_and_empty_support():
    source = torch.tensor([[[1., -1.], [1.00001, 1.], [3., -2.], [-2., 2.]]])
    keep = torch.ones_like(source, dtype=torch.bool)
    keep[:, 2:, 0] = False
    _, y, _, scale = normalize_remaining_support(source, source, keep)
    assert scale[0, 0, 0] > .5
    assert y.abs().max() < 10
    x, y, _, scale = normalize_remaining_support(source, source, torch.zeros_like(keep))
    assert torch.isnan(x).all() and torch.isfinite(y).all() and (scale > 0).all()


def test_same_masks_and_unchanged_non_augmented_episodes():
    source = torch.randn(2, 20, 12)
    source[:, ::2, ::3] = float("nan")
    target = source.nan_to_num()
    for mode in ("mixed", "broad_support", "sparse_query"):
        a, b, mask, stats = apply_support_mask(source, target, mode, 172)
        x, y, new_mask, new_stats = apply_shrunk_support_mask(source, target, mode, 172)
        assert torch.equal(mask, new_mask)
        assert torch.equal(torch.isnan(a), torch.isnan(x))
        if not stats["support_episode"]:
            torch.testing.assert_close(a, x, equal_nan=True)
            torch.testing.assert_close(b, y)
        assert torch.isfinite(y).all()


def test_axis_permutations_and_invalid_df():
    source = torch.randn(2, 12, 8)
    keep = torch.rand_like(source) > .4
    rows, cols = torch.randperm(12), torch.randperm(8)
    x, y, _, _ = normalize_remaining_support(source, source, keep)
    xp, yp, _, _ = normalize_remaining_support(source[:, rows][:, :, cols],
        source[:, rows][:, :, cols], keep[:, rows][:, :, cols])
    torch.testing.assert_close(x[:, rows][:, :, cols], xp, equal_nan=True)
    torch.testing.assert_close(y[:, rows][:, :, cols], yp)
    for df in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalize_remaining_support(source, source, keep, df)


def test_paired_queue_preserves_control_configuration():
    from run_spectral_query_experiments import build_commands as baseline
    from run_spectral_shrinkage_experiments import build_commands as followup
    root = Path("/tmp/shrinkage_test_queue")
    _, control, prior, anchor = baseline(root)
    trials, commands, new_prior, new_anchor = followup(root)
    assert new_prior == prior and new_anchor == anchor
    assert len(trials) == 2
    for command in commands.values():
        for flag in ("--lr", "--phases-json", "--model-overrides-json", "--prior-overrides-json",
                     "--data-seed", "--init-model-checkpoint", "--fixed-validation-cache"):
            assert command[command.index(flag) + 1] == control["joint_low"][control["joint_low"].index(flag) + 1]


def test_combination_changes_only_lr_and_run_name():
    from run_spectral_shrinkage_experiments import build_commands as baseline
    from run_spectral_combination_experiments import build_commands as combined
    root = Path("/tmp/spectral_combination_test")
    _, originals, prior, anchor = baseline(root)
    trials, commands, new_prior, new_anchor = combined(root)
    assert prior == new_prior and anchor == new_anchor
    for trial in trials:
        command = list(commands[trial["tag"]])
        assert float(command[command.index("--lr") + 1]) == trial["lr"]
        for flag in ("--lr", "--run-name"):
            command[command.index(flag) + 1] = originals["shrinkage_df1"][originals["shrinkage_df1"].index(flag) + 1]
        assert command == originals["shrinkage_df1"]
