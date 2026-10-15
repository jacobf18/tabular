import pytest

import archive_single_model_root_gradients as gradients
import archive_single_model_supplemental_snapshots as snapshots
import relocate_single_model_capacity_pair as pair


def test_supplement_scopes_are_disjoint_and_cancelled_queue_is_excluded():
    assert set(snapshots.NAMES).isdisjoint(snapshots.baseline.NAMES)
    assert snapshots.NAMES == ("spectral_combination_20260913", "cell_query_20260909_checked")
    assert "spectral_optimizer_20260915" not in snapshots.NAMES
    assert snapshots.TARGET_ROOT_FREE == 20 * 1024**3
    assert snapshots.MINIMUM_ROOT_FREE == snapshots.baseline.MINIMUM_ROOT_FREE == 3 * 1024**3
    assert snapshots.writer_arguments(["python", "scripts/run_cell_query_experiments.py"])
    assert snapshots.writer_arguments(["python", "scripts/run_spectral_combination_experiments.py"])


@pytest.mark.parametrize("name", ["best_checkpoint.pth", "best_validation_checkpoint.pth"])
def test_supplement_never_archives_selected_checkpoints(name):
    with pytest.raises(ValueError):
        snapshots.snapshot_epoch(name, 36000)


def test_pair_relocation_preserves_the_full_trial_capacity_floor():
    assert pair.NAMES == pair.baseline.NAMES[:2]
    assert pair.RESERVE == pair.baseline.RESERVE == 6 * 1024**3
    assert 'shutil.disk_usage(ROOT).free + sum(sizes) < 31 * 1024**3' in pair.ADAPTED_SOURCE
    assert 'shutil.disk_usage(ROOT).free < 31 * 1024**3' in pair.ADAPTED_SOURCE
    assert 'systemctl", "restart"' in pair.ADAPTED_SOURCE
    assert 'supplemental_capacity_ready()' in pair.ADAPTED_SOURCE


@pytest.mark.parametrize("module", [gradients, snapshots, pair])
def test_existing_archives_are_never_restarted(tmp_path, monkeypatch, module):
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError, match="Never|never"):
        module.main()


def test_root_gradient_compressor_leaves_checkpoint_volume_untouched():
    compressor = gradients.legacy.compressor()
    assert compressor.__globals__["ROOT"] == gradients.ROOT
    assert compressor.__globals__["MINIMUM_FREE"] == 3 * 1024**3
    assert gradients.DURABLE.is_relative_to(gradients.ROOT)
    assert gradients.SOURCE.is_relative_to(pair.ROOT)


def test_incomplete_supplement_cannot_authorize_relocation(monkeypatch):
    monkeypatch.setattr(pair.supplement, "require_predecessors", lambda: {})
    monkeypatch.setattr(pair.baseline, "terminal", lambda path: None)
    monkeypatch.setattr(pair.baseline, "read", lambda path: {"complete": False})
    with pytest.raises(ValueError, match="Supplement"):
        pair.supplemental_capacity_ready()
