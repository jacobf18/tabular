import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import relocate_single_model_v53_root_archive as preservation


def original_audit():
    unit = 'original.mount'
    return {'complete': True, 'all_original_hashes_verified': True, 'torch_mmap_readable': True,
        'persistent_mount_enabled': True, 'image': str(preservation.ARCHIVE / 'gradients.squashfs'),
        'image_bytes': preservation.IMAGE_BYTES, 'image_sha256': preservation.IMAGE_SHA,
        'unit_sha256': preservation.UNIT_SHA, 'unit': '/etc/systemd/system/' + unit,
        'records': [{'source': str(preservation.OWNER / 'raw_final_update.pt'), 'sha256': preservation.RAW_SHA}]}


def test_exact_original_inventory(monkeypatch):
    monkeypatch.setattr(preservation.subprocess, 'check_output', lambda *args, **kwargs: 'original.mount\n')
    preservation.validate_original(original_audit())


@pytest.mark.parametrize('field,value', [
    ('complete', False), ('all_original_hashes_verified', False), ('torch_mmap_readable', False),
    ('persistent_mount_enabled', False), ('image', '/unrelated/image'), ('image_bytes', 1),
    ('image_sha256', 'other'), ('unit_sha256', 'other'), ('records', []),
])
def test_changed_inventory_rejected(field, value):
    audit = original_audit()
    audit[field] = value
    with pytest.raises(ValueError, match='original V13'):
        preservation.validate_original(audit)


def test_different_unit_rejected(monkeypatch):
    monkeypatch.setattr(preservation.subprocess, 'check_output', lambda *args, **kwargs: 'other.mount\n')
    with pytest.raises(ValueError, match='exact existing'):
        preservation.validate_original(original_audit())


@pytest.mark.parametrize('visible,initialized', [('0', False), (None, False), ('', True)])
def test_gpu_context_rejected(monkeypatch, visible, initialized):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(preservation.torch.cuda, 'is_initialized', lambda: initialized)
    with pytest.raises(ValueError, match='hidden, uninitialized'):
        preservation.require_cpu()


def test_hidden_cpu_accepted(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(preservation.torch.cuda, 'is_initialized', lambda: False)
    preservation.require_cpu()


def test_existing_preservation_not_restarted(monkeypatch, tmp_path):
    monkeypatch.setattr(preservation, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError, match='Never overwrite or restart'):
        preservation.preserve()


@pytest.mark.parametrize('short_volume', ['checkpoint', 'scratch', 'root'])
def test_final_reserves_unchanged(monkeypatch, short_volume):
    free = {str(preservation.ROOT): 18 * 1024**3, str(preservation.scratch.SCRATCH): 50 * 1024**3,
        '/': 6 * 1024**3 + 16 * 1024**2}
    path = {'checkpoint': str(preservation.ROOT), 'scratch': str(preservation.scratch.SCRATCH), 'root': '/'}[short_volume]
    free[path] -= 1
    monkeypatch.setattr(preservation.shutil, 'disk_usage', lambda path: SimpleNamespace(free=free[str(path)]))
    with pytest.raises(OSError):
        preservation.capacity(restored=True)


def test_recovery_can_restore_initial_root_shortfall_without_relaxing_final_gate(monkeypatch):
    monkeypatch.setattr(preservation.shutil, 'disk_usage',
        lambda path: SimpleNamespace(free=1 if str(path) == '/' else 100 * 1024**3))
    preservation.capacity(copying=True)
    with pytest.raises(OSError, match='unchanged root reserve'):
        preservation.capacity(restored=True)


def test_scratch_copy_allowance_includes_original_image(monkeypatch):
    monkeypatch.setattr(preservation.shutil, 'disk_usage', lambda path: SimpleNamespace(
        free=50 * 1024**3 + preservation.IMAGE_BYTES - 1 if Path(path) == preservation.scratch.SCRATCH else 100 * 1024**3))
    with pytest.raises(OSError, match='checkpoint and scratch'):
        preservation.capacity(copying=True)


@pytest.mark.parametrize('state,error,pid', [
    ('complete', '6442450944 bytes on root before confirm', 99999999),
    ('failed', 'unrelated failure', 99999999),
    ('failed', '6442450944 bytes on root before confirm', 1),
    ('failed', '6442450944 bytes on root before confirm', None),
])
def test_only_original_stopped_preconfirm_failure_allowed(monkeypatch, tmp_path, state, error, pid):
    monkeypatch.setattr(preservation, 'ROOT', tmp_path)
    monkeypatch.setattr(preservation.mounts, 'terminal', lambda path: 'unchanged')
    folder = tmp_path / 'balanced_structured_components_v53_pipeline'
    folder.mkdir()
    (folder / 'status.json').write_text(json.dumps({'state': state, 'error': error, 'pid': pid}))
    with pytest.raises(ValueError, match='stopped V53'):
        preservation.require_terminal()


def test_unchanged_preconfirm_failure_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(preservation, 'ROOT', tmp_path)
    monkeypatch.setattr(preservation.mounts, 'terminal', lambda path: 'unchanged')
    folder = tmp_path / 'balanced_structured_components_v53_pipeline'
    folder.mkdir()
    (folder / 'status.json').write_text(json.dumps({'state': 'failed', 'pid': 99999999,
        'error': 'Preserve 6442450944 bytes on root before confirm'}))
    preservation.require_terminal()
