import inspect
from pathlib import Path

import pytest

import archive_single_model_compact_qualification_checkpoints as archive


def test_exact_inventory_excludes_active_trial_and_published_baselines():
    pairs = [pair for group in archive.GROUPS for pair in group]
    assert len(pairs) == len(set(pairs)) == 10
    assert len(archive.GROUPS) == 4
    assert {name for name, filename in pairs} == {"v8_decoder_v17", "replicated_decoder_v18", "paired_decoder_v19",
        "structured_teacher_v20", "point_inclusive_v21", "teacher_continuation_v22", "teacher_continuation_v22_cached_recovery_v1"}
    assert {filename for name, filename in pairs} == {"selected_checkpoint.pth", "latest_checkpoint.pth"}
    assert not any("v23" in name or filename.startswith("best_") for name, filename in pairs)
    assert archive.storage.archive.MINIMUM_FREE == 21 * 1024**3


def test_preserves_original_failure_negative_selection_and_exact_hashes():
    source = inspect.getsource(archive.inventory)
    for expression in ('"failed" if name == "teacher_continuation_v22"', 'Path(f\'/proc/{current["pid"]}\').exists()',
        'report["input_sha256"]', 'report["artifact_sha256"]', 'sha256(source) != expected',
        'result["gate"]["passed"] is not False', 'result["campaign"] is not None', 'source.is_symlink()'):
        assert expression in source
    assert "readers exit" in archive.POLICY
    assert "not a promise of immediate free-space" in archive.POLICY


def test_recheck_refuses_omitted_group(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="all four"):
        archive.recheck({"groups": [[]]}, [])


def test_source_uses_existing_lossless_mmap_coarchiver_not_reserialization():
    source = Path(archive.__file__).read_text()
    assert 'storage.compressor()(OUTPUT / f"group_{index:02d}", records)' in source
    assert 'torch.load(source, map_location="cpu", weights_only=False, mmap=True)' in source
    assert "torch.save(" not in source and ".unlink(" not in source
