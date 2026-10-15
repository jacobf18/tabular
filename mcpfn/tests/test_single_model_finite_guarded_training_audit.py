import json

import pytest

from audit_single_model_finite_guarded_training import primitive, read_prefix


def test_prefix_ignores_later_incomplete_append(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step":')
    rows, raw = read_prefix(path, [1, 2])
    assert rows == [{"step": 1}, {"step": 2}]
    assert raw == '{"step": 1}\n{"step": 2}\n'
    with pytest.raises(ValueError, match="complete"):
        read_prefix(path, [1, 2, 3])


def test_prefix_rejects_reordered_or_missing_batches(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('{"step": 2}\n')
    with pytest.raises(ValueError, match="ordered"):
        read_prefix(path, [1])


def test_generation_metadata_normalizes_only_json_container_types():
    expected = {"shape": [2, 3], "domain": "imputation", "seed": 17}
    assert primitive({"shape": (2, 3), "domain": "imputation", "seed": 17}) == expected
    assert json.loads(json.dumps(expected)) == expected
