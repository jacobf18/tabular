from copy import deepcopy
import inspect
from types import SimpleNamespace
import types

import numpy as np
import pytest

import relocate_single_model_early_candidate_predictions as preservation
import test_single_model_legacy_root_banks as predecessor_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize('fault', [None, 'missing', 'extra', 'shape', 'precision', 'nonfinite', 'count_shape', 'count_dtype', 'zero_count', 'queries'])
def test_exact_typed_prediction_and_overlap_arrays(fault):
    arrays = {'prediction': np.array([1.25, 3.75, 2.5], dtype=np.float64), 'counts': np.array([1, 2, 4], dtype=np.uint8)}
    queries = 3
    if fault == 'missing':
        arrays.pop('counts')
    elif fault == 'extra':
        arrays['target'] = arrays['prediction'].copy()
    elif fault == 'shape':
        arrays['prediction'] = arrays['prediction'][:2]
    elif fault == 'precision':
        arrays['prediction'] = arrays['prediction'].astype(np.float32)
    elif fault == 'nonfinite':
        arrays['prediction'][1] = np.nan
    elif fault == 'count_shape':
        arrays['counts'] = arrays['counts'][:2]
    elif fault == 'count_dtype':
        arrays['counts'] = arrays['counts'].astype(np.int64)
    elif fault == 'zero_count':
        arrays['counts'][1] = 0
    elif fault == 'queries':
        queries = 0
    if fault:
        with pytest.raises(ValueError):
            preservation.validate_arrays(arrays, queries)
    else:
        before = deepcopy(arrays)
        preservation.validate_arrays(arrays, queries)
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], before[name])


@pytest.mark.parametrize('fault', [None, 'missing', 'duplicate', 'source', 'destination', 'sha256', 'bytes'])
def test_exact_complete_eight_file_scope(fault):
    records = [{'name': name, 'source': str(preservation.SOURCE / name), 'destination': str(preservation.DESTINATION / name),
        'sha256': digest, 'bytes': size} for name, (digest, size) in preservation.TARGETS.items()]
    if fault == 'missing':
        records.pop()
    elif fault == 'duplicate':
        records[-1] = records[0]
    elif fault:
        records[0][fault] = 0 if fault == 'bytes' else 'changed'
    if fault:
        with pytest.raises(ValueError):
            preservation.validate_records(records)
    else:
        preservation.validate_records(records)
        assert sum(row['bytes'] for row in records) == 35079020
        assert 2 * sum(preservation.QUERIES.values()) == 4682850
        assert all(name.endswith('/student.npz') for name in preservation.TARGETS)


for name in ('test_copy_checks_before_publishing_and_keeps_source_bytes',
        'test_wrong_copy_size_cannot_reach_typed_validation', 'test_all_original_reserves_and_root_headroom_are_mandatory'):
    function = getattr(predecessor_tests, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_callbacks_are_complete_and_protect_both_live_source_sets():
    assert len(preservation.LIVE_CONFIGS) == 2
    assert len(preservation.OWNERS) == 2
    assert 'protected_training_source_sha256' in preservation.RECHECK_SOURCE
    assert 'validate_records' in preservation.RECHECK_SOURCE
    for function in vars(preservation).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(preservation):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize('fault', [None, 'wrong_hash', 'wrong_size', 'unknown', 'damaged_copy'])
def test_real_npz_copy_is_verified_before_atomic_publication(tmp_path, monkeypatch, fault):
    source = tmp_path / 'student.npz'
    arrays = {'prediction': np.array([1.25, 3.75, 2.5], dtype=np.float64), 'counts': np.array([1, 2, 4], dtype=np.uint8)}
    np.savez_compressed(source, **arrays)
    name = 'owner/recsys/10M/student.npz'
    expected, size = preservation.sha256(source), source.stat().st_size
    monkeypatch.setattr(preservation, 'TARGETS', {name: (expected, size)})
    monkeypatch.setattr(preservation, 'QUERIES', {'10M': 3})
    if fault in ('wrong_hash', 'wrong_size', 'unknown'):
        if fault == 'wrong_hash':
            preservation.TARGETS[name] = ('0' * 64, size)
        elif fault == 'wrong_size':
            preservation.TARGETS[name] = (expected, size + 1)
        else:
            name = 'owner/recsys/10M/parent_same_recipe.npz'
        with pytest.raises(ValueError):
            preservation.check_file(source, name)
        return
    destination = tmp_path / 'preserved/student.npz'
    row = {'name': name, 'source': str(source), 'destination': str(destination), 'sha256': expected, 'bytes': size}
    preservation.copy_and_verify(row)
    assert not source.is_symlink() and source.read_bytes() == destination.read_bytes()
    if fault == 'damaged_copy':
        destination.write_bytes(b'not the original bytes')
        with pytest.raises(ValueError):
            preservation.copying.publish(row)
        assert not source.is_symlink() and preservation.sha256(source) == expected
        return
    preservation.copying.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    preservation.check_file(source, name)
    with np.load(source, allow_pickle=False) as copied:
        for key in arrays:
            np.testing.assert_array_equal(copied[key], arrays[key])
