import numpy as np
import json
import pytest

import benchmark_tabimpute_runtime_scaling as benchmark
from benchmark_tabimpute_runtime_scaling import make_input


def test_runtime_inputs_are_matched_deterministic_and_fixed_width():
    for rows in (16, 32, 64, 128, 256, 512, 1024, 2048):
        matrix = make_input(rows, 10, 0.4, 42)
        np.testing.assert_array_equal(matrix, make_input(rows, 10, 0.4, 42))
        assert matrix.shape == (rows, 10)
        assert np.isnan(matrix).sum() == rows * 4
        assert not np.isnan(matrix).all(axis=0).any()


def test_runtime_input_seed_changes_data_and_masks():
    first = make_input(64, 10, 0.4, 42)
    second = make_input(64, 10, 0.4, 43)
    assert not np.array_equal(np.isnan(first), np.isnan(second))
    assert not np.allclose(np.nan_to_num(first), np.nan_to_num(second))


def test_suspended_workload_is_resumed_even_when_measurement_fails(tmp_path, monkeypatch):
    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.state = 'running'

        def create_time(self):
            return float(self.pid)

        def status(self):
            return self.state

        def cmdline(self):
            return ['/usr/bin/python', '/example/compare_recsys_column_context.py']

        def children(self, recursive=True):
            return [processes[102]] if self.pid == 101 else []

        def suspend(self):
            self.state = benchmark.psutil.STATUS_STOPPED
            events.append(('suspend', self.pid))

        def resume(self):
            self.state = 'running'
            events.append(('resume', self.pid))

    processes = {pid: Process(pid) for pid in (101, 102, 103)}
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid=103: processes[pid])
    monkeypatch.setattr(benchmark, 'live_process', lambda identity: processes[identity['pid']])
    monkeypatch.setattr(benchmark, 'gpu_pids', lambda: {102})
    monkeypatch.setattr(benchmark.subprocess, 'Popen', lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark.subprocess, 'check_output', lambda *args, **kwargs: '0\n')
    monkeypatch.setattr(benchmark.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='measurement failure'):
        with benchmark.exclusive_compute(tmp_path, 101):
            assert processes[101].state == benchmark.psutil.STATUS_STOPPED
            assert processes[102].state == benchmark.psutil.STATUS_STOPPED
            raise RuntimeError('measurement failure')
    assert events == [('suspend', 101), ('suspend', 102), ('resume', 102), ('resume', 101)]
    assert json.loads((tmp_path / 'gpu_guard.json').read_text())['state'] == 'released'
