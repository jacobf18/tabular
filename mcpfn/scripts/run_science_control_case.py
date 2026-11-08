"""Run one frozen calibration-control case and atomically publish its records."""

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

import control_realistic_calibration as runner
import control_realistic_linear as linear
import evaluate_cylinder_completion as shared


def run(root, identifier, kind):
    protocol = json.loads((root / 'protocol.json').read_text())
    case = next(case for case in protocol['cases'] if case['identifier'] == identifier)
    worker = root / 'workers' / identifier
    worker.mkdir(parents=True, exist_ok=True)
    with (worker / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        filenames = [f'{identifier}_{method.replace(" ", "_")}.json' for method in protocol['methods']]
        if all((root / 'records' / name).exists() for name in filenames):
            return
        protocol['cases'] = [case]
        protocol['source_sha256'][str(Path(__file__))] = shared.sha256(__file__)
        shared.write_json(worker / 'protocol.json', protocol)
        if kind == 'linear':
            runner.predict = linear.predict
        runner.run(worker, case['split'], [case['dataset']])
        (root / 'records').mkdir(exist_ok=True)
        for filename in filenames:
            source = worker / 'records' / filename
            destination = root / 'records' / filename
            if destination.exists():
                first = np.load(source.with_suffix('.npz'))['prediction']
                second = np.load(destination.with_suffix('.npz'))['prediction']
                np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
                continue
            for suffix in ('.npz', '.json'):
                target = destination.with_suffix(suffix)
                if not target.exists():
                    os.link(source.with_suffix(suffix), target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--kind', choices=['linear', 'calibration'], required=True)
    arguments = parser.parse_args()
    run(arguments.output, arguments.case, arguments.kind)
