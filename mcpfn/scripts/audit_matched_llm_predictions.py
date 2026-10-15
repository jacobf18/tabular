"""Independently rescore matched prediction files and reconstruct the blend."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    audit = json.loads((output / 'final_audit.json').read_text())
    assert audit['state'] == 'complete' and audit['fold_settings'] == 1800
    paths = sorted((output / 'predictions').rglob('*.json'))
    assert len(paths) == 1800
    maximum_r2, maximum_rmse, maximum_blend = 0., 0., 0.
    scored = 0
    for path in paths:
        record = json.loads(path.read_text())
        prediction_path = path.with_suffix('.npz')
        assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == record['prediction_sha256']
        with np.load(prediction_path) as predictions:
            truth = np.maximum(-10., np.minimum(10., predictions['target_z']))
            denominator = float(np.dot(truth, truth))
            assert denominator > 1e-12
            for variant in ('general', 'specialist', 'observed_blend', 'gaussian'):
                estimate = np.maximum(-10., np.minimum(10., predictions[variant + '_z']))
                residual = truth - estimate
                squared_error = float(np.dot(residual, residual))
                r2 = 1. - squared_error / denominator
                rmse = float(np.sqrt(squared_error / len(truth)))
                stored = record['metrics'][variant]
                np.testing.assert_allclose([r2, rmse, squared_error, denominator],
                                           [stored['r2'], stored['rmse'], stored['ss_res'], stored['ss_tot']],
                                           rtol=1e-12, atol=1e-12)
                assert stored['n_eval'] == len(truth)
                maximum_r2 = max(maximum_r2, abs(r2 - stored['r2']))
                maximum_rmse = max(maximum_rmse, abs(rmse - stored['rmse']))
                scored += 1
            weights = np.array([decision['specialist_weight'] for decision in record['blend_decisions']])
            assert ((weights >= 0.) & (weights <= 1.)).all()
            weights = weights[predictions['query_rows'] // 200]
            blend = (1. - weights) * predictions['general_z'] + weights * predictions['specialist_z']
            np.testing.assert_allclose(blend, predictions['observed_blend_z'], rtol=1e-12, atol=1e-12)
            maximum_blend = max(maximum_blend, float(np.max(np.abs(blend - predictions['observed_blend_z']))))
    result = {'state': 'complete', 'fold_settings': len(paths), 'predictor_scores_verified': scored,
              'blend_panels_reconstructed': len(paths), 'maximum_absolute_r2_difference': maximum_r2,
              'maximum_absolute_rmse_difference': maximum_rmse, 'maximum_absolute_blend_difference': maximum_blend,
              'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'final_audit_sha256': hashlib.sha256((output / 'final_audit.json').read_bytes()).hexdigest()}
    archived_script = output / 'audit_matched_llm_predictions.py'
    if Path(__file__).resolve() != archived_script:
        shutil.copyfile(__file__, archived_script)
    (output / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
