"""Check whether predictive means are degraded by broad outer distribution bins."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diagnose_science_preprocessing as diagnosis
import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT


OUTPUT = ROOT / 'decoding_diagnosis_v1'


@torch.inference_mode()
def predictions(matrix, model, distribution):
    normalized, inverse = original.transform(matrix, 'column')
    values, center, scale = shared.normalize_observed(normalized)
    tensor = torch.as_tensor(values, dtype=torch.float32, device='cuda')[None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        embedded = model(tensor, return_embeddings=True)[0]
    features = embedded.reshape(-1, embedded.shape[-1])
    result = {name: np.empty(matrix.size, dtype=float) for name in ('mean', 'median', 'interior_mean')}
    for start in range(0, matrix.size, 2048):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.decoder(features[start:start + 2048])
        logits = logits.float()
        stop = start + len(logits)
        result['mean'][start:stop] = distribution.mean(logits).cpu().numpy()
        result['median'][start:stop] = distribution.median(logits).cpu().numpy()
        interior_probabilities = logits[:, 1:-1].softmax(-1)
        centers = distribution._full_bucket_means[1:-1]
        result['interior_mean'][start:stop] = (interior_probabilities @ centers).cpu().numpy()
    for name, values in result.items():
        restored = inverse(values.reshape(matrix.shape) * scale + center)
        restored[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
        result[name] = restored
    return result


def run(output):
    parent = json.loads((diagnosis.OUTPUT / 'protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output / 'protocol.json', dict(created=datetime.now(timezone.utc).isoformat(),
                     parent=str(diagnosis.OUTPUT), parent_sha256=shared.sha256(diagnosis.OUTPUT / 'protocol.json'),
                     source_sha256=shared.sha256(__file__), variants=['mean', 'median', 'interior_mean'],
                     task='Frozen retrospective decoder ablation, fixed observed-column normalization; no test-driven estimator selection'))
    models, distribution = shared.load_models({name: Path(record['path']) for name, record in parent['checkpoints'].items()})
    rows = []
    for case in parent['cases']:
        arrays = np.load(Path(case['parent']) / f'{case["identifier"]}.npz')
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        for name, model in models.items():
            validation_predictions = predictions(reduced, model, distribution)
            test_predictions = predictions(context, model, distribution)
            destination = output / f'{case["identifier"]}_{name}.npz'
            np.savez_compressed(destination, **test_predictions,
                                **{'validation_' + key: value for key, value in validation_predictions.items()})
            for variant, prediction in test_predictions.items():
                scores = original.metrics(truth, prediction, missing, context, case['targets'])
                validation_scores = original.metrics(truth, validation_predictions[variant], validation, reduced, case['targets'])
                rows.append(dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'], model=name,
                                 variant=variant, nrmse=scores['nrmse'], validation_nrmse=validation_scores['nrmse']))
            print(case['identifier'], name, [(row['variant'], round(row['nrmse'], 5)) for row in rows[-3:]], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_case.csv', index=False)
    frame.groupby(['dataset', 'split', 'model', 'variant']).nrmse.agg(['mean', 'std', 'count']).to_csv(output / 'summary.csv')
    shared.write_json(output / 'completion.json', dict(records=len(frame), complete=len(frame)==len(parent['cases'])*6,
                      predictions_sha256={path.name: shared.sha256(path) for path in output.glob('*.npz')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    run(arguments.output)
