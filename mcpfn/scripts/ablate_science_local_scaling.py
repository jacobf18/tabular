"""Separate row localization from local column scaling and a local-mean null."""

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch

import diagnose_science_preprocessing as diagnosis
import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT
from science_context_preprocessing import temporal_context_predict


@torch.inference_mode()
def run():
    output = ROOT / 'local_scaling_ablation_v1'
    output.mkdir(exist_ok=False)
    parent = json.loads((diagnosis.OUTPUT / 'protocol.json').read_text())
    cases = [case for case in parent['cases'] if case['dataset'] == 'intel_temperature']
    shared.write_json(output / 'protocol.json', dict(created=datetime.now(timezone.utc).isoformat(), cases=cases,
                     source_sha256=shared.sha256(__file__), rows=32, delays=[],
                     scope='Fixed paired retrospective mechanism probe; no model or condition selected using test errors'))
    models, distribution = shared.load_models({name: diagnosis.Path(record['path']) for name, record in parent['checkpoints'].items()})
    rows = []
    for case in cases:
        arrays = np.load(diagnosis.Path(case['parent']) / f'{case["identifier"]}.npz')
        context = np.where(arrays['missing'], np.nan, arrays['truth'])
        normalized, inverse = original.transform(context, 'column')
        for name, model in models.items():
            global_scaled = np.empty_like(normalized)
            mean_scaled = np.empty_like(normalized)
            for start in range(0, len(context), 16):
                stop = min(start + 16, len(context))
                center = (start + stop) // 2
                left = max(0, min(center - 16, len(context) - 32))
                right = min(left + 32, len(context))
                predicted, _ = diagnosis.forward(normalized[left:right], model, distribution)
                global_scaled[start:stop] = predicted[start-left:stop-left]
                local, inverse_local = original.transform(normalized[left:right], 'column')
                mean_scaled[start:stop] = inverse_local(np.zeros_like(local))[start-left:stop-left]
            predictions = dict(local_rows_global_column_scale=inverse(global_scaled), local_mean=inverse(mean_scaled),
                               local_rows_local_column_scale=temporal_context_predict(context, case['targets'], model,
                                               distribution, diagnosis.forward, 32, []))
            for mode, prediction in predictions.items():
                prediction[np.isfinite(context)] = context[np.isfinite(context)]
                np.testing.assert_array_equal(prediction[np.isfinite(context)], context[np.isfinite(context)])
                score = original.metrics(arrays['truth'], prediction, arrays['missing'], context, case['targets'])
                np.savez_compressed(output / f'{case["identifier"]}_{name}_{mode}.npz', prediction=prediction)
                rows.append(dict(identifier=case['identifier'], split=case['split'], model=name, mode=mode, nrmse=score['nrmse']))
            print(case['identifier'], name, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_case.csv', index=False)
    frame.groupby(['split', 'model', 'mode']).nrmse.agg(['mean', 'std', 'count']).to_csv(output / 'summary.csv')


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
