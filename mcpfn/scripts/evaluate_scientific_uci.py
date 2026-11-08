"""Full, frozen UCI comparison of a scientific checkpoint with original MatrixFormer."""

import argparse
from collections import Counter
import fcntl
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json


ROOT = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261108')
DATASETS = ('airfoil_self_noise', 'blood_transfusion', 'california_housing',
            'concrete_compression', 'diabetes', 'ionosphere', 'iris',
            'letter_recognition', 'libras_movement', 'spam_base',
            'wine_quality_red', 'wine_quality_white')
PATTERNS = ('MCAR', 'MAR', 'MNAR')
LABELS = dict(MCAR='MCAR', MAR='Col-MAR', MNAR='Self-MNAR')


def read(path):
    return json.loads(Path(path).read_text())


def select_uci_cases(protocol):
    cases = [case for case in protocol['stages']['tabular'] if case['suite'] == 'uci']
    actual = Counter((case['dataset'], case['pattern'], case['missingness'], case['repeat']) for case in cases)
    expected = Counter(itertools.product(DATASETS, PATTERNS, (.1, .2, .3, .4, .5), range(10)))
    if actual != expected or len({case['case_id'] for case in cases}) != 1800:
        raise ValueError('Require all 1,800 unique frozen UCI cases')
    return cases


def ci_half_width(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    return float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def aggregate(rows, dimensions=()):
    frame = pd.DataFrame(rows)
    if frame.empty or frame.case_id.duplicated().any():
        raise ValueError('Missing or duplicate results')
    if not np.isfinite(frame[['original_nrmse', 'candidate_nrmse']].to_numpy()).all():
        raise ValueError('Nonfinite results')
    groups = frame.groupby(list(dimensions)) if dimensions else [((), frame)]
    result = []
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        means = group.groupby('dataset')[['original_nrmse', 'candidate_nrmse']].mean()
        original, candidate = means.mean()
        difference = means.candidate_nrmse - means.original_nrmse
        result.append(dict(zip(dimensions, keys), original_nrmse=float(original), candidate_nrmse=float(candidate),
                           original_ci95=ci_half_width(means.original_nrmse), candidate_ci95=ci_half_width(means.candidate_nrmse),
                           delta_nrmse=float(difference.mean()), delta_ci95=ci_half_width(difference),
                           change_percent=float(100 * (candidate / original - 1)),
                           improved_datasets=int((difference < 0).sum()), datasets=len(means), cases=len(group)))
    return result


def write_summaries(output, rows):
    write_csv(output / 'per_case.csv', rows)
    summaries = {}
    for name, dimensions in [('overall', ()), ('per_pattern', ('pattern',)), ('per_dataset', ('dataset',)),
                             ('per_missingness', ('missingness',)), ('per_pattern_rate', ('pattern', 'missingness'))]:
        summaries[name] = aggregate(rows, dimensions)
        write_csv(output / f'{name}.csv', summaries[name])
    lines = ['# Full UCI: scientific continuation, step 3000', '',
             'All 1,800 cases: 12 datasets × 3 patterns × 5 missingness rates (10–50%) × 10 repeats.',
             'Same frozen matrices and masks as the original combined MatrixFormer evaluation.',
             'Observed-column preprocessing; native orientation; predictive mean; BF16 autocast;',
             'maximum 2,000 × 1,000 context with overlapping tail windows; no row train/test split.', '',
             'Historical NRMSE averages column RMSE divided by the SD of held-out true values,',
             'using the unchanged original constant-column rules. This scoring normalization',
             'is distinct from observed-only input preprocessing.',
             'Cases are averaged within each dataset, then datasets receive equal weight.',
             '95% t confidence intervals use the 12 dataset means, not the 1,800 cases.',
             'Paired difference intervals likewise use dataset-level differences.',
             'These previously inspected datasets are development benchmarks, not untouched tests.', '',
             '| Pattern | Original NRMSE ± 95% CI | Candidate NRMSE ± 95% CI | Change |',
             '|---|---:|---:|---:|']
    for row in [dict(summaries['overall'][0], pattern='Overall')] + summaries['per_pattern']:
        lines.append(f"| {LABELS.get(row['pattern'], row['pattern'])} | "
                     f"{row['original_nrmse']:.4f} ± {row['original_ci95']:.4f} | "
                     f"{row['candidate_nrmse']:.4f} ± {row['candidate_ci95']:.4f} | {row['change_percent']:+.2f}% |")
    lines += ['', '| Dataset | Original NRMSE | Candidate NRMSE | Change |', '|---|---:|---:|---:|']
    for row in summaries['per_dataset']:
        lines.append(f"| {row['dataset']} | {row['original_nrmse']:.4f} | "
                     f"{row['candidate_nrmse']:.4f} | {row['change_percent']:+.2f}% |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summaries


def evaluate(arguments):
    import torch
    import evaluate_single_matrixformer_tabular as tabular

    output = arguments.output.resolve()
    checkpoint = arguments.checkpoint.resolve()
    protocol_path = arguments.protocol.resolve() / 'protocol.json'
    protocol = read(protocol_path)
    cases = select_uci_cases(protocol)
    config_path = checkpoint.with_name('config.json')
    sources = dict(read(config_path)['source_sha256'], **protocol['source_sha256'])
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    tabular.verify_hashes(sources)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    config_sha256=sha256(config_path), protocol=str(protocol_path), protocol_sha256=sha256(protocol_path),
                    source_sha256=sources, case_ids=[case['case_id'] for case in cases],
                    reference='Original combined MatrixFormer, archived predictions independently rescored',
                    expected_cases=1800, training_performed=False, orientation='native', context=[2000, 1000],
                    precision='amp_bfloat16', seed=42, cudnn_sdp=False, matmul_tf32=False,
                    aggregation='Cases within datasets, then equal dataset weights; 95% t CI across dataset means')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if read(manifest_path) != manifest:
            raise ValueError('Refusing to resume a different checkpoint, protocol or source version')
    else:
        write_json(manifest_path, manifest)
        for source in sources:
            destination = output / 'sources' / Path(source).relative_to('/root/tabular')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    fingerprint = sha256(manifest_path)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45)
    expert = tabular.TabularExpert(tabular.make_model(checkpoint))
    expert.suite = 'uci'
    scorer = tabular.load_original(tabular.SCORERS['uci'], 'compute_normalized_rmse_columnwise')
    rows = []
    started = time.monotonic()
    for case in cases:
        tabular.check_reserve(output)
        missing, truth = tabular.load_case(case)
        tabular.verify_hashes({path: protocol['input_sha256'][path]
                               for path in (case['parent_record'], case['parent_prediction'])})
        parent = read(case['parent_record'])
        previous = tabular.prediction_grid(missing, Path(case['parent_prediction']), 'prediction')
        parent_score, _ = tabular.checked_score(truth, previous, missing, 'uci', scorer)
        np.testing.assert_allclose(parent_score, parent['new_nrmse'], rtol=1e-6)
        destination = output / 'predictions' / f'{case["case_id"]}.npz'
        record_path = output / 'cases' / f'{case["case_id"]}.json'
        metadata = {key: case[key] for key in ('case_id', 'dataset', 'pattern', 'missingness', 'repeat')}
        if record_path.exists():
            record = read(record_path)
            if any(record[key] != value for key, value in metadata.items()):
                raise ValueError('Cached case identity changed')
            if record['manifest_sha256'] != fingerprint or record['prediction_sha256'] != sha256(destination):
                raise ValueError('Cached prediction or manifest changed')
            prediction = tabular.prediction_grid(missing, destination, 'prediction')
            score, diagnostics = tabular.checked_score(truth, prediction, missing, 'uci', scorer)
            np.testing.assert_allclose([score, parent_score], [record['candidate_nrmse'], record['original_nrmse']], rtol=1e-6)
        else:
            prediction = expert.impute(missing.copy(order='K'))
            score, diagnostics = tabular.checked_score(truth, prediction, missing, 'uci', scorer)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_npz(destination, prediction=prediction[np.isnan(missing)], shape=np.array(missing.shape))
            saved = tabular.prediction_grid(missing, destination, 'prediction')
            saved_score, _ = tabular.checked_score(truth, saved, missing, 'uci', scorer)
            np.testing.assert_allclose(saved_score, score, rtol=1e-6)
            record = dict(metadata, candidate_nrmse=score, original_nrmse=parent_score,
                          prediction_sha256=sha256(destination), manifest_sha256=fingerprint, diagnostics=diagnostics)
            write_json(record_path, record)
        rows.append({key: value for key, value in record.items() if key != 'diagnostics'})
        write_json(output / 'status.json', dict(state='evaluating', completed=len(rows), expected=1800,
                   pid=os.getpid(), elapsed_seconds=time.monotonic() - started, updated=time.time()))
        if len(rows) % 50 == 0:
            print(f'Full UCI: {len(rows)}/1800, elapsed {time.monotonic() - started:.0f}s', flush=True)
    tabular.verify_hashes(sources)
    tabular.verify_hashes({str(checkpoint): manifest['checkpoint_sha256'], str(config_path): manifest['config_sha256'],
                           str(protocol_path): manifest['protocol_sha256']})
    summaries = write_summaries(output, rows)
    write_json(output / 'audit.json', dict(complete=True, cases=len(rows), original_predictions_rescored=True,
               saved_candidate_predictions_rescored=True, source_hashes_verified=True, manifest_sha256=fingerprint))
    write_json(output / 'status.json', dict(state='complete', completed=len(rows), expected=1800,
               elapsed_seconds=time.monotonic() - started, updated=time.time(), overall=summaries['overall'][0]))
    print(json.dumps(summaries['overall'][0]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'retention_full_v1')
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            evaluate(arguments)
        except Exception:
            write_json(arguments.output / 'error.json', dict(error=traceback.format_exc(), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
