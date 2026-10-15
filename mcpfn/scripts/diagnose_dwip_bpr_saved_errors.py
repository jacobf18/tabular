"""Describe existing inner predictions without inference, training or calibration."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from diagnose_dwip_recsys_macro_errors import per_user_errors
from evaluate_dwip_recsys import sha256, write_json


def compare_predictions(before, after):
    for name in ('users', 'items', 'truth', 'prediction'):
        if before[name].ndim != 1 or after[name].shape != before[name].shape:
            raise ValueError('Require aligned one-dimensional saved queries')
    if not len(before['users']) or not all(np.array_equal(before[name], after[name]) for name in ('users', 'items', 'truth')):
        raise ValueError('Saved inner query identities or targets differ')
    if any(not np.isfinite(values[name]).all() for values in (before, after) for name in ('truth', 'prediction')):
        raise ValueError('Nonfinite saved predictions or targets')
    counts, original = per_user_errors(before['users'], before['truth'], before['prediction'])
    after_counts, current = per_user_errors(after['users'], after['truth'], after['prediction'])
    if not np.array_equal(counts, after_counts):
        raise ValueError('Saved query user counts differ')
    included = counts > 0
    delta = after['prediction'] - before['prediction']
    squared_change = np.bincount(before['users'], weights=delta ** 2, minlength=len(counts)) / counts.clip(1)
    mse_change = float(np.mean((current['MSE'] - original['MSE'])[included]))
    bias_change = float(np.mean((current['bias'] ** 2 - original['bias'] ** 2)[included]))
    centered_change = float(np.mean(((current['MSE'] - current['bias'] ** 2)
                                    - (original['MSE'] - original['bias'] ** 2))[included]))
    if not np.isclose(mse_change, bias_change + centered_change, rtol=1e-12, atol=1e-12):
        raise ValueError('User-MSE descriptive decomposition does not reproduce')
    result = {label: {**{metric: float(np.mean(values[metric][included])) for metric in ('MAE', 'RMSE')},
                      'users': int(included.sum())} for label, values in (('before', original), ('after', current))}
    result.update({
        'before_macro_signed_error': float(np.mean(original['bias'][included])),
        'after_macro_signed_error': float(np.mean(current['bias'][included])),
        'macro_prediction_shift': float(np.mean((current['bias'] - original['bias'])[included])),
        'macro_prediction_change_rms': float(np.mean(np.sqrt(squared_change[included]))),
        'user_mean_squared_error_change': mse_change,
        'bias_squared_component_change': bias_change,
        'within_user_centered_error_component_change': centered_change,
        'users_with_improved_rmse_fraction': float(np.mean(current['RMSE'][included] < original['RMSE'][included])),
        'query_prediction_change_quantiles': np.quantile(delta, [0, .1, .5, .9, 1]).tolist(),
    })
    return result


def diagnose(trial, step):
    if step < 0:
        raise ValueError('Require a nonnegative saved validation step')
    canonical = {index: json.loads((trial / f'validation_{index}.json').read_text()) for index in (0, step)}
    result = {'partition': 'existing inner validation only', 'training_or_calibration_fit': False,
              'model_updated': False, 'new_inference_performed': False, 'step': step, 'datasets': {},
              'quantile_probabilities': [0, .1, .5, .9, 1],
              'decomposition_scope': 'Macro mean of user MSE, not the squared macro RMSE; descriptive only',
              'source_sha256': sha256(Path(__file__)), 'artifact_sha256': {}}
    for dataset in ('movielens_100k', 'netflix_small'):
        predictions, metrics, profiles = {}, {}, {}
        for label, index in (('before', 0), ('after', step)):
            directory = trial / f'validation_details_{index}' / dataset
            audit = json.loads((directory / 'audit.json').read_text())
            if audit['gradients_enabled'] or not audit['finite_predictions'] or not audit['observed_ratings_preserved']:
                raise ValueError('Saved evaluation audit is not eligible for read-only comparison')
            with np.load(directory / 'query_predictions.npz', allow_pickle=False) as data:
                predictions[label] = {name: data[name].copy() for name in data.files}
            metrics[label] = json.loads((directory / 'metrics.json').read_text())
            if metrics[label] != canonical[index]['datasets'][dataset]:
                raise ValueError('Saved metric details differ from canonical validation')
            profiles[label] = json.loads((directory / 'ranking_profile.json').read_text())['all']
            if not np.isclose(profiles[label]['recall10'], metrics[label]['Recall@10'], rtol=1e-12, atol=1e-12):
                raise ValueError('Saved ranking profile Recall differs from validation')
            with np.load(directory / 'ranking_metrics.npz', allow_pickle=False) as ranking:
                for metric in ranking.files:
                    if metric != 'users' and not np.isclose(ranking[metric].mean(), metrics[label][metric], rtol=1e-12, atol=1e-12):
                        raise ValueError('Saved per-user ranking values do not reproduce validation')
            for path in (trial / f'validation_{index}.json', *directory.iterdir()):
                if path.is_file():
                    result['artifact_sha256'][str(path)] = sha256(path)
        comparison = compare_predictions(predictions['before'], predictions['after'])
        for label in ('before', 'after'):
            for metric in ('MAE', 'RMSE'):
                if not np.isclose(comparison[label][metric], metrics[label][metric], rtol=1e-12, atol=1e-12):
                    raise ValueError('Saved query rating metrics do not reproduce validation')
            comparison[f'{label}_ranking_profile'] = profiles[label]
        result['datasets'][dataset] = comparison
    if any(sha256(Path(path)) != digest for path, digest in result['artifact_sha256'].items()):
        raise ValueError('Saved input artifact changed during diagnosis')
    result['all_saved_rating_and_ranking_metrics_reproduce'] = True
    result['input_artifacts_unchanged'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or re.match(r'validation_[0-9]', args.output.name):
        raise ValueError('Require a new diagnostic path outside canonical validation filenames')
    report = diagnose(args.trial, args.step)
    write_json(args.output, report)
    print(json.dumps({'step': args.step, 'metrics_reproduce': True,
                      'datasets': {name: {key: value[key] for key in ('before', 'after', 'macro_prediction_shift', 'user_mean_squared_error_change')}
                                   for name, value in report['datasets'].items()}}), flush=True)


if __name__ == '__main__':
    main()
