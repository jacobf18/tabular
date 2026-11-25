"""Preview completed MissBench/UCI rate sweeps without modifying the paper or archives."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from distributed_imputation import digest, read, write
from plot_final_four_quadrant_preview import SOURCE, generate as combined_figure
from plot_repeated_imputation_preview import (
    BASE, FINAL, LABELS, PAPER, check_prediction, load_records, overlay_mice_retry,
    paper_snapshot, plot_values, validate_protocols,
)
from repeated_imputation_statistics import summarize


EXPANSION = Path('/mnt/matrixformer_scratch/missbench_rate_expansion_20261123')
RETRY = Path('/mnt/matrixformer_scratch/mice_uci40_timeout_retry_20261123')
OUTPUT = Path('/root/tabular/mcpfn/runs/benchmark_evaluations/final_four_quadrant_all_rates_20261124')
RATES = [.1, .2, .3, .4, .5]


def exclusion_reason(truth, mask, suite):
    if not mask.any() or not (~mask).any():
        return 'No held-out entries or no observed context'
    threshold = 1e-3 if suite == 'missbench' else 1e-4
    deviations = [np.std(truth[mask[:, column], column]) for column in range(truth.shape[1])
                  if mask[:, column].any()]
    if not any(value >= threshold for value in deviations):
        return 'No variable held-out columns under the historical NRMSE threshold'
    return None


def exclude_unscorable(root, cases, records, methods):
    expected = {case['case_id']: case for case in cases}
    failures = {row['case_id'] for row in records if row['status'] != 'ok'}
    excluded = []
    for identifier in sorted(failures):
        case = expected[identifier]
        truth = np.load(case['truth_path'], allow_pickle=False).astype(np.float64)
        with np.load(root / 'masks' / (identifier + '.npz'), allow_pickle=False) as packed:
            shape = tuple(packed['shape'])
            mask = np.unpackbits(packed['packed'], count=int(np.prod(shape))).reshape(shape).astype(bool)
        if shape != truth.shape:
            raise ValueError('Truth/mask shape mismatch')
        reason = exclusion_reason(truth, mask, case['suite'])
        if reason is None:
            continue
        matched = [row for row in records if row['case_id'] == identifier]
        if len(matched) != len(methods) or {row['method'] for row in matched} != set(methods):
            raise ValueError('Missing method coverage for exclusion')
        if any(row['status'] not in ('error', 'unevaluable_mask') for row in matched):
            raise ValueError('Unscorable case has inconsistent method outcomes')
        excluded.append(dict(case, exclusion_reason=reason, applies_to_methods=list(methods)))
    removed = {row['case_id'] for row in excluded}
    return ([case for case in cases if case['case_id'] not in removed],
            [row for row in records if row['case_id'] not in removed], excluded)


def incomplete_rate_groups(registered, eligible):
    keys = ['suite', 'dataset', 'pattern']
    original = pd.DataFrame(registered).groupby(keys).missingness.agg(lambda values: sorted(set(values)))
    selected = pd.DataFrame(eligible).groupby(keys).missingness.agg(lambda values: sorted(set(values)))
    return [dict(zip(keys, identity), registered_rates=rates, eligible_rates=selected.get(identity, []))
            for identity, rates in original.items() if selected.get(identity, []) != rates]


def generate(base, final, expansion, retry, source, output):
    if any(output.is_relative_to(root) for root in (base, final, expansion, retry, source, PAPER)):
        raise ValueError('Use a separate preview output directory')
    before = paper_snapshot(PAPER)
    original, latest, masks = validate_protocols(base, final)
    manifest = read(expansion / 'manifest.json')
    extension = manifest['rate_extension']
    if (extension['parent_manifest_sha256'] != digest(base / 'manifest.json')
            or extension['final_manifest_sha256'] != digest(final / 'tabular/manifest.json')
            or manifest['models']['matrixformer'] != latest['models']['matrixformer']
            or set(manifest['methods']) != set(LABELS)):
        raise ValueError('Expansion lineage/model/methods mismatch')
    for field in ('scoring', 'uncertainty', 'inference_seed', 'mask_repeats', 'deterministic_repeats'):
        if original[field] != manifest[field]:
            raise ValueError(f'Expansion protocol mismatch: {field}')
    for dataset in manifest['datasets']:
        if digest(Path(dataset['truth_path'])) != dataset['truth_sha256']:
            raise ValueError('Expansion truth changed')
    expanded_masks = {}
    for case in manifest['cases']:
        if case['suite'] != 'missbench':
            raise ValueError('Expansion contains a different suite')
        path = expansion / 'masks' / (case['case_id'] + '.npz')
        actual = digest(path)
        if actual != read(path.with_suffix('.json'))['mask_sha256']:
            raise ValueError('Expansion mask changed')
        if case['missingness'] == .4 and masks[case['case_id']] != actual:
            raise ValueError('Reused 40% mask differs')
        expanded_masks[case['case_id']] = actual
    base_records, expanded_records, checks = [], [], []
    for method in LABELS:
        root, protocol = (final / 'tabular', latest) if method == 'matrixformer' else (base, original)
        base_records.extend(load_records(root, method, protocol, masks))
        selected = load_records(expansion, method, manifest, expanded_masks)
        expanded_records.extend(selected)
        for rate in RATES:
            successful = sorted((row for row in selected if row['status'] == 'ok' and row['missingness'] == rate),
                                key=lambda row: row['case_id'])
            if not successful:
                raise ValueError(f'No successful predictions for {method}/{rate}')
            checks.append(check_prediction(expansion, method, successful[len(successful) // 2]))
    base_records, retry_checks, receipt = overlay_mice_retry(base, retry, original, masks, base_records)
    checks.extend(retry_checks)
    registered = manifest['cases'] + [case for case in original['cases'] if case['suite'] == 'uci']
    cases, results, excluded = exclude_unscorable(expansion, manifest['cases'], expanded_records, LABELS)
    cases += [case for case in original['cases'] if case['suite'] == 'uci']
    results += [row for row in base_records if row['suite'] == 'uci']
    rates = pd.DataFrame(cases).groupby('suite').missingness.agg(lambda values: sorted(set(values)))
    if any(value != RATES for value in rates):
        raise ValueError('Both suites must include the same five rates')
    scores, coverage, summary, paired = summarize(cases, results, list(LABELS), shared_methods=True)
    values = plot_values(scores, summary, coverage)
    dropped_rates = incomplete_rate_groups(registered, cases)
    tabular = output / 'tabular'
    tabular.mkdir(parents=True, exist_ok=True)
    outputs = dict(values=values, dataset_scores=pd.DataFrame(scores), coverage=pd.DataFrame(coverage),
                   summary=pd.DataFrame(summary), paired_tests=pd.DataFrame(paired),
                   common_exclusions=pd.DataFrame(excluded), prediction_spot_checks=pd.DataFrame(checks),
                   incomplete_rate_groups=pd.DataFrame(dropped_rates),
                   failed_cases=pd.DataFrame([row for row in results if row['status'] != 'ok']))
    for filename, frame in outputs.items():
        frame.to_csv(tabular / f'{filename}.csv', index=False)
    write(tabular / 'validation.json', dict(
        selected_missingness=None, averaged_missingness_rates=RATES, final_model=latest['models']['matrixformer'],
        mice_retry=receipt, common_exclusions=excluded, incomplete_rate_groups=dropped_rates,
        metric=original['scoring'], aggregation='seed -> eligible rate -> pattern -> dataset; equal weights at each level',
        missing_policy='Only truth/mask-verified common unscorable cases excluded; method-specific failures retained, incomplete panels unavailable',
        uncertainty='1000 seed-0 percentile bootstrap resamples of complete dataset means',
        original_case_count=len(registered), eligible_case_count=len(cases),
        case_statuses=dict(Counter(row['status'] for row in results)), prediction_spot_checks=len(checks),
        source_sha256={str(path): digest(path) for path in
                      (base / 'manifest.json', final / 'tabular/manifest.json', expansion / 'manifest.json',
                       retry / 'manifest.json', Path(__file__))}))
    combined_figure(source, final, tabular, output)
    if paper_snapshot(PAPER) != before:
        raise ValueError('Paper changed during preview generation')
    write(output / 'validation.json', dict(paper_unchanged=True, source_archives_read_only=True,
                                         selected_missingness_rates=RATES, predictions_checked=len(checks)))
    lines = ['# All-rates empirical benchmark preview', '',
             'Local preview only; paper and archived evaluation files unchanged.',
             'Only the tabular panels change. Recommendation and causal panels retain the existing results.',
             'Both tabular suites include 10%, 20%, 30%, 40%, and 50% missingness and all 11 patterns.',
             'Average seeds within each valid dataset-pattern-rate, then eligible rates, patterns, and datasets equally.',
             '95% confidence intervals bootstrap dataset means, not rates, seeds, or cells.',
             f'{len(excluded)} truth/mask-verified unscorable MissBench cases are excluded identically for all methods.',
             'Empty conditions are not filled with zero or predictions from another rate.',
             f"UCI MICE includes {len(receipt['replaced_cases'])} verified timeout recoveries. "
             f"Remaining UCI MICE failures: {sum(row['failed'] for row in coverage if row['suite'] == 'uci' and row['method'] == 'mice')}. "
             'Only complete dataset-cohort bars are shown.',
             'The historical held-out-truth-SD normalized column RMSE is unchanged.', '',
             '## Rate-coverage exceptions', '']
    for row in dropped_rates:
        lines.append(f"- {row['suite']}/{row['dataset']}/{row['pattern']}: eligible rates {row['eligible_rates']}; "
                     f"registered rates {row['registered_rates']}.")
    lines.extend(['', 'Exact scores, coverage, excluded cases, and provenance are saved alongside the figure.', '',
                  'Reproduce: `PYTHONPATH=mcpfn/scripts OPENBLAS_NUM_THREADS=1 python mcpfn/scripts/plot_all_rates_benchmark_preview.py`'])
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(values[values.pattern == 'overall'][['suite', 'label', 'nrmse', 'ci95_low', 'ci95_high', 'available']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('base', BASE), ('final', FINAL), ('expansion', EXPANSION), ('retry', RETRY),
                          ('source', SOURCE), ('output', OUTPUT)]:
        parser.add_argument('--' + name, type=Path, default=default)
    arguments = parser.parse_args()
    generate(*(getattr(arguments, name).resolve() for name in ('base', 'final', 'expansion', 'retry', 'source', 'output')))


if __name__ == '__main__':
    main()
