"""Validate completed evaluations and export the active manuscript's results."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distributed_imputation import digest
from plot_final_four_quadrant_preview import (
    CAUSAL_NAMES, METRICS, RATING_NAMES, aggregate_causal, draw_panel,
    select_causal, verify_receipt,
)
from plot_repeated_imputation_preview import bootstrap_bounds


METHODS = (('matrixformer', 'MatrixFormer'), ('missforest', 'MissForest'),
           ('knn', 'K-NN'), ('hyperimpute', 'HyperImpute'), ('mice', 'MICE'),
           ('ice', 'ICE'), ('softimpute', 'SoftImpute'), ('miwae', 'MIWAE'),
           ('gain', 'GAIN'), ('notmiwae', 'not-MIWAE'))
KEYS = [key for key, _ in METHODS]
PATTERNS = (('MCAR', 'MCAR'), ('MNARLatentFactorPattern', 'Latent-MCAR'),
            ('MNARClusterLevelPattern', 'Clust-MCAR'), ('MAR', 'Col-MAR'),
            ('MNARTwoPhaseSubsetPattern', 'Phase-MAR'), ('MNAR', 'Self-MNAR'),
            ('MAR_Neural', 'NN-MNAR'), ('MAR_BlockNeural', 'Block-MNAR'),
            ('MNARPolarizationPattern', 'Pol-MNAR'),
            ('MNARSoftPolarizationPattern', 'Soft-Pol-MNAR'),
            ('MNARCensoringPattern', 'Censor-MNAR'))


def module_from(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hierarchical_scores(records):
    within_rate = records.groupby(
        ['suite', 'dataset', 'method', 'pattern', 'missingness']).nrmse.mean()
    by_pattern = within_rate.groupby(['suite', 'dataset', 'method', 'pattern']).mean()
    overall = by_pattern.groupby(['suite', 'dataset', 'method']).mean().rename('nrmse')
    by_rate = within_rate.groupby(['suite', 'dataset', 'method', 'missingness']).mean()
    return by_pattern.reset_index(), overall.reset_index(), by_rate.reset_index()


def load_tabular(args):
    expanded = pd.read_csv(args.expanded / 'summary/per_case.csv', low_memory=False)
    base = pd.read_csv(args.base / 'summary/per_case.csv', low_memory=False)
    current = pd.read_csv(args.final / 'tabular/summary/per_case.csv')
    records = pd.concat([
        expanded.query("suite == 'missbench'"),
        base.query("suite == 'uci' and method != 'matrixformer'"),
        current.query("suite == 'uci' and method == 'matrixformer'"),
    ], ignore_index=True)
    records = records[records.method.isin(KEYS)].copy()
    audit = json.loads((args.preview / 'tabular/validation.json').read_text())
    if digest(args.mice_retry / 'manifest.json') != audit['mice_retry']['manifest_sha256']:
        raise ValueError('MICE recovery manifest differs from the audited preview')
    retries = {record['case_id']: record for path in
               (args.mice_retry / 'results/mice/cases').rglob('*.json')
               for record in [json.loads(path.read_text())] if record['status'] == 'ok'}
    for identifier in audit['mice_retry']['replaced_cases']:
        record = retries[identifier]
        selected = (records.case_id == identifier) & (records.method == 'mice')
        if selected.sum() != 1:
            raise ValueError(f'Nonunique MICE case: {identifier}')
        for field in ('mask_sha256', 'truth_sha256'):
            if records.loc[selected, field].item() != record[field]:
                raise ValueError(f'MICE recovery changes {field}')
        records.loc[selected, ['status', 'nrmse', 'prediction_sha256']] = [
            record['status'], record['nrmse'], record['prediction_sha256']]
    exclusions = pd.read_csv(args.preview / 'tabular/common_exclusions.csv')
    records = records[~records.case_id.isin(exclusions.case_id)]
    if records.duplicated(['case_id', 'method']).any():
        raise ValueError('Duplicated method-case records')
    if not records.status.eq('ok').all() or not np.isfinite(records.nrmse).all():
        raise ValueError('Incomplete or nonfinite evaluation results')
    support = records.groupby('case_id').agg(methods=('method', 'nunique'),
                                             masks=('mask_sha256', 'nunique'),
                                             truths=('truth_sha256', 'nunique'))
    if not ((support.methods == len(METHODS)) & (support.masks == 1) & (support.truths == 1)).all():
        raise ValueError('Methods must share all cases, masks and ground truth')
    if len(support) != 13504 or len(records) != 135040:
        raise ValueError('Unexpected eligible-case count')
    by_pattern, overall, by_rate = hierarchical_scores(records)
    expected = pd.read_csv(args.preview / 'tabular/dataset_scores.csv')
    actual = pd.concat([by_pattern.rename(columns={'pattern': 'panel'}),
                        overall.assign(panel='Overall')], ignore_index=True)
    keys = ['suite', 'dataset', 'method', 'panel']
    expected = expected[expected.panel.isin([name for name, _ in PATTERNS] + ['Overall'])]
    joined = actual.merge(expected, on=keys, validate='one_to_one')
    if len(joined) != len(actual) or len(joined) != len(expected):
        raise ValueError('Dataset/pattern support differs from the audited figure')
    np.testing.assert_allclose(joined.nrmse_x, joined.nrmse_y, atol=1e-12, rtol=0)
    shapes = []
    for root in (args.expanded, args.base):
        manifest = json.loads((root / 'manifest.json').read_text())
        for dataset in manifest['datasets']:
            shapes.append(dict(suite=dataset['suite'], dataset=dataset['dataset'],
                               rows=dataset['shape'][0], columns=dataset['shape'][1]))
    shapes = pd.DataFrame(shapes).drop_duplicates(['suite', 'dataset'])
    shapes = shapes.merge(overall[['suite', 'dataset']].drop_duplicates(), validate='one_to_one')
    return by_pattern, overall, by_rate, shapes


def export_tabular(args, by_pattern, overall, by_rate, shapes):
    helper = module_from(args.paper / 'scripts/export_imputation_tables_v2.py')
    helper.METHODS, helper.KEYS = METHODS, KEYS
    helper.UCI_NAMES['statlog_landsat_satellite'] = 'Landsat Satellite'
    output = args.paper / 'sections_final/appendix/imputation_v2_tables'
    files = {}
    for suite, title, count in [('missbench', 'MissBench', 41), ('uci', 'UCI', 12)]:
        names = helper.tex if suite == 'missbench' else helper.UCI_NAMES.__getitem__
        means = by_pattern[by_pattern.suite == suite].groupby(['pattern', 'method']).nrmse.mean().unstack('method')[KEYS]
        means.loc['Overall'] = overall[overall.suite == suite].groupby('method').nrmse.mean()
        columns = PATTERNS + (('Overall', 'Overall'),)
        nothing = pd.DataFrame(False, index=means.index, columns=KEYS)
        label = 'missbench-mechanisms' if suite == 'missbench' else 'uci-mechanisms-rates'
        filename = 'missbench_mechanisms' if suite == 'missbench' else 'uci_mechanisms_rates'
        files[filename] = helper.wrap(
            f'{title} RMSE by missingness mechanism; lower is better. Seeds are averaged within each '
            f'dataset--mechanism--rate condition, then eligible rates from 10\\% to 50\\% are weighted equally, '
            f'followed by the {count} datasets. Overall additionally weights the eleven mechanisms equally '
            'within each dataset. All ten methods use identical eligible cases. Bold marks the lowest value in each column.',
            f'tab:{label}', '@{}l' + 'r' * len(columns) + '@{}',
            ['Method & ' + ' & '.join(label for _, label in columns) + ' \\\\'],
            helper.by_method(means, nothing, columns))
        rates = by_rate[by_rate.suite == suite].groupby(['missingness', 'method']).nrmse.mean().unstack('method')[KEYS]
        rate_columns = tuple((rate, f'{100*rate:.0f}\\%') for rate in rates.index)
        files[f'{suite}_rates'] = helper.wrap(
            f'{title} RMSE by nominal missingness rate, with seeds averaged within each dataset--mechanism '
            f'condition, followed by mechanisms and the {count} datasets. Unscorable conditions are excluded '
            'identically for every method. Bold marks the lowest value in each column.',
            f'tab:{suite}-rates', '@{}lrrrrr@{}',
            ['Method & ' + ' & '.join(label for _, label in rate_columns) + ' \\\\'],
            helper.by_method(rates, pd.DataFrame(False, index=rates.index, columns=KEYS), rate_columns))
        per = overall[overall.suite == suite].pivot(index='dataset', columns='method', values='nrmse')[KEYS]
        per = per.loc[sorted(per.index, key=lambda name: names(name).lower())]
        files[f'{suite}_per_dataset'] = helper.wrap(
            f'{title} RMSE by dataset. Seeds are averaged within each mechanism--rate condition, '
            'then eligible rates and the eleven mechanisms are weighted equally. All ten methods use '
            'identical eligible cases. Bold marks the lowest value in each row.',
            f'tab:{suite}-per-dataset', '@{}l' + 'r' * len(KEYS) + '@{}',
            ['Dataset & ' + ' & '.join(name for _, name in METHODS) + ' \\\\'],
            helper.by_dataset(per, pd.DataFrame(False, index=per.index, columns=KEYS), names),
            place='p' if suite == 'missbench' else '!htbp')
        halves = 3 if suite == 'missbench' else 2
        sizes = shapes[shapes.suite == suite].assign(dataset=lambda frame: frame.dataset.map(names))
        files[f'{suite}_datasets'] = helper.wrap(
            f'The {count} {title} datasets and their matrix dimensions.', f'tab:{suite}-datasets',
            '@{}' + '@{\\qquad}'.join(['lrr'] * halves) + '@{}',
            [' & '.join(['Dataset & Rows & Cols'] * halves) + ' \\\\'], helper.dataset_list(sizes, str, halves))
    for filename, lines in files.items():
        (output / f'{filename}.tex').write_text('\n'.join(lines) + '\n')


def export_causal(args, data):
    current = pd.read_csv(args.final / 'causal/per_draw_metrics.csv')
    if set(current.method) != {'MatrixFormer-final'}:
        raise ValueError('Expected only the final standard-orientation predictor')
    current = current.assign(method='matrixformer')
    reference = pd.read_csv(args.final / 'causal/reference_per_draw_metrics.csv')
    reference = reference[reference.method.isin(set(CAUSAL_NAMES) - {'matrixformer'})]
    cate = pd.read_csv(args.paper / 'results/single_model/causal_cate_methods/per_draw_metrics.csv')
    cate = cate[cate.method.isin(['T-learner', 'CausalPFN', 'T-learner [mu0]', 'CausalPFN [mu0]'])]
    draws = pd.concat([current, reference, cate], ignore_index=True)
    support_keys = ['family', 'path', 'draw_name']
    support = set(current[support_keys].itertuples(index=False, name=None))
    for method, group in draws.groupby('method'):
        if len(group) != 3000 or set(group[support_keys].itertuples(index=False, name=None)) != support:
            raise ValueError(f'Unequal causal support: {method}')
    scores = draws.groupby(['family', 'path', 'method'])[list(METRICS)].mean().reset_index().melt(
        id_vars=['family', 'path', 'method'], var_name='metric', value_name='raw_mean')
    scores.to_csv(data / 'causal_per_design.csv', index=False)
    values = select_causal(aggregate_causal(draws))
    values.to_csv(data / 'causal_summary.csv', index=False)
    helper = module_from(args.paper / 'scripts/export_causal_design_tables.py')
    output = args.paper / 'sections_final/appendix/causal_tables_v2.tex'
    lines = []
    for metric, methods, label, caption in helper.METRICS:
        lines.extend(helper.design_table(scores, metric, methods, label, caption))
        lines.append('')
    output.write_text('\n'.join(lines) + '\n')
    paired = []
    for metric in METRICS:
        wide = draws.pivot(index=support_keys, columns='method', values=metric)
        for (family, path), group in wide.groupby(['family', 'path']):
            difference = group.matrixformer - group.sdid
            half = 1.96 * difference.sem()
            paired.append(dict(family=family, path=path, metric=metric, mean=difference.mean(),
                               lower=difference.mean()-half, upper=difference.mean()+half))
    pd.DataFrame(paired).to_csv(data / 'causal_paired_sdid.csv', index=False)
    return values


def export_recsys(args, data):
    manifest = json.loads((args.final / 'manifest.json').read_text())
    expected = pd.read_csv(args.final / 'recsys/summary.csv')
    archived = pd.read_csv(args.paper / 'results/accuracy_uncertainty/recsys_summary.csv')
    rows = []
    for case in manifest['rating_cases']:
        if digest(Path(case['split'])) != case['split_sha256']:
            raise ValueError('Rating split changed')
        with np.load(case['split']) as split, \
                np.load(args.final / f"recsys/{case['dataset']}/per_user.npz") as current, \
                np.load(args.final / f"recsys/{case['dataset']}/predictions.npz") as predictions, \
                np.load(args.paper / f"results/accuracy_uncertainty/recsys_{case['dataset']}_per_user.npz") as baseline:
            users, inverse, counts = np.unique(split['test_users'], return_inverse=True, return_counts=True)
            np.testing.assert_array_equal(users, current['users'])
            np.testing.assert_array_equal(users, baseline['users'])
            errors = predictions['prediction'].astype(float) - split['test_ratings']
            if not (predictions['counts'] > 0).all():
                raise ValueError('Missing rating predictions')
            for index, metric in enumerate(('RMSE', 'MAE')):
                recomputed = np.bincount(inverse, weights=errors**2 if metric == 'RMSE' else abs(errors))/counts
                if metric == 'RMSE':
                    recomputed = np.sqrt(recomputed)
                np.testing.assert_allclose(recomputed, current[metric], atol=1e-10, rtol=1e-10)
                for method, label in RATING_NAMES.items():
                    scores = current[metric] if method == 'matrixformer' else baseline[method][:, index]
                    lower, upper = bootstrap_bounds(scores)
                    if method == 'matrixformer':
                        check = expected[(expected.dataset == case['label']) & (expected.metric == metric)].iloc[0]
                        checked = [check.score, check.ci_lower, check.ci_upper]
                    else:
                        check = archived[(archived.dataset == case['label']) & (archived.metric == metric) & (archived.method == method)].iloc[0]
                        checked = [check.value, check.ci95_low, check.ci95_high]
                    np.testing.assert_allclose([scores.mean(), lower, upper], checked, atol=1e-12, rtol=0)
                    rows.append(dict(domain='recsys', dataset=case['label'], metric=metric,
                                     method=method, label=label, value=scores.mean(),
                                     ci_lower=lower, ci_upper=upper, available=True))
    values = pd.DataFrame(rows)
    values.to_csv(data / 'recsys_summary.csv', index=False)
    style = module_from(args.paper / 'scripts/figure_style.py')
    style.apply()
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 2.9))
    for axis, (dataset, metric) in zip(axes, [(case['label'], metric) for case in manifest['rating_cases'] for metric in ('RMSE', 'MAE')]):
        selected = values[(values.dataset == dataset) & (values.metric == metric)]
        draw_panel(axis, selected, style, limits=(.6, 1.09, [.6, .8, 1.], 3))
        axis.set_title(f'{dataset}\n{metric}', fontsize=11)
        axis.set_xlabel('Error (lower is better)', fontsize=10)
    figure.tight_layout(pad=.6, w_pad=1.)
    style.save(figure, args.paper / 'figures/previews/table1_recommendation_bars_four_columns')
    plt.close(figure)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('paper', 'preview', 'final', 'base', 'expanded', 'mice-retry'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    manifest_hash = digest(args.final / 'manifest.json')
    for domain in ('causal', 'recsys'):
        verify_receipt(args.final / f'{domain}.done.json', manifest_hash)
    by_pattern, overall, by_rate, shapes = load_tabular(args)
    data = args.paper / 'results/final_model'
    data.mkdir(parents=True, exist_ok=True)
    for name, frame in [('tabular_by_pattern', by_pattern), ('tabular_overall', overall),
                        ('tabular_by_rate', by_rate), ('dataset_shapes', shapes)]:
        frame.to_csv(data / f'{name}.csv', index=False)
    export_tabular(args, by_pattern, overall, by_rate, shapes)
    causal = export_causal(args, data)
    recsys = export_recsys(args, data)
    expected = pd.read_csv(args.preview / 'values.csv')
    actual = pd.concat([causal, recsys[recsys.metric == 'RMSE']], ignore_index=True)
    matched = actual.merge(expected, on=['domain', 'dataset', 'metric', 'method'], validate='one_to_one')
    if len(matched) != len(actual):
        raise ValueError('Figure support mismatch')
    for field in ('value', 'ci_lower', 'ci_upper'):
        np.testing.assert_allclose(matched[field+'_x'], matched[field+'_y'], atol=1e-12, rtol=0)
    for extension in ('pdf', 'png'):
        shutil.copy2(args.preview / f'four_quadrant_summary_sorted.{extension}', args.paper / f'figures/four_quadrant_summary_sorted.{extension}')
    expected.to_csv(data / 'figure_values.csv', index=False)
    audit = json.loads((args.preview / 'tabular/validation.json').read_text())
    metadata = dict(checkpoint_sha256=audit['final_model']['checkpoint_sha256'],
                    config_sha256=audit['final_model']['config_sha256'],
                    datasets={'missbench': 41, 'uci': 12}, eligible_cases={'missbench': 10446, 'uci': 3058},
                    methods=dict(METHODS), rates=[.1, .2, .3, .4, .5], stochastic_repeats=5,
                    deterministic_repeats=1, shared_support_verified=True,
                    score='columnwise held-out-truth SD normalized RMSE',
                    tabular_intervals='1000 dataset-mean percentile bootstrap resamples, seed 0',
                    recsys_intervals='1000 test-user percentile bootstrap resamples, seed 0')
    (data / 'validation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Updated eight tabular tables, two causal tables, recommendation figure and empirical summary.')


if __name__ == '__main__':
    main()
