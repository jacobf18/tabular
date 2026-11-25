"""Render the attached empirical summary with the final checkpoint, outside the paper."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from distributed_imputation import digest, read, write
from plot_repeated_imputation_preview import FINAL, PAPER, bootstrap_bounds, paper_snapshot


SOURCE = Path('/root/tabular/mcpfn/69d7e55fd826e0325c3115d0')
TABULAR = Path('/root/tabular/mcpfn/runs/benchmark_evaluations/repeated_protocol_preview_20261123_missing40')
OUTPUT = TABULAR.parent / 'final_four_quadrant_preview_20261123'
OLD_CAUSAL = Path('/root/tabular/mcpfn/runs/benchmark_evaluations/single_matrixformer_causal_20261102')
ATTACHMENT = Path('/root/.codex/attachments/38aaf5ef-d584-4102-86e2-85a629e37bb6/four_quadrant_summary_sorted.png')
FAMILIES = ('California smoking', 'CPS', 'Kansas')
METRICS = ('treated_block_rmse', 'tau_abs_error')
CAUSAL_NAMES = dict(matrixformer='MatrixFormer', sdid='SDiD', sc='SC', did='DiD',
                    hyperimpute='HyperImpute', knn='K-NN', softimpute='SoftImpute',
                    **{'T-learner': 'T-learner', 'CausalPFN': 'CausalPFN'})
RATING_NAMES = dict(matrixformer='MatrixFormer', SVD='Biased MF', PMF='PMF', UserKNN='UserKNN', ItemKNN='ItemKNN')
VALUE_DECIMALS = 3
VALUE_LABEL_SIZE = 9.5
CAUSAL_AXES = {('California smoking', 'treated_block_rmse'): (10., 25., [10, 15, 20], 2),
               ('CPS', 'treated_block_rmse'): (.50, .68, [.50, .60], 3),
               ('Kansas', 'treated_block_rmse'): (0., .15, [0, .05, .10], 3),
               ('California smoking', 'tau_abs_error'): (2., 10., [2, 6, 10], 2),
               ('CPS', 'tau_abs_error'): (.08, .20, [.10, .15, .20], 3),
               ('Kansas', 'tau_abs_error'): (0., .14, [0, .05, .10], 3)}


def aggregate_causal(draws):
    keys = ['family', 'path', 'draw_name', 'method']
    if draws.duplicated(keys).any() or not np.isfinite(draws[list(METRICS)]).all().all():
        raise ValueError('Duplicate or nonfinite causal draws')
    rows = []
    for (family, method), group in draws.groupby(['family', 'method']):
        for metric in METRICS:
            design = group.groupby('path')[metric].agg(['mean', 'sem', 'count'])
            if (design['count'] < 2).any():
                raise ValueError('Causal intervals require repeated draws within each design')
            value = design['mean'].mean()
            half = 1.96 * np.sqrt(np.square(design['sem']).sum()) / len(design)
            rows.append(dict(domain='causal', dataset=family, metric=metric, method=method,
                             value=value, ci_lower=value-half, ci_upper=value+half,
                             units=int(design['count'].sum()), designs=len(design), available=True))
    return pd.DataFrame(rows)


def select_causal(scores):
    rows = []
    for row in scores.to_dict('records'):
        method = row['method']
        if method in ('T-learner', 'CausalPFN', 'T-learner [mu0]', 'CausalPFN [mu0]'):
            untreated = method.endswith(' [mu0]')
            if untreated != (row['metric'] == 'treated_block_rmse'):
                continue
            method = method.removesuffix(' [mu0]')
        if method in CAUSAL_NAMES:
            rows.append({**row, 'method': method, 'label': CAUSAL_NAMES[method]})
    return pd.DataFrame(rows)


def verify_receipt(path, manifest_hash):
    receipt = read(path)
    if receipt['manifest_sha256'] != manifest_hash:
        raise ValueError(f'Campaign mismatch: {path}')
    for output, expected in receipt['outputs'].items():
        if digest(Path(output)) != expected:
            raise ValueError(f'Changed campaign output: {output}')


def causal_values(source, final, old):
    reference = pd.read_csv(final / 'causal/reference_per_draw_metrics.csv')
    previous_reference = pd.read_csv(old / 'reference_per_draw_metrics.csv')
    keys = ['family', 'path', 'draw_name', 'method']
    previous_reference = previous_reference[previous_reference.method.isin(reference.method.unique())]
    joined = reference.merge(previous_reference, on=keys, validate='one_to_one')
    if len(joined) != len(reference) or len(joined) != len(previous_reference):
        raise ValueError('Causal baseline support changed')
    for metric in METRICS:
        np.testing.assert_allclose(joined[metric + '_x'], joined[metric + '_y'], atol=1e-12, rtol=0)
    current = pd.read_csv(final / 'causal/per_draw_metrics.csv')
    if set(current.method) != {'MatrixFormer-final'}:
        raise ValueError('Expected only standard-orientation final causal predictions')
    current = current.assign(method='matrixformer')
    previous = pd.read_csv(old / 'per_draw_metrics.csv').query("method == 'matrixformer'")
    cate = pd.read_csv(source / 'results/single_model/causal_cate_methods/per_draw_metrics.csv')
    cate = cate[cate.method.isin(['T-learner', 'T-learner [mu0]', 'CausalPFN', 'CausalPFN [mu0]'])]
    reference = reference[reference.method.isin(CAUSAL_NAMES)]
    support_keys = ['family', 'path', 'draw_name']
    support = set(current[support_keys].itertuples(index=False, name=None))
    if len(current) != len(support) or len(current) != 3000:
        raise ValueError('Unexpected final causal draw count')
    for method, group in pd.concat([reference, previous, cate]).groupby('method'):
        if len(group) != len(support) or set(group[support_keys].itertuples(index=False, name=None)) != support:
            raise ValueError(f'Causal draw support differs: {method}')
    old_scores = aggregate_causal(pd.concat([reference, previous, cate], ignore_index=True))
    archived = pd.read_csv(source / 'results/single_model/causal_cate_methods/raw_scores.csv')
    matched = old_scores.merge(archived, left_on=['dataset', 'metric', 'method'], right_on=['family', 'metric', 'method'],
                               validate='one_to_one')
    if len(matched) != len(old_scores):
        raise ValueError('The attachment has different causal methods')
    np.testing.assert_allclose(matched.value, matched.score, atol=5.1e-7, rtol=0)
    np.testing.assert_allclose(matched.ci_upper - matched.value, matched.ci95, atol=5.1e-7, rtol=0)
    values = select_causal(aggregate_causal(pd.concat([reference, current, cate], ignore_index=True)))
    comparison = old_scores[old_scores.method == 'matrixformer'][['dataset', 'metric', 'value']].rename(columns={'value': 'previous'})
    comparison = comparison.merge(values[values.method == 'matrixformer'][['dataset', 'metric', 'value']],
                                  on=['dataset', 'metric'], validate='one_to_one').rename(columns={'value': 'final'})
    return values, comparison.assign(domain='causal')


def rating_values(source, final, manifest):
    archived = pd.read_csv(source / 'results/accuracy_uncertainty/recsys_summary.csv')
    current = pd.read_csv(final / 'recsys/summary.csv')
    rows, comparisons = [], []
    for case in manifest['rating_cases']:
        dataset = case['label']
        if digest(Path(case['split'])) != case['split_sha256']:
            raise ValueError('Rating split changed')
        with np.load(case['split'], allow_pickle=False) as split, \
                np.load(final / f"recsys/{case['dataset']}/predictions.npz", allow_pickle=False) as saved, \
                np.load(final / f"recsys/{case['dataset']}/per_user.npz", allow_pickle=False) as latest, \
                np.load(source / f"results/accuracy_uncertainty/recsys_{case['dataset']}_per_user.npz", allow_pickle=False) as prior:
            users, inverse, counts = np.unique(split['test_users'], return_inverse=True, return_counts=True)
            np.testing.assert_array_equal(users, latest['users'])
            np.testing.assert_array_equal(users, prior['users'])
            if len(saved['prediction']) != case['queries'] or not (saved['counts'] > 0).all():
                raise ValueError('Incomplete rating predictions')
            errors = saved['prediction'].astype(float) - split['test_ratings']
            recomputed = np.sqrt(np.bincount(inverse, weights=errors ** 2) / counts)
            np.testing.assert_allclose(recomputed, latest['RMSE'], rtol=1e-10, atol=1e-10)
            for method, label in RATING_NAMES.items():
                scores = latest['RMSE'] if method == 'matrixformer' else prior[method][:, 0]
                lower, upper = bootstrap_bounds(scores)
                value = float(scores.mean())
                if method == 'matrixformer':
                    expected = current[(current.dataset == dataset) & (current.metric == 'RMSE')].iloc[0]
                    expected_values = [expected.score, expected.ci_lower, expected.ci_upper]
                else:
                    expected = archived[(archived.dataset == dataset) & (archived.metric == 'RMSE') & (archived.method == method)].iloc[0]
                    expected_values = [expected.value, expected.ci95_low, expected.ci95_high]
                np.testing.assert_allclose([value, lower, upper], expected_values, rtol=0, atol=1e-12)
                rows.append(dict(domain='recsys', dataset=dataset, metric='RMSE', method=method, label=label,
                                 value=value, ci_lower=lower, ci_upper=upper, units=len(scores), available=True))
            comparisons.append(dict(domain='recsys', dataset=dataset, metric='RMSE',
                                    previous=float(prior['matrixformer_single'][:, 0].mean()), final=float(latest['RMSE'].mean())))
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def draw_panel(axis, data, style, limits=None):
    data = data.assign(order_score=data.value.round(6)).sort_values(['order_score', 'label'], na_position='last')
    if limits is None:
        start, end, ticks, digits = 0., data.ci_upper.max() * 1.3, None, 3
    else:
        start, end, ticks, digits = limits
    span = end - start
    best = data.value.round(digits).min()
    bars = []
    for position, row in enumerate(data.itertuples()):
        if not row.available:
            axis.text(start + span * .025, position, '—', va='center', fontsize=8.45, color='#777777')
            continue
        if row.ci_lower < start:
            raise ValueError(f'Lower axis truncates an interval: {row.dataset}/{row.method}')
        clipped = row.value > end
        endpoint = min(row.value, end)
        bar = axis.barh(position, endpoint-start, left=start, height=.7, linewidth=0,
                        color=style.BLUE if row.method == 'matrixformer' else style.NEUTRAL)[0]
        bars.append((bar, endpoint))
        if not clipped:
            axis.errorbar(row.value, position, xerr=[[row.value-row.ci_lower], [row.ci_upper-row.value]],
                          fmt='none', ecolor=style.ERROR, elinewidth=.91, capsize=1.95)
        label = f'{row.value:.{VALUE_DECIMALS}f}' + (' ▸' if clipped else '')
        axis.text(end-span*.025 if clipped else row.ci_upper+span*.035, position, label, va='center',
                  ha='right' if clipped else 'left', fontsize=VALUE_LABEL_SIZE,
                  color=style.LABEL_BLUE if row.method == 'matrixformer' else style.LABEL_GRAY,
                  fontweight='bold' if round(row.value, digits) == best else 'normal',
                  bbox=None if clipped else {'facecolor': 'white', 'edgecolor': 'none', 'pad': .4})
    axis.set_xlim(start, end)
    axis.set_ylim(len(data)-.45, -.6)
    axis.set_yticks(range(len(data)), data.label)
    if ticks is None:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        axis.set_xticks(ticks)
    style.style_axes(axis)
    axis.grid(False, which='both')
    axis.tick_params(labelsize=9.75, length=2.6, pad=1.95)
    for tick, method in zip(axis.get_yticklabels(), data.method):
        if method == 'matrixformer':
            tick.set_color(style.BLUE)
            tick.set_fontweight('bold')
    np.testing.assert_allclose([bar.get_x()+bar.get_width() for bar, endpoint in bars], [endpoint for bar, endpoint in bars])


def tabular_description(audit):
    if audit['selected_missingness'] == .4:
        return '40% missing, all 11 patterns'
    if audit['selected_missingness'] is None and audit.get('averaged_missingness_rates') == [.1, .2, .3, .4, .5]:
        return '10–50% missing (rate-averaged), all 11 patterns'
    raise ValueError('Unsupported tabular missingness selection')


def draw(values, output, style, tabular_note='40% missing, all 11 patterns'):
    style.apply()
    plt.rcParams['axes.linewidth'] = 1.04
    width, height = 864., 500.
    figure = plt.figure(figsize=(width/72, height/72))

    def heading(left, top, text):
        figure.text(left/width, 1-top/height, text, fontsize=14, weight='bold', va='top', color='#222222')

    def panel(left, cell, padding, top, bottom, title, data, limits=None):
        axis = figure.add_axes([(left+padding)/width, 1-bottom/height,
                                (cell-padding-4)/width, (bottom-top)/height])
        figure.text((left+padding+(cell-padding-4)/2)/width, 1-(top-18)/height,
                    title, ha='center', va='center', fontsize=10.5, weight='bold', color='#222222')
        draw_panel(axis, data, style, limits)

    right = 20 + 3*1.3*106.6
    heading(20, 9, '(a) Tabular imputation: RMSE (↓)')
    heading(right, 9, '(b) Recommendation system: RMSE (↓)')
    heading(20, 264, '(c) Causal inference: Treated-block RMSE (↓)')
    heading(right, 264, '(d) Causal inference: Treatment effect error (↓)')
    for index, dataset in enumerate(('MissBench', 'UCI')):
        panel(20+index*205.4, 205.4, 69., 57, 237, dataset,
              values[(values.domain == 'tabular') & (values.dataset == dataset)])
    for index, dataset in enumerate(('MovieLens 100K', 'Netflix small')):
        panel(right+index*205.4, 205.4, 69., 57, 237, dataset,
              values[(values.domain == 'recsys') & (values.dataset == dataset)], (.82, 1.08, [.85, .95, 1.05], 4))
    for metric_index, metric in enumerate(METRICS):
        for index, family in enumerate(FAMILIES):
            panel(20+(metric_index*3+index)*138.58, 138.58, 53., 314, 470, family,
                  values[(values.domain == 'causal') & (values.dataset == family) & (values.metric == metric)],
                  CAUSAL_AXES[family, metric])
    missing_note = '' if values.available.all() else '  — incomplete coverage.'
    figure.text(20/width, 8/height,
                f'Tabular: {tabular_note}.  Whiskers: 95% CIs.' + missing_note + '  ▸ beyond axis limit.',
                fontsize=8, color='#555555')
    style.save(figure, output / 'four_quadrant_summary_sorted')
    plt.close(figure)


def generate(source, final, tabular, output):
    if any(output.is_relative_to(protected) for protected in (source, final, tabular, PAPER)):
        raise ValueError('Use a separate local output directory')
    before = paper_snapshot(PAPER)
    inputs = [final/'manifest.json', tabular/'validation.json', tabular/'values.csv',
              source/'results/accuracy_uncertainty/recsys_summary.csv',
              source/'results/single_model/causal_cate_methods/raw_scores.csv',
              source/'results/single_model/causal_cate_methods/per_draw_metrics.csv',
              final/'causal/per_draw_metrics.csv', final/'causal/reference_per_draw_metrics.csv',
              final/'recsys/summary.csv', source/'scripts/assemble_four_quadrant_figure.py',
              source/'scripts/figure_style.py', Path(__file__)]
    manifest = read(final/'manifest.json')
    tabular_audit = read(tabular/'validation.json')
    if tabular_audit['final_model'] != manifest['model']:
        raise ValueError('Tabular preview has different checkpoint')
    tabular_note = tabular_description(tabular_audit)
    if manifest['protocol']['orientation'] != 'standard':
        raise ValueError('Expected standard orientation')
    for receipt in ('causal.done.json', 'recsys.done.json'):
        verify_receipt(final/receipt, digest(final/'manifest.json'))
    tabular_values = pd.read_csv(tabular/'values.csv').query("pattern == 'overall'").copy()
    tabular_values = tabular_values.rename(columns={'nrmse': 'value', 'ci95_low': 'ci_lower', 'ci95_high': 'ci_upper', 'datasets': 'units'})
    tabular_values = tabular_values.assign(domain='tabular', metric='RMSE', dataset=tabular_values.suite.map({'missbench': 'MissBench', 'uci': 'UCI'}))
    recsys, recsys_comparison = rating_values(source, final, manifest)
    causal, causal_comparison = causal_values(source, final, OLD_CAUSAL)
    columns = ['domain', 'dataset', 'metric', 'method', 'label', 'value', 'ci_lower', 'ci_upper', 'units', 'available']
    values = pd.concat([tabular_values[columns], recsys[columns], causal[columns]], ignore_index=True)
    if len(values) != 84 or values.duplicated(['domain', 'dataset', 'metric', 'method']).any():
        raise ValueError('Unexpected figure inventory')
    comparisons = pd.concat([recsys_comparison, causal_comparison], ignore_index=True)
    comparisons['change_percent'] = 100*(comparisons.final/comparisons.previous-1)
    output.mkdir(parents=True, exist_ok=True)
    specification = importlib.util.spec_from_file_location('preview_style', source/'scripts/figure_style.py')
    style = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(style)
    draw(values, output, style, tabular_note)
    values.to_csv(output/'values.csv', index=False)
    comparisons.to_csv(output/'checkpoint_comparison.csv', index=False)
    if paper_snapshot(PAPER) != before:
        raise ValueError('Paper files changed during generation')
    write(output/'provenance.json', dict(paper_unchanged=True, final_model=manifest['model'],
          value_decimals=VALUE_DECIMALS, value_label_size_points=VALUE_LABEL_SIZE,
          attachment_matches_source=digest(ATTACHMENT) == digest(source/'figures/four_quadrant_summary_sorted.png'),
          source_sha256={str(path): digest(path) for path in inputs},
          orientation='standard', tabular=tabular_note + '; held-out-SD normalized column RMSE',
          recsys='Mean per-test-user RMSE; same split and baseline predictions as attachment',
          causal='Raw outcome-unit errors; equal design weights; same draws and baselines as attachment; CATE RMSE uses mu0',
          intervals={'tabular': '1000 seed-0 dataset bootstrap', 'recsys': '1000 seed-0 test-user bootstrap',
                     'causal': '1.96 times within-design standard errors combined over equally weighted designs, as in attachment'},
          missing_policy='No partial-cohort bars', mice_retry=tabular_audit.get('mice_retry'),
          common_exclusions=tabular_audit.get('common_exclusions', []),
          non_tabular_baselines='Same as attachment; no ensemble or specialists',
          source_figure_script=str(source/'scripts/assemble_four_quadrant_figure.py')))
    print(comparisons.to_string(index=False))
    print(output/'four_quadrant_summary_sorted.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--final', type=Path, default=FINAL)
    parser.add_argument('--tabular', type=Path, default=TABULAR)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    generate(*(path.resolve() for path in (arguments.source, arguments.final, arguments.tabular, arguments.output)))


if __name__ == '__main__':
    main()
