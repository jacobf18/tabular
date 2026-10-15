"""Independently rescore synthetic predictions and export RMSE figures/tables."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import t as student_t

from recsys_evaluation_common import sha256, write_csv, write_json
from synthetic_low_rank_benchmark import (
    LABELS, METHODS, array_digest, generate_case, score_prediction,
)


GROUP_KEYS = ('rows', 'cols', 'rank', 'noise_std', 'missing_rate')


def read(path):
    return json.loads(path.read_text())


def statistics(values):
    count = len(values)
    mean = float(np.mean(values)) if count else None
    deviation = float(np.std(values, ddof=1)) if count > 1 else None
    half_width = float(student_t.ppf(.975, count - 1) * deviation / np.sqrt(count)) if count > 1 else None
    return {'n': count, 'mean': mean, 'std': deviation,
            'ci95_low': mean - half_width if half_width is not None else None,
            'ci95_high': mean + half_width if half_width is not None else None}


def audit_results(output, manifest):
    records = []
    counts = {'ok': 0, 'failed': 0, 'pending': 0}
    hashes = {}
    for case in manifest['cases']:
        directory = output / 'cases' / case['case_id']
        inputs = directory / 'inputs.npz'
        if inputs.exists():
            input_sha = sha256(inputs)
            hashes[str(inputs.relative_to(output))] = input_sha
            with np.load(inputs, allow_pickle=False) as saved:
                signal, truth, observed, missing = (saved[name] for name in ('signal', 'truth', 'observed', 'missing'))
            generated = generate_case(case, manifest['protocol']['master_seed'])
            if array_digest(signal, truth, observed, missing) != array_digest(*generated):
                raise ValueError('Synthetic data/masks do not reproduce')
        for method in METHODS:
            path = directory / f'{method}.json'
            if not path.exists():
                counts['pending'] += 1
                continue
            if not inputs.exists():
                raise ValueError('Result without archived inputs')
            record = read(path)
            if record['method'] != method or any(record[key] != value for key, value in case.items()):
                raise ValueError('Result case/method mismatch')
            if record['inputs_sha256'] != input_sha or record['neural_updates'] != 0:
                raise ValueError('Result violates frozen input/evaluation-only protocol')
            hashes[str(path.relative_to(output))] = sha256(path)
            counts[record['status']] += 1
            if record['status'] == 'ok':
                prediction_path = directory / f'{method}.npz'
                if sha256(prediction_path) != record['predictions_sha256']:
                    raise ValueError('Prediction hash mismatch')
                hashes[str(prediction_path.relative_to(output))] = record['predictions_sha256']
                with np.load(prediction_path, allow_pickle=False) as saved:
                    values = saved['prediction']
                if values.shape != (int(missing.sum()),):
                    raise ValueError('Incomplete hidden-cell prediction coverage')
                prediction = observed.copy()
                prediction[missing] = values
                metrics, _ = score_prediction(truth, signal, observed, prediction)
                for name, value in metrics.items():
                    if (value is None and record[name] is not None) or (value is not None and
                            not math.isclose(value, record[name], rel_tol=1e-12, abs_tol=1e-12)):
                        raise ValueError(f'Saved {name} does not independently reproduce')
            records.append(record)
    return records, counts, hashes


def aggregate(records, cases):
    expected = defaultdict(int)
    groups = defaultdict(list)
    for case in cases:
        key = tuple(case[name] for name in GROUP_KEYS)
        expected[key] += 1
    for record in records:
        key = tuple(record[name] for name in GROUP_KEYS)
        groups[key, record['method']].append(record)
    summaries, paired = [], []
    for key, expected_count in sorted(expected.items()):
        common = dict(zip(GROUP_KEYS, key))
        plus = {record['seed']: record for record in groups[key, 'matrixformer_plus'] if record['status'] == 'ok'}
        for method in METHODS:
            selected = groups[key, method]
            valid = [record for record in selected if record['status'] == 'ok']
            rmse = statistics([record['rmse'] for record in valid])
            signal = statistics([record['signal_rmse'] for record in valid])
            summaries.append({**common, 'method': method, 'expected_n': expected_count,
                'failed_n': sum(record['status'] == 'failed' for record in selected),
                'pending_n': expected_count - len(selected), **{f'rmse_{name}': value for name, value in rmse.items()},
                **{f'signal_rmse_{name}': value for name, value in signal.items()}})
            if method != 'matrixformer_plus':
                differences = [record['rmse'] - plus[record['seed']]['rmse'] for record in valid if record['seed'] in plus]
                paired.append({**common, 'method': method, 'difference': 'baseline_minus_matrixformer_plus',
                    'expected_n': expected_count, **statistics(differences)})
    return summaries, paired


def plots(output, summaries, provisional):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination = output / 'figures'
    destination.mkdir(exist_ok=True)
    paths = []
    for rows, cols, noise in sorted({(row['rows'], row['cols'], row['noise_std']) for row in summaries}):
        selected = [row for row in summaries if (row['rows'], row['cols'], row['noise_std']) == (rows, cols, noise)]
        ranks = sorted({row['rank'] for row in selected})
        columns = min(3, len(ranks))
        figure, axes = plt.subplots(math.ceil(len(ranks) / columns), columns, squeeze=False,
                                   figsize=(5 * columns, 3.7 * math.ceil(len(ranks) / columns)))
        for axis, rank in zip(axes.flat, ranks):
            for method in METHODS:
                points = sorted([row for row in selected if row['rank'] == rank and row['method'] == method],
                                key=lambda row: row['missing_rate'])
                if not any(row['rmse_n'] for row in points):
                    continue
                missingness = [100 * row['missing_rate'] for row in points]
                means = [row['rmse_mean'] if row['rmse_mean'] is not None else np.nan for row in points]
                line, = axis.plot(missingness, means, marker='o', markersize=3,
                                  linewidth=2.4 if method == 'matrixformer_plus' else 1.2, label=LABELS[method])
                lower = [row['rmse_ci95_low'] if row['rmse_ci95_low'] is not None else np.nan for row in points]
                upper = [row['rmse_ci95_high'] if row['rmse_ci95_high'] is not None else np.nan for row in points]
                axis.fill_between(missingness, lower, upper, color=line.get_color(), alpha=.1)
            title = f'Rank {rank}' + (' (full-rank control)' if rank == min(rows, cols) else '')
            axis.set(title=title, xlabel='Missing entries (%)', ylabel='Hidden-entry RMSE')
            axis.grid(alpha=.2)
        for axis in list(axes.flat)[len(ranks):]:
            axis.set_visible(False)
        handles, labels = [], []
        for axis in axes.flat:
            current_handles, current_labels = axis.get_legend_handles_labels()
            for handle, label in zip(current_handles, current_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
        figure.suptitle(f'{"PROVISIONAL — " if provisional else ""}{rows} × {cols}, noise SD {noise:g}')
        figure.tight_layout(rect=(0, .10, 1, .96))
        filename = f'rmse_{rows}x{cols}_noise{noise:g}'.replace('.', 'p')
        for suffix in ('pdf', 'png'):
            figure.savefig(destination / f'{filename}.{suffix}', dpi=170)
        paths.append(destination / f'{filename}.pdf')
        plt.close(figure)
    return paths


def paper_files(output, manifest, summaries, figures, counts):
    destination = output / 'paper'
    destination.mkdir(exist_ok=True)
    profile = manifest['profile']
    provisional = bool(counts['pending']) or profile == 'smoke' or manifest['device'] == 'cpu'
    rows = [r'\begin{longtable}{llrrrlr}', r'\toprule',
            r'Shape & Method & Rank & $\sigma$ & Missing & RMSE $\pm$ SD & $n/N$ \\', r'\midrule\endhead']
    for value in summaries:
        estimate = '---' if value['rmse_mean'] is None else f"{value['rmse_mean']:.4f}"
        if value['rmse_std'] is not None:
            estimate += f" $\\pm$ {value['rmse_std']:.4f}"
        rows.append(f"{value['rows']}x{value['cols']} & {LABELS[value['method']]} & {value['rank']} & "
                    f"{value['noise_std']:g} & {100 * value['missing_rate']:g}\\% & {estimate} & "
                    f"{value['rmse_n']}/{value['expected_n']} " + r'\\')
    rows.extend([r'\bottomrule', r'\end{longtable}'])
    (destination / 'synthetic_tables.tex').write_text('\n'.join(rows) + '\n')
    content = [r'\subsection{Synthetic low-rank matrix completion}', r'\label{sec:synthetic-low-rank-' + profile + '}',
        'We evaluate the fixed MatrixFormer+ experts without neural training or checkpoint selection. '
        r'For each condition, $L=UV^\top/\sqrt{r}$ with independent standard Gaussian factors, and '
        r'$X=L+\sigma E$ with independent standard Gaussian noise. '
        'The expected signal variance is one at every rank; no hidden-value normalization enters inference.',
        'Uniform exact-count MCAR masks are shared by all methods and nested across missingness rates. '
        'Matrices with unsupported rows or columns are retained. We report cell-pooled RMSE on all hidden '
        'entries of $X$, then average across independent seed replicates; noise-free-signal RMSE is a separate diagnostic.',
        'MatrixFormer+ retains the observed-error convex blend with one weight per matrix, observed-column '
        'scaling, predictive means, and full matrix context. Its routing evidence remains in-sample. '
        'Column mean, K-NN, SoftImpute, and IterativeSVD receive the identical visible matrix. '
        'Nontrivial baseline hyperparameters use only a common support-preserving holdout of observations '
        'and are refit on the full visible matrix; neither hidden targets nor the generating rank enter selection.',
        r'Curves show per-condition means and 95\% Student-$t$ intervals across seeds, not pooled-cell '
        'uncertainty. Conditions share seeds and are not independent significance tests. '
        'Higher rank, extreme missingness, added noise, and rectangular shapes expose failure regimes. '
        'The observed-entry count divided by $r(m+n-r)$ is descriptive, not a recovery guarantee.',
        f'This export contains {counts["ok"]} successful, {counts["failed"]} failed, and {counts["pending"]} pending method--case evaluations. '
        'Tables retain $n/N$ coverage and missing results; comparisons with failures require matching the successful cases.']
    if provisional:
        content.insert(2, r'\textbf{Infrastructure/partial results only; not the completed paper experiment.}')
    for figure in figures:
        content.extend([r'\begin{figure}[t]\centering',
            r'\includegraphics[width=\linewidth]{results/synthetic_low_rank/' + profile + '/' + figure.name + '}',
            r'\caption{Synthetic hidden-entry RMSE across ranks and missingness. Shading is a per-condition 95\% interval across seeds; see the exported CSV for coverage and failures.}',
            r'\end{figure}'])
    (destination / 'synthetic_results.tex').write_text('\n\n'.join(content) + '\n')
    return provisional


def summarize(output, paper_dir=None):
    from evaluate_matrixformer_plus_synthetic import verify_manifest
    from threadpoolctl import threadpool_limits

    manifest = verify_manifest(output)
    with threadpool_limits(limits=manifest['threads']):
        records, counts, hashes = audit_results(output, manifest)
    summaries, paired = aggregate(records, manifest['cases'])
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired_differences.csv', paired)
    provisional = bool(counts['pending']) or manifest['profile'] == 'smoke' or manifest['device'] == 'cpu'
    figures = plots(output, summaries, provisional)
    paper_files(output, manifest, summaries, figures, counts)
    audit = {'passed': True, 'scope': 'All currently saved results; completeness reported separately',
        'complete': counts['pending'] == 0, 'provisional': provisional, **counts,
        'all_saved_metrics_independently_reproduced': True, 'neural_updates': 0,
        'manifest_sha256': sha256(output / 'manifest.json'), 'artifact_sha256': hashes,
        'summary_sha256': sha256(output / 'summary.csv'), 'source_sha256': sha256(Path(__file__))}
    write_json(output / 'summary_audit.json', audit)
    if paper_dir is not None:
        if provisional:
            raise ValueError('Do not publish smoke, CPU-diagnostic, or incomplete results into the paper')
        target = paper_dir / 'results/synthetic_low_rank' / manifest['profile']
        section = paper_dir / 'sections' / f'synthetic_{manifest["profile"]}_results.tex'
        if target.exists() or section.exists():
            raise ValueError('Do not overwrite existing paper results or section')
        target.mkdir(parents=True, exist_ok=False)
        for path in [output / 'summary.csv', output / 'paired_differences.csv', output / 'manifest.json',
                     output / 'summary_audit.json', output / 'paper/synthetic_tables.tex', *figures]:
            shutil.copyfile(path, target / path.name)
        shutil.copyfile(output / 'paper/synthetic_results.tex', section)
    print(json.dumps({'output': str(output), 'complete': audit['complete'], 'provisional': provisional, **counts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper-dir', type=Path)
    args = parser.parse_args()
    summarize(args.output.resolve(), args.paper_dir)
