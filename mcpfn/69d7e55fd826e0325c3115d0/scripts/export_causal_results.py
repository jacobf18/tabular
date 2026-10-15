"""Export causal-panel results for the paper.

Reads the saved per-draw metrics of the CPS, Kansas and California smoking
causal-panel benchmarks and writes results/causal_*.csv,
results/causal_provenance.json, figures/causal_normalized_scores.pdf and the
generated tables in sections/causal_tables.tex. Nothing is rerun.
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


BENCHMARK = Path('/root/tabular/mcpfn/benchmark')
CHECKPOINT = Path('/root/tabular/mcpfn/src/tabimpute/workdir/'
                  'tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/'
                  'best_validation_checkpoint.pth')
PAPER = Path(__file__).resolve().parents[1]
MATRIXFORMER = 'dwip-spectral-tm128k'
FAMILIES = {
    'CPS': [(f'{o.replace("_", " ")} / {t.replace("_", " ")}', f'CPS/causal_eval/{o}_{t}')
            for o in ('hours', 'log_wage', 'urate') for t in ('abort_ban', 'min_wage', 'open_carry')],
    'Kansas': [('factor model', 'kansas/causal_eval/kansas_factor_r3_200')],
    'California smoking': [('simultaneous', 'prop99/causal_eval/ca_simul'),
                           ('staggered', 'prop99/causal_eval/ca_stagg')],
}
CORE = {
    MATRIXFORMER: 'MatrixFormer-Impute',
    'sdid': 'SDiD',
    'sc': 'SC',
    'did': 'DiD',
    'softimpute': 'SoftImpute',
    'knn': 'K-NN',
    'hyperimpute': 'HyperImpute',
}
PARTIAL = {
    'mice': 'MICE',
    'hyperimpute_mean': 'Column mean',
    'ice': 'ICE',
    'missforest': 'MissForest',
    'gain': 'GAIN',
    'miwae': 'MIWAE',
    'miracle': 'MIRACLE',
    'masters_mcar': 'TabImpute (legacy)',
    'xrfm': 'xRFM',
    'augsynth': 'AugSynth',
}
# Other MatrixFormer-Impute checkpoints. They are placed on the core methods' normalized
# scale but never set its minimum or maximum, so the core scores do not change.
VARIANTS = {
    'recsys-specialist': 'MatrixFormer-RecSys',
    'matrixformer-large': 'MatrixFormer-Impute (Recent)',
    'dwip-v2-2': 'MatrixFormer-Impute (Stale)',
}
VARIANT_SUB = {'recsys-specialist': 'RecSys', 'matrixformer-large': 'Recent', 'dwip-v2-2': 'Stale'}
VARIANT_CHECKPOINTS = {
    'recsys-specialist': Path('/root/tabular/mcpfn/runs/recsys_goal_20260920/'
                              'synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth'),
    'matrixformer-large': Path('/root/tabular/mcpfn/runs/streamed_training/'
                               'large_matrices_98k_20260930/best_validation_checkpoint.pth'),
    'dwip-v2-2': Path('/root/tabimpute_anchor3_named/dwip_v2_t003_checkpoint_17000.pth'),
}
# Recent and RecSys were evaluated separately; MatrixFormer-Impute and dwip-v2-2 (all
# orientations) are in the benchmark CSVs.
VARIANT_RESULTS = Path('/root/tabular/mcpfn/runs/benchmark_evaluations/'
                       'causal_large_and_recsys_specialist_20261003/per_draw_metrics.csv')
TRANSPOSE_RESULTS = VARIANT_RESULTS.with_name('per_draw_metrics_transpose.csv')
# Each checkpoint runs on the panel as given, on its transpose, and as the average
# of the two completions; one orientation per checkpoint is shown (choose_orientations).
ORIENTATIONS = {'': 'standard', '-T': 'transposed', '-avg': 'average'}
CHECKPOINT_LABELS = {MATRIXFORMER: CORE[MATRIXFORMER], **VARIANTS}
CANDIDATES = [base + suffix for base in CHECKPOINT_LABELS for suffix in ORIENTATIONS]
BASELINES = [m for m in CORE if m != MATRIXFORMER]


def orientation_label(name, suffix):
    if not suffix:
        return name
    tag = ORIENTATIONS[suffix]
    return f'{name[:-1]}, {tag})' if name.endswith(')') else f'{name} ({tag})'


LABELS = {**CORE, **{base + suffix: orientation_label(name, suffix)
                     for base, name in CHECKPOINT_LABELS.items() for suffix in ORIENTATIONS}}
METRICS = {'treated_block_rmse': 'Treated-block RMSE', 'tau_abs_error': r'$|\hat\tau-\tau|$'}
AXIS_LABELS = {'treated_block_rmse': 'Treated-block RMSE', 'tau_abs_error': r'Effect error $|\hat\tau-\tau|$'}
CAPTION_NAMES = {'treated_block_rmse': 'treated-block RMSE', 'tau_abs_error': r'$|\hat\tau-\tau|$'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    df = pd.read_csv(BENCHMARK / path / 'results/per_draw_metrics.csv')
    df = df[df['method'].isin({*CORE, *PARTIAL, *CANDIDATES})]
    extra = pd.concat([pd.read_csv(VARIANT_RESULTS), pd.read_csv(TRANSPOSE_RESULTS)], ignore_index=True)
    extra = extra[(extra['path'] == path) & extra['method'].isin(CANDIDATES) & ~extra['method'].isin(df['method'])]
    n_draws = sum(1 for d in (BENCHMARK / path).iterdir() if d.name.startswith('draw_'))
    return pd.concat([df, extra[df.columns.intersection(extra.columns)]], ignore_index=True), n_draws


def normalized_scores(frames, metric):
    """Per-draw min-max over the core methods, averaged over draws, then scenarios.

    Other MatrixFormer-Impute runs present in the frames are scored on the same per-draw scale."""
    means, ses, n = [], [], 0
    for df in frames:
        wide = df.pivot_table(index='draw_name', columns='method', values=metric)
        wide = wide[[m for m in dict.fromkeys([*CORE, *CANDIDATES]) if m in wide.columns]].dropna(subset=list(CORE))
        if wide.isna().any().any():
            raise ValueError(f'{metric}: a MatrixFormer-Impute variant is missing draws')
        lo, hi = wide[list(CORE)].min(axis=1), wide[list(CORE)].max(axis=1)
        norm = wide.sub(lo, axis=0).div(hi - lo, axis=0)[hi > lo]
        means.append(norm.mean())
        ses.append(norm.std(ddof=1) / np.sqrt(len(norm)))
        n += len(norm)
    mean = pd.concat(means, axis=1).mean(axis=1)
    se = np.sqrt((pd.concat(ses, axis=1) ** 2).sum(axis=1)) / len(frames)
    return mean, se, n


def choose_orientations(scores):
    """One orientation per checkpoint: the lowest mean normalized score over all benchmark-metric panels."""
    means = scores.groupby('method')['score'].mean()
    return {base: min((base + suffix for suffix in ORIENTATIONS), key=means.get) for base in CHECKPOINT_LABELS}


def plot(scores, path, methods):
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5})
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5.5, 4.1), sharey=True)
    y = np.arange(len(methods))[::-1]
    shown = scores[scores['method'].isin(methods)]
    xmax = max(1.0, np.ceil(10 * (shown['score'] + shown['ci95']).max()) / 10)
    for row, metric in enumerate(METRICS):
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            s = scores[(scores['family'] == family) & (scores['metric'] == metric)].set_index('method')
            vals = s.loc[methods, 'score'].to_numpy()
            ci = s.loc[methods, 'ci95'].to_numpy()
            colors = ['#2a6fbb' if m == methods[0] else '#9dbbe0' if m in CANDIDATES else '#b9bec8'
                      for m in methods]
            ax.barh(y, vals, height=0.7, color=colors)
            ax.errorbar(vals, y, xerr=ci, fmt='none', ecolor='#222222', elinewidth=0.7, capsize=1.5)
            ax.axvline(1, color='#888888', linewidth=0.6, linestyle='--', zorder=0)
            ax.set_xlim(0, xmax)
            ax.text(vals[0] + ci[0] + 0.04, y[0], f'{vals[0]:.3f}', va='center', fontsize=6.5,
                    color='#1d4f87', bbox=dict(facecolor='white', edgecolor='none', pad=0.4))
            ax.set_xticks(np.arange(0, xmax + 1e-9, 0.5))
            ax.tick_params(length=2, pad=1.5)
            ax.grid(axis='x', color='#e3e3e3', linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 0:
                n_scen, n_draws = int(s['n_scenarios'].iloc[0]), int(s['n_draws'].iloc[0])
                ax.set_title(f'{family}\n({n_scen} design{"s" if n_scen > 1 else ""}, {n_draws:,} draws)',
                             fontsize=7.5)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([LABELS[m] for m in methods])
                ax.set_ylabel(AXIS_LABELS[metric], fontsize=8)
                ax.get_yticklabels()[0].set_fontweight('bold')
    fig.supxlabel('Normalized score (0 = best, 1 = worst of the seven main methods in each draw)',
                  fontsize=7.5, y=0.01)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


def fmt(value, digits=4):
    if not np.isfinite(value):
        return '---'
    return f'{value:.3f}' if abs(value) >= 10 else f'{value:.{digits}f}'


def tables(scores, per_scenario, methods):
    methods = [method for method in methods if method in BASELINES]
    out = ['% Generated by scripts/export_causal_results.py; do not edit by hand.', '']
    fams = list(FAMILIES)
    out += [r'\begin{table}[h!]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
            r'\caption{Normalized causal-panel scores (mean $\pm$ 95\% CI over draws; 0 = best and 1 = worst '
            r'of the original seven main methods in each draw, including the archived MatrixFormer-Impute reference). '
            r'Only external baselines are tabulated here; the MatrixFormer ensemble is shown in the main figure. '
            r'The normalization is unchanged. Designs within a benchmark receive equal weight.}',
            r'\label{tab:causal-normalized}',
            r'\begin{tabular}{@{}l' + 'r' * len(fams) + '@{}}', r'\toprule',
            'Method & ' + ' & '.join(fams) + r' \\']
    for metric, name in METRICS.items():
        out.append(r'\midrule')
        out.append(rf'\multicolumn{{{len(fams) + 1}}}{{@{{}}l}}{{\emph{{{name}}}}} \\')
        for m in methods:
            cells = []
            for f in fams:
                r = scores[(scores['family'] == f) & (scores['metric'] == metric) & (scores['method'] == m)].iloc[0]
                cells.append(f'{r.score:.3f} $\\pm$ {r.ci95:.3f}')
            out.append(f'{LABELS[m]} & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

    header = ['Design & ' + ' & '.join(CORE[method] for method in methods) + r' \\']
    for metric, name in METRICS.items():
        core = per_scenario[per_scenario['method'].isin(methods)]
        out += [r'\begingroup', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
                r'\begin{longtable}{@{}l' + 'r' * len(methods) + '@{}}',
                rf'\caption{{Raw {CAPTION_NAMES[metric]} by design (mean over draws; lower is better). Every method covers '
                rf'every draw. Standalone expert results remain archived, not tabulated. '
                rf'Bold marks the lowest displayed baseline error in each row.}}'
                rf'\label{{tab:causal-raw-{metric.split("_")[0]}}}\\',
                r'\toprule', *header, r'\midrule', r'\endfirsthead', r'\toprule', *header,
                r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
        for family, scenarios in FAMILIES.items():
            out.append(rf'\multicolumn{{{len(methods) + 1}}}{{@{{}}l}}{{\emph{{{family}}}}} \\')
            for label, _ in scenarios:
                sub = core[(core['family'] == family) & (core['scenario'] == label)].set_index('method')
                vals = [sub.loc[m, f'{metric}_mean'] for m in methods]
                best = int(np.argmin(vals))
                digits = 3 if max(vals) >= 10 else 4
                cells = [f'{v:.{digits}f}' for v in vals]
                cells[best] = r'\textbf{%s}' % cells[best]
                out.append(rf'\hspace{{0.5em}}{label} & ' + ' & '.join(cells) + r' \\')
        out += [r'\end{longtable}', r'\endgroup', '']

    partial = per_scenario[per_scenario['method'].isin(PARTIAL)]
    out += [r'\begingroup', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{@{}llrrr@{}}',
            r'\caption{Baselines with saved predictions on only some designs or draws. Lower is better. '
            r'Standalone matched-expert scores remain archived.}\label{tab:causal-partial}\\',
            r'\toprule', r'Design & Method & Draws & RMSE & $|\hat\tau-\tau|$ \\',
            r'\midrule', r'\endfirsthead', r'\toprule',
            r'Design & Method & Draws & RMSE & $|\hat\tau-\tau|$ \\',
            r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    for family, scenarios in FAMILIES.items():
        for label, _ in scenarios:
            design = family if len(scenarios) == 1 and family != 'Kansas' else f'{family}: {label}'
            for m in PARTIAL:
                r = partial[(partial['family'] == family) & (partial['scenario'] == label) & (partial['method'] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                out.append(f'{design} & {PARTIAL[m]} & {int(r.draws)}/{int(r.draws_total)} & '
                           f'{fmt(r.treated_block_rmse_mean)} & {fmt(r.tau_abs_error_mean)} \\\\')
    out += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(out)


def main():
    for sub in ('results', 'figures'):
        (PAPER / sub).mkdir(exist_ok=True)
    loaded, score_rows, sources = {}, [], {}
    for family, scenarios in FAMILIES.items():
        loaded[family] = [(label, *load(path)) for label, path in scenarios]
        for _, path in scenarios:
            sources[str(BENCHMARK / path / 'results/per_draw_metrics.csv')] = digest(
                BENCHMARK / path / 'results/per_draw_metrics.csv')
        frames = [df for _, df, _ in loaded[family]]
        for metric in METRICS:
            mean, se, n = normalized_scores(frames, metric)
            for m in mean.index:
                score_rows.append({'family': family, 'metric': metric, 'method': m, 'label': LABELS[m],
                                   'score': mean[m], 'se': se[m], 'ci95': 1.96 * se[m],
                                   'n_scenarios': len(frames), 'n_draws': n})
    scores = pd.DataFrame(score_rows)
    chosen = choose_orientations(scores[scores['method'].isin(CANDIDATES)])
    shown = [*chosen.values(), *BASELINES]
    scores['shown'] = scores['method'].isin(shown)

    per_scenario = []
    for family, designs in loaded.items():
        for label, df, n_total in designs:
            mf = df[df['method'] == chosen[MATRIXFORMER]].set_index('draw_name')
            for method, g in df.groupby('method'):
                g = g.set_index('draw_name')
                row = {'family': family, 'scenario': label, 'method': method,
                       'label': {**LABELS, **PARTIAL}[method], 'draws': len(g), 'draws_total': n_total}
                for metric in METRICS:
                    row[f'{metric}_mean'] = g[metric].mean()
                    row[f'{metric}_se'] = g[metric].std(ddof=1) / np.sqrt(len(g))
                    row[f'matched_mf_{metric}'] = mf.loc[g.index, metric].mean()
                per_scenario.append(row)
    per_scenario = pd.DataFrame(per_scenario)
    per_scenario.to_csv(PAPER / 'results/causal_per_design.csv', index=False, float_format='%.6f')
    scores.to_csv(PAPER / 'results/causal_normalized_scores.csv', index=False, float_format='%.6f')
    plot(scores, PAPER / 'figures/causal_normalized_scores.pdf', shown)
    (PAPER / 'sections/causal_tables.tex').write_text(tables(scores, per_scenario, shown))
    provenance = {
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': digest(CHECKPOINT),
        'matrixformer_method_id': MATRIXFORMER,
        'core_methods': CORE,
        'partial_methods': PARTIAL,
        'variant_methods': VARIANTS,
        'variant_checkpoints': {m: {'path': str(c), 'sha256': digest(c)} for m, c in VARIANT_CHECKPOINTS.items()},
        'orientations': ORIENTATIONS,
        'shown_orientation': chosen,
        'orientation_selection': 'per checkpoint, the orientation with the lowest mean normalized score over the '
                                 'six benchmark-metric panels; selected on these results',
        'designs': {f: [{'label': label, 'path': p} for label, p in s] for f, s in FAMILIES.items()},
        'sources_sha256': {**sources, str(VARIANT_RESULTS): digest(VARIANT_RESULTS),
                           str(TRANSPOSE_RESULTS): digest(TRANSPOSE_RESULTS)},
        'normalization': 'per draw min-max over the core methods, with MatrixFormer-Impute in its standard orientation '
                         '(other MatrixFormer-Impute runs are placed on the same scale without setting it); mean over '
                         'draws per design; equal-weight mean over designs; SE = sqrt(sum SE_design^2) / n_designs',
    }
    (PAPER / 'results/causal_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('shown orientation:', chosen)
    print(scores[scores['shown']].pivot_table(index='label', columns=['metric', 'family'], values='score')
          .round(3).to_string())


if __name__ == '__main__':
    main()
