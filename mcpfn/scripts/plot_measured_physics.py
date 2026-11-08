"""Create manuscript-style bar plots from audited measured-physics reports."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from report_measured_physics import seed_summary


REPLACEMENTS = {'SoftImpute (converged)': 'SoftImpute', 'HaLRTC (extended)': 'HaLRTC',
                'TNN (extended)': 'TNN'}
BLUE = '#2A6FBB'
NEUTRAL = '#b9bec8'


def apply_style():
    sns.set_theme(style='white', font='STIXGeneral', font_scale=.7)
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.bbox': None, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def curated(frame):
    frame = frame.copy()
    for extended, original in REPLACEMENTS.items():
        if extended in set(frame.method):
            frame = frame[frame.method != original]
            frame.loc[frame.method == extended, 'method'] = original
    return frame


def plot_bars(axis, summary, order, title):
    summary = summary.set_index('method').loc[order]
    coordinates = np.arange(len(order))
    colors = [BLUE if name.startswith('MatrixFormer') else NEUTRAL for name in order]
    axis.barh(coordinates, summary['mean'], height=.68, color=colors, edgecolor='none')
    errors = summary.ci95_half_width.to_numpy()
    if np.isfinite(errors).all():
        axis.errorbar(summary['mean'], coordinates, xerr=errors, fmt='none', ecolor='#222222',
                      elinewidth=.8, capsize=2, capthick=.8)
    axis.set_yticks(coordinates, order)
    axis.set_ylim(len(order) - .5, -.5)
    axis.set_title(title)
    axis.set_xlabel('RMSE $\downarrow$')
    axis.set_xlim(left=0)
    axis.grid(axis='x', color='#e3e3e3', linewidth=.5)
    axis.set_axisbelow(True)
    axis.tick_params(length=2, pad=1.5)
    sns.despine(ax=axis)


def save(figure, destination):
    figure.savefig(destination.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    figure.savefig(destination.with_suffix('.png'), dpi=240)
    plt.close(figure)


def run(report, output, dataset, hide_specialist=False):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(report / 'scores.csv')
    frame = curated(frame[frame.dataset == dataset])
    if hide_specialist:
        frame = frame[frame.method != 'MatrixFormer-Impute']
    if frame.empty:
        raise ValueError('No records for requested measured dataset')
    counts = frame.groupby(['method', 'rate']).seed.nunique()
    if counts.nunique() != 1:
        raise ValueError('Do not plot unequal method coverage')
    order = frame.groupby('method').nrmse.mean().sort_values().index.tolist()
    apply_style()
    sparse = seed_summary(frame[frame.rate >= .8], ['method'])
    figure, axis = plt.subplots(figsize=(5.5, 2.55))
    plot_bars(axis, sparse, sparse.sort_values('mean').method.tolist(),
              'Measured spectra: 80% and 95% missing entries')
    figure.tight_layout(pad=.4)
    save(figure, output / f'{dataset}_sparse')
    figure, axes = plt.subplots(1, 3, figsize=(5.5, 2.55), sharey=True)
    for axis, rate in zip(axes, (.5, .8, .95)):
        summary = seed_summary(frame[frame.rate == rate], ['method'])
        plot_bars(axis, summary, order, f'{rate:.0%} missing')
    figure.tight_layout(pad=.4, w_pad=.4)
    save(figure, output / f'{dataset}_all_rates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--hide-specialist', action='store_true')
    args = parser.parse_args()
    run(args.report, args.output, args.dataset, args.hide_specialist)
