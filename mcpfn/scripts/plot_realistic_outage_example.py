"""Show recorded sensor-gap geometry and a prespecified reconstruction example."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns


def make_plot(root, output, identifier, baseline, controls):
    protocol = json.loads((root / 'protocol.json').read_text())
    case = next(record for record in protocol['cases'] if record['identifier'] == identifier)
    metadata = protocol['datasets'][case['dataset']]
    arrays = np.load(root / f'{identifier}.npz')
    truth, missing = arrays['truth'], arrays['missing']
    labels = metadata['columns'][:-3]
    states = np.where(~np.isfinite(truth[:, :-3]), 1, 0)
    states[missing[:, :-3]] = 2
    target = metadata['targets'][0]
    sns.set_theme(style='whitegrid', font='STIXGeneral', font_scale=.85)
    plt.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True,
                               gridspec_kw={'height_ratios': [1.1, 1.]})
    axes[0].imshow(states.T, origin='upper', aspect='auto', interpolation='nearest',
                   extent=(0, len(truth) / 24, len(labels) - .5, -.5),
                   cmap=ListedColormap(['#f6f6f6', '#b9bec8', '#2A6FBB']), vmin=0, vmax=2)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axes[0].grid(False)
    axes[0].set_title('Recorded sensor gaps replayed on a separate measured window', pad=30)
    axes[0].legend(handles=[Patch(color=color, label=label) for color, label in
                            [('#f6f6f6', 'Observed'), ('#b9bec8', 'Originally unavailable'),
                             ('#2A6FBB', 'Withheld for evaluation')]],
                   loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=7, frameon=True)
    days = np.arange(len(truth)) / 24
    axes[1].plot(days, truth[:, target], color='#252525', linewidth=1., alpha=.6, label='Measured reference')
    for method, color, linestyle in [('MatrixFormer', '#2A6FBB', '-'),
                                     ('MatrixFormer-Impute', '#174778', ':'), (baseline, '#939ba8', '--')]:
        filename = f'{identifier}_{method.replace(" ", "_")}.npz'
        path = next((folder / 'records' / filename for folder in [root] + controls
                     if (folder / 'records' / filename).exists()), None)
        if path is None:
            raise FileNotFoundError(filename)
        prediction = np.load(path)['prediction']
        displayed = np.where(missing[:, target], prediction[:, target], np.nan)
        axes[1].plot(days, displayed, color=color, linestyle=linestyle, linewidth=1.3, label=method)
    axes[1].set_xlabel(f'Days since {case["target_date"][:10]}')
    axes[1].set_ylabel('CO (mg/m³)' if case['dataset'] == 'air_quality' else 'PM₂.₅ (µg/m³)')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, fontsize=7)
    axes[1].set_xlim(0, len(truth) / 24)
    axes[1].grid(axis='x', visible=False)
    sns.despine(fig=figure)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(output.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', default='air_quality_confirmation_00')
    parser.add_argument('--baseline', default='SoftImpute')
    parser.add_argument('--controls', nargs='+', type=Path, default=[])
    arguments = parser.parse_args()
    make_plot(arguments.run, arguments.output, arguments.case, arguments.baseline, arguments.controls)
