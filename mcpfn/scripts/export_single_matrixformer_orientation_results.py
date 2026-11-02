"""Export audited orientation ablations without changing primary paper scores."""

import argparse
import json
import shutil
from pathlib import Path

from recsys_evaluation_common import sha256, write_json


OUTPUT = Path('/mnt/matrixformer_scratch/single_matrixformer_orientation_20261102')
CSV_FILES = ('tabular/per_case.csv', 'tabular/per_dataset.csv', 'tabular/summary.csv',
             'recsys/summary.csv', 'llm/per_fold.csv', 'llm/panels.csv', 'llm/summary.csv')


def verified_inputs(run):
    status = json.loads((run / 'status.json').read_text())
    audit = json.loads((run / 'audit.json').read_text())
    if (status.get('state') != 'complete_revised_scope'
            or status.get('audit_sha256') != sha256(run / 'audit.json')
            or audit.get('passed') is not True
            or audit.get('prediction_artifacts') != 4065 or audit.get('expected_artifacts') != 4065):
        raise ValueError('Require a completed, audited 4,065-case campaign')
    hashes = {'manifest.json': audit['manifest_sha256'], 'scope_override.json': audit['scope_override_sha256'],
              'RESULTS.md': audit['report_sha256'],
              **{name: audit['output_sha256'][name] for name in CSV_FILES}}
    for filename, expected in hashes.items():
        if sha256(run / filename) != expected:
            raise ValueError(f'Audited input changed: {filename}')
    scope = json.loads((run / 'scope_override.json').read_text())
    if (scope.get('excluded_datasets') != {'recsys': ['10M']}
            or scope.get('manifest_sha256') != audit['manifest_sha256']
            or audit.get('included_rating_datasets') != ['100K', 'netflix', '1M']):
        raise ValueError('Expected only MovieLens 10M to be excluded')
    for assertion in ('all_average_predictions_exact', 'all_standard_predictions_unchanged', 'all_metrics_rescored'):
        if audit.get(assertion) is not True:
            raise ValueError(f'Missing audit assertion: {assertion}')
    return audit, json.loads((run / 'manifest.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=OUTPUT)
    parser.add_argument('--paper-root', type=Path, default=Path('/root/MatrixFormerPaper'))
    args = parser.parse_args()
    audit, manifest = verified_inputs(args.run)
    destination = args.paper_root / 'results/single_model/orientation'
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (*CSV_FILES, 'audit.json', 'scope_override.json'):
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.run / filename, target)
    text = '''# Combined MatrixFormer: exploratory orientation ablations

Standard orientation remains the primary result in every domain. These exports
retain standard, transpose and arithmetic prediction averaging without selecting
a variant from test outcomes. They do not overwrite the primary result CSVs.
All benchmarks have been inspected; these are exploratory ablations, not a new
untouched evaluation or a claim that inference choices were preregistered.

## Coverage and protocol

- MissBench: 42 datasets × 11 patterns, whole-matrix inference.
- UCI: 12 datasets × 3 patterns × 5 rates × 10 repeats, at most 2,000 × 1,000 windows.
- Ratings: MovieLens 100K, Netflix small and MovieLens 1M, the original rating splits.
- LLM: all 1,800 fold settings; plots average greedy-MI budgets 5/10/15 within folds.
  All methods/budgets and per-panel scores are retained in the exported CSVs.
- MovieLens 10M was cancelled at the user's request because it is not used in the
  paper. Partial window accumulators remain in the research archive and are not scored.

All orientations share the same checkpoint, masks, splits and scoring. Standard
predictions are bit-identical to the archived originals. Tabular/LLM inference
freezes original-axis observed-column normalization before transposing. Ratings
retain original two-way biases and user scaling. UCI/rating window memberships
change with orientation. Average means aligned predictions, not averaged metrics.
Ratings clip each orientation before averaging; LLM averages before scorer clipping.

The causal orientation ablation is separate (`../causal/`): it refits observed-column
normalization in each orientation. Its averaging benefit therefore does not isolate
the same scale-preserving intervention as these cross-domain ablations. No universal
variance-reduction claim is made. The causal main comparison restores its original
standard-primary protocol; intermediate post-evaluation reporting history is retained.

## Uncertainty and results

The saved CSVs include paired percentile-bootstrap 95% intervals and differences
from standard: 10,000 dataset resamples for tabular, 1,000 user resamples for ratings,
and 1,000 fold resamples for LLM, with common draws across orientations and seed 0.
Settings are averaged within each independent resampling unit first. LLM folds
share context; intervals are conditional and do not adjust for selecting a variant.
Tabular intervals use 42/12 datasets, not dataset-pattern pairs. Causal controls
retain their separate equal-design-weighted, draw-level normal intervals.

Averaging increases overall tabular NRMSE on both suites, improves MovieLens 1M
RMSE slightly but not its MAE, and increases Netflix errors. For greedy-MI budgets
5/10/15, it improves MMLU and Merged R² but lowers MTEB and BenchPress R². Across
all LLM selections/budgets, MTEB's average R² improves instead. These different
summaries are both retained, rather than choosing the favorable one.

## Reproduce

From the research repository:
`python mcpfn/scripts/export_single_matrixformer_orientation_results.py`

From the paper repository:
`python scripts/plot_single_model_orientation_results.py`

The original full manifest, predictions and complete source snapshot remain at
`/mnt/matrixformer_scratch/single_matrixformer_orientation_20261102/`.
`export_manifest.json` binds this export to that manifest and the completed audit.
'''
    (destination / 'README.md').write_text(text)
    write_json(destination / 'export_manifest.json', {
        'evaluation_directory': str(args.run), 'evaluation_manifest_sha256': audit['manifest_sha256'],
        'evaluation_audit_sha256': sha256(args.run / 'audit.json'),
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'exporter_sha256': sha256(Path(__file__)),
        'primary_orientation': 'standard', 'reporting_role': 'exploratory ablation; no best-variant selection',
        'excluded_datasets': {'recsys': ['10M']}, 'prediction_artifacts': 4065,
        'files': {str(path.relative_to(destination)): sha256(path) for path in destination.rglob('*')
                  if path.is_file() and path.name not in ('export_manifest.json', 'figure_manifest.json')},
    })
    print(f'Exported all three orientations to {destination}; primary results unchanged')


if __name__ == '__main__':
    main()
