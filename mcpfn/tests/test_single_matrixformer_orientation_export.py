import json

import pytest

from export_single_matrixformer_orientation_results import CSV_FILES, verified_inputs
from recsys_evaluation_common import sha256, write_json


@pytest.fixture
def audited_run(tmp_path):
    write_json(tmp_path / 'manifest.json', {'checkpoint_sha256': 'checkpoint'})
    write_json(tmp_path / 'scope_override.json', {
        'excluded_datasets': {'recsys': ['10M']}, 'manifest_sha256': sha256(tmp_path / 'manifest.json')})
    (tmp_path / 'RESULTS.md').write_text('All three orientations retained.\n')
    for filename in CSV_FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('variant,mean\nstandard,1.0\n')
    audit = {
        'passed': True, 'prediction_artifacts': 4065, 'expected_artifacts': 4065,
        'manifest_sha256': sha256(tmp_path / 'manifest.json'),
        'scope_override_sha256': sha256(tmp_path / 'scope_override.json'),
        'report_sha256': sha256(tmp_path / 'RESULTS.md'),
        'output_sha256': {filename: sha256(tmp_path / filename) for filename in CSV_FILES},
        'included_rating_datasets': ['100K', 'netflix', '1M'],
        'all_average_predictions_exact': True, 'all_standard_predictions_unchanged': True,
        'all_metrics_rescored': True,
    }
    write_json(tmp_path / 'audit.json', audit)
    write_json(tmp_path / 'status.json', {
        'state': 'complete_revised_scope', 'audit_sha256': sha256(tmp_path / 'audit.json')})
    return tmp_path


def test_export_requires_the_complete_revised_scope(audited_run):
    audit, manifest = verified_inputs(audited_run)
    assert audit['prediction_artifacts'] == 4065
    assert manifest['checkpoint_sha256'] == 'checkpoint'
    write_json(audited_run / 'status.json', {'state': 'failed'})
    with pytest.raises(ValueError, match='completed, audited'):
        verified_inputs(audited_run)


@pytest.mark.parametrize('filename', ['manifest.json', 'scope_override.json', 'RESULTS.md', *CSV_FILES])
def test_export_rejects_modified_audited_files(audited_run, filename):
    with (audited_run / filename).open('a') as stream:
        stream.write('\nchanged\n')
    with pytest.raises(ValueError, match='Audited input changed'):
        verified_inputs(audited_run)


def test_export_requires_unchanged_standard_predictions(audited_run):
    audit = json.loads((audited_run / 'audit.json').read_text())
    audit['all_standard_predictions_unchanged'] = False
    write_json(audited_run / 'audit.json', audit)
    write_json(audited_run / 'status.json', {
        'state': 'complete_revised_scope', 'audit_sha256': sha256(audited_run / 'audit.json')})
    with pytest.raises(ValueError, match='Missing audit assertion'):
        verified_inputs(audited_run)
