"""Preserve three closed nonselected artifacts for the family-balanced retention trial."""

import inspect

import relocate_single_model_legacy_blend_capacity as blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


OWNERS_SPEC = {
    'cohort_anchor_retention_v42': {'kind': 'unselected_tail', 'run': 'cohort_anchor_retention_v42',
        'step': 64, 'updates': 58, 'selected': 0, 'grids': 1368, 'view_flag': 'full_network_training'},
    'cohort_anchor_retention_preflight_v1': {'kind': 'vector', 'parameters': 77998496,
        'frozen': 16, 'tasks': 28, 'grids': 56},
    'family_balanced_retention_preflight_v1': {'kind': 'vector', 'parameters': 77998496,
        'frozen': 16, 'tasks': 28, 'grids': 56},
}
POLICY_TEXT = ('Preserve exactly V42\'s completely audited unselected step64 optimizer tail and '
    'the two completed cohort-reference/family-balanced feasibility vectors. Preserve every byte, '
    'canonical path, source, metric and adverse result. The current family-balanced probe is complete '
    'and independently audited before its unselected vector may move; never warm-start a trial from '
    'either probe. No selected model, teacher, published checkpoint or live artifact moves. Require '
    'stopped producers and no scientific workers/open readers. Copy/fsync/hash/mmap before atomic '
    'canonical symlinks, independently reopen all three and verify all unchanged inputs. Preserve '
    '18/6/50 GiB reserves and restore18 GiB plus2432 MiB minimum training admission. UUID-bound '
    'scratch is machine-local, not provider-persistent backup. This storage-only registration '
    'authorizes no training, fresh seed, selection, confirmation or real benchmark evaluation.')

SOURCE = inspect.getsource(blueprint).split('\nif __name__ == "__main__":')[0]
SOURCE = SOURCE.replace('preflight_single_model_legacy_blend_teacher as probe',
    'preflight_single_model_family_balanced_retention as probe')
SOURCE = SOURCE.replace('legacy_blend_capacity_relocation_v1', 'family_balanced_capacity_relocation_v1')
SOURCE = SOURCE.replace('single_model_goal_20261005_owned/legacy_blend_capacity_v1',
    'single_model_goal_20261005_owned/family_balanced_capacity_v1')
SOURCE = SOURCE.replace('test_single_model_legacy_blend_capacity.py', 'test_single_model_family_balanced_capacity.py')
SOURCE = SOURCE.replace('copying_completed_legacy_blend_capacity', 'copying_completed_family_balanced_capacity')
SOURCE = replace_assignment(SOURCE, 'OWNERS', OWNERS_SPEC)
SOURCE = replace_assignment(SOURCE, 'POLICY', POLICY_TEXT)
SOURCE = replace_assignment(SOURCE, 'PROBE_CONFIG_SHA256',
    '25dbd1d2782ce45af86648695205ec3655b76c3ad57e8140dbfd21fb6c07c816')
SOURCE = replace_once(SOURCE, '    probe.require_exact_predecessor()',
    '    probe.require_completed_v42()\n    probe.retention_records(probe.OUTPUT)')
SOURCE = replace_once(SOURCE, 'if (retention["reference_grids"] != 26 or current.get("state") != "complete"',
    'if (result["record"].get("accepted") is not True\n'
    '            or retention["reference_grids"] != 26 or len(retention["endpoint_penalties"]) != 26\n'
    '            or retention["reference_role_counts"] != {"v8": 4, "v29": 22}\n'
    '            or current.get("state") != "complete"')
exec(compile(SOURCE, __file__, 'exec'), globals())


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise ValueError('Invoke this tested one-shot preservation once without stage arguments')
    main()
