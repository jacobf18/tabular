"""Train one V46 checkpoint with stronger teacher means and unchanged qualification."""

import ast

import train_single_matrixformer_all_imputation_blend as training_blueprint
import single_model_fixed_half_trial as fixed_trial
import single_model_stronger_mean_trial as all_trial
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = 'ea84bf2f6d088c32184ca71287591f76ccb2989da38d9396e703785dabb04279'
previous_training = training_blueprint
ADAPTATION_SOURCE = training_blueprint.ADAPTATION_SOURCE
for old, new in {
    'relocate_single_model_all_blend_capacity': 'relocate_single_model_stronger_mean_capacity',
    'all_imputation_blend_v45': 'stronger_mean_distillation_v46',
    '1151100901, 1152100901': '1181100901, 1182100901',
    '1161100901, 1162100901': '1191100901, 1192100901',
    'preflight_single_model_all_imputation_blend_teacher': 'preflight_single_model_stronger_mean_distillation',
    'audit_single_model_all_imputation_blend.py': 'audit_single_model_stronger_mean.py',
    'run_single_model_all_imputation_blend_pipeline.py': 'run_single_model_stronger_mean_pipeline.py',
    'test_single_model_all_imputation_blend_trial.py': 'test_single_model_stronger_mean_trial.py',
    'test_single_model_all_imputation_blend_audit.py': 'test_single_model_stronger_mean_full_audit.py',
    'test_single_model_all_imputation_blend_pipeline.py': 'test_single_model_stronger_mean_pipeline.py',
    'V45': 'V46',
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit stronger-mean full-trial adaptation: {old}')
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = replace_assignment(ADAPTATION_SOURCE, 'ANCHOR_POLICY', all_trial.POLICY)
node = next(node for node in ast.parse(ADAPTATION_SOURCE).body
    if isinstance(node, ast.FunctionDef) and node.name == 'retention_evidence')
RETENTION_EVIDENCE_SOURCE = ('def retention_evidence():\n'
    '    return all_trial.storage_evidence(capacity, CAPACITY_RECEIPT_SHA256)\n')
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(ast.get_source_segment(ADAPTATION_SOURCE, node), RETENTION_EVIDENCE_SOURCE)
BATCH_WRAPPER = training_blueprint.BATCH_WRAPPER
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
