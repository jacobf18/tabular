"""Preserve the accepted balanced-prior probe's complete positive and adverse evidence."""

import hashlib
import inspect
import os
from pathlib import Path

import recheck_single_model_all_imputation_column_mean_probe as blueprint


TEMPLATE_SHA256 = '3b0ca8a4bef988a13bcab2d41efa4ac47704b1738cb0a6666eee69283abf3917'
if hashlib.sha256(Path(blueprint.__file__).read_bytes()).hexdigest() != TEMPLATE_SHA256:
    raise ValueError('Keep the original completed mean-probe recheck unchanged')
RECHECK_SOURCE = inspect.getsource(blueprint)
for before, after in (
    ('import preflight_single_model_all_imputation_column_mean as probe',
     'import preflight_single_model_balanced_structured_components as probe'),
    ('669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11',
     'a5d077f51bffbaa44f1f577b19f61f0f7a53864b581d29e8993a3c84fcfa26ed'),
    ('# All-imputation column-mean probe: audited feasibility pass',
     '# Balanced structured components: audited feasibility pass'),
    ('The actual V29 initializer receives one fresh update. Extend native count-shrunk equal-column mean20 Huber weighting to all structured imputation tasks, uniformly across every family and component. Native and rating objectives, mixed priors, teachers, task masses and protected optimizer stay unchanged.',
     'The actual V29 initializer receives one fresh update. Balance the four original structured components across all twelve families, with six scheduled tasks and one-quarter of structured supervised weight each. Native rehearsal, ratings, all-imputation count-shrunk mean20, teachers, family/domain weights and protected optimizer stay unchanged. Requested geometry and missingness recipes remain fixed, but changing components can change actual intrinsic shapes and masks.'),
    ('def main():\n',
     'def main():\n'
     '    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or probe.torch.cuda.is_initialized():\n'
     '        raise ValueError("Require a CUDA-hidden saved-evidence recheck")\n'),
    ("    assert status['state'] == 'complete' and status['retention_audited']",
     "    assert config['structured_component_balance'] == probe.components.SPECIFICATION\n"
     "    assert config['batch'] == probe.batch()\n"
     "    assert Counter(task['specification']['structured']['component'] for task in config['batch']['tasks']\n"
     "        if 'structured' in task['specification']) == {component: 6 for component in probe.components.COMPONENTS}\n"
     "    assert status['state'] == 'complete' and status['retention_audited']"),
    ("    inventory = {}", "    inventory = {str(Path(blueprint.__file__).resolve()): TEMPLATE_SHA256}"),
):
    if before not in RECHECK_SOURCE:
        raise ValueError(f'Missing explicit balanced-prior recheck adaptation: {before}')
    RECHECK_SOURCE = RECHECK_SOURCE.replace(before, after)
exec(compile(RECHECK_SOURCE, __file__, 'exec'), globals())
