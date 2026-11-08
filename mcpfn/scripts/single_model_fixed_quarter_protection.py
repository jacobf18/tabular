"""Prospective quarter-displacement protection without changing proposals or acceptance guards."""

from pathlib import Path

import single_model_fixed_half_protection as half_blueprint
from single_model_registered_adaptation import replace_assignment


QUARTER_POLICY = (
    'Prospective optimizer hypothesis after the fully audited rejected native-coverage probe '
    'and complete saved-step decomposition. Keep the original half-composite/half-point '
    'AdamW proposal, native strata, paired structured/rating point protections,0.1 inward '
    'margin and2x Adam norm cap. Contract the rounded projected FP32 displacement ONCE '
    'by a fixed fraction0.25 instead of0.5, using FP64 before+0.25*(projected-before) '
    'followed by FP32 casting. This is not a smaller Adam learning rate: retain original '
    'proposal moments when accepted. Save both full uncontracted/contracted vectors and '
    'all84 recorded directional products; audit actual geometry after rounding and retain '
    'the unchanged finite30-guard decision and exact parameter/moment/rate rollback. '
    'The failed probe had a favorable native RMSE output tangent outweighed by a positive '
    'output-norm remainder; this motivates a smaller trust step but does not predict the '
    'behavior of a fresh network endpoint, establish a causal mechanism or guarantee '
    'acceptance. Never search fractions, reevaluate a consumed proposal, replace a draw, '
    'target a failing family or relax a guard. Smaller steps can slow learning and finite '
    'precision may defeat first-order predictions. Preserve priors, both orientations, '
    'teachers, loss/retention weights, V29 initialization, model/inference and all22 '
    'development guards,32 fresh confirmation comparisons and frozen real suites/full '
    'review. This helper authorizes no seed, GPU probe, training, checkpoint or full trial; '
    'require separate prospective registration, fresh streams and complete original audits.'
)

SOURCE = Path(half_blueprint.__file__).read_text()
SOURCE = replace_assignment(SOURCE, 'FRACTION', .25)
SOURCE = replace_assignment(SOURCE, 'POLICY', QUARTER_POLICY)
SOURCE = SOURCE.replace('half_step', 'quarter_step')
SOURCE = SOURCE.replace('prospectively fixed half displacement', 'prospectively fixed quarter displacement')
exec(compile(SOURCE, __file__, 'exec'), globals())
