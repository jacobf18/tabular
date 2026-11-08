"""Prospective quarter-step full-trial integration, gated by separate registration."""

from pathlib import Path

import single_model_fixed_half_trial as half_blueprint
from single_model_registered_adaptation import replace_assignment


QUARTER_POLICY = (
    'Prospective full-trial integration of the fixed quarter projected-displacement rule. '
    'Retain the original AdamW proposal and moments, all30 finite guards and exact rollback. '
    'Save both full proposal vectors and independently reconstruct the actual contracted '
    'candidate on GPU before replaying its predictions; the separate CPU replay audit binds '
    'that candidate exactly to the raw quarter vector even after rejection. Use explicit '
    'fixed_quarter_step and fixed_quarter_training audit keys, never half-step labels. '
    'Structured half-weight V8/V29 retention is unchanged and distinct from contraction. '
    'Initialize only from actual V29, never a feasibility endpoint. Preserve all22 development '
    'guards,32 fresh confirmation comparisons and every frozen real benchmark requirement. '
    'This helper alone authorizes no scientific seed, training, checkpoint selection or trial. '
    'Require a completed accepted fresh quarter probe with every original audit, independent '
    'storage preservation, a separately registered recipe and fresh streams. No fraction '
    'search, consumed-context retry, full historical autograd claim, fallback selection or '
    'qualification claim. A synthetic pass does not establish benchmark parity.'
)
SOURCE = Path(half_blueprint.__file__).read_text()
SOURCE = SOURCE.replace('single_model_fixed_half', 'single_model_fixed_quarter')
SOURCE = SOURCE.replace('fixed_half', 'fixed_quarter').replace('half_step', 'quarter_step')
SOURCE = SOURCE.replace('fixed-half', 'fixed-quarter')
SOURCE = replace_assignment(SOURCE, 'POLICY', QUARTER_POLICY)
exec(compile(SOURCE, __file__, 'exec'), globals())
