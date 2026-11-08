"""Reconstruct all saved standardized teacher-mean terms in the fresh quarter-step probe."""

from pathlib import Path

import audit_single_model_standardized_teacher_huber_probe as audit_blueprint
from single_model_registered_adaptation import replace_once


SOURCE = Path(audit_blueprint.__file__).read_text()
SOURCE = replace_once(SOURCE, 'import preflight_single_model_standardized_teacher_huber as probe',
    'import preflight_single_model_balanced_native_quarter as probe')
SOURCE = SOURCE.replace('test_single_model_standardized_teacher_huber_probe_audit.py',
    'test_single_model_balanced_native_quarter_probe_audit.py')
exec(compile(SOURCE, __file__, 'exec'), globals())
