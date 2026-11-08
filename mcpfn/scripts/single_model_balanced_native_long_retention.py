"""Preserve V55 retention computations over the prospectively extended attempt range."""

from pathlib import Path

import single_model_standardized_huber_training as template
import single_model_balanced_native_long_budget as budget
from single_model_registered_adaptation import replace_once


SOURCE = Path(template.__file__).read_text()
SOURCE = replace_once(SOURCE, "exec(compile(RETENTION_TRAINING_SOURCE, __file__, 'exec'), globals())",
    "RETENTION_TRAINING_SOURCE = budget.adapt_retention(RETENTION_TRAINING_SOURCE)\n"
    "exec(compile(RETENTION_TRAINING_SOURCE, __file__, 'exec'), globals())")
exec(compile(SOURCE, __file__, "exec"), globals())
