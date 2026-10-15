"""Bind equal-column retention to every full-trial training task and reference replay."""

import inspect

import single_model_anchor_retention_training as predecessor
from single_model_registered_adaptation import replace_once


SOURCE = replace_once(inspect.getsource(predecessor),
    "import single_model_anchor_retention as retention", "import single_model_column_anchor_retention as retention")
SOURCE = SOURCE.replace("V37", "V38")
exec(compile(SOURCE, __file__, "exec"), globals())
