"""Use unchanged count-shrunk retention in the complete V39 task/reference audit."""

import single_model_column_anchor_retention_training as predecessor
from single_model_registered_adaptation import replace_once


SOURCE = replace_once(predecessor.SOURCE, "import single_model_column_anchor_retention as retention",
    "import single_model_shrunk_column_anchor_retention as retention")
SOURCE = SOURCE.replace("V38", "V39")
exec(compile(SOURCE, __file__, "exec"), globals())
