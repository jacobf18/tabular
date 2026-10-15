"""Independently replay V29 references for every V40 imputation task."""

import single_model_fixed_half_shrunk_retention_training as predecessor
from single_model_registered_adaptation import replace_once


SOURCE = replace_once(predecessor.SOURCE, "import single_model_shrunk_column_anchor_retention as retention",
    "import single_model_all_imputation_retention as retention")
SOURCE = SOURCE.replace("V39", "V40").replace("native-anchor retention", "all-imputation retention")
exec(compile(SOURCE, __file__, "exec"), globals())
