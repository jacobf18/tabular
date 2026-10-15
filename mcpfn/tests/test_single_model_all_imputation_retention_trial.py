import inspect

import test_single_model_fixed_half_shrunk_retention_trial as previous_tests


SOURCE = inspect.getsource(previous_tests).split("\ndef test_v39_changes_only_registered_retention")[0]
for old, new in {
    "train_single_matrixformer_fixed_half_shrunk_retention": "train_single_matrixformer_all_imputation_retention",
    "971100901": "1001100901", "972100901": "1002100901",
    "981100901": "1011100901", "982100901": "1012100901",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
