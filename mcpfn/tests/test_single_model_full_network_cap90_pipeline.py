import inspect

import test_single_model_full_network_pipeline as template


SOURCE = inspect.getsource(template)
for old, new in {
    "import run_single_model_full_network_pipeline as pipeline": "import run_single_model_full_network_cap90_pipeline as pipeline",
    "full_network_v34": "full_network_cap90_v35",
    "run_single_model_full_network_pipeline.py": "run_single_model_full_network_cap90_pipeline.py",
    "train_single_matrixformer_full_network.py": "train_single_matrixformer_full_network_cap90.py",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
