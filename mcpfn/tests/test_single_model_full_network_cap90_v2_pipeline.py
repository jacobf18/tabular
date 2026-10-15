import test_single_model_full_network_cap90_pipeline as predecessor


SOURCE = predecessor.SOURCE
for old, new in {
    "import run_single_model_full_network_cap90_pipeline as pipeline": "import run_single_model_full_network_cap90_v2_pipeline as pipeline",
    "full_network_cap90_v35": "full_network_cap90_v36",
    "run_single_model_full_network_cap90_pipeline.py": "run_single_model_full_network_cap90_v2_pipeline.py",
    "train_single_matrixformer_full_network_cap90.py": "train_single_matrixformer_full_network_cap90_v2.py",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_policy_names_the_current_fresh_trial():
    assert "V36 is a separately registered fresh-seed scientific trial" in pipeline.POLICY
