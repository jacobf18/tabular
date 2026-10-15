import inspect

import test_single_model_full_network_trial as template


SOURCE = inspect.getsource(template)
for old, new in {
    "import train_single_matrixformer_full_network as training": "import train_single_matrixformer_full_network_cap90 as training",
    "811100901, 812100901": "831100901, 832100901",
    "821100901, 822100901": "841100901, 842100901",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_only_allocator_changes_in_base_recipe_and_memory_evidence_is_mandatory():
    assert training.REGISTER_SOURCE.count('base["memory_fraction"] = .9') == 1
    assert '"tests/test_single_model_full_network_cap90_evidence.py"' in training.REGISTER_SOURCE
    assert training.require_evidence.__globals__["memory_evidence"] is training.memory_evidence
    assert training.PARENT == training.predecessor.PARENT
    assert training.TRAINABLE_PARAMETERS == training.predecessor.TRAINABLE_PARAMETERS
    assert training.TRIAL_QUALIFICATION == training.predecessor.TRIAL_QUALIFICATION
    assert training.BOUNDARIES == training.predecessor.BOUNDARIES
    before = training.original.alignment.fixed.seeds_in([training.predecessor.batches(), training.predecessor.confirmation_cases()])
    after = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(after) == 3868 and not before.intersection(after)
