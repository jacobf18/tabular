from collections import Counter
from copy import deepcopy
import random

import pytest

import train_single_matrixformer_family_rehearsal as training


def test_schedule_is_deterministic_private_balanced_and_complete():
    random.seed(12)
    state = random.getstate()
    cases = training.training_cases()
    assert random.getstate() == state
    assert cases == training.training_cases()
    assert [row["step"] for row in cases] == list(range(2, 401, 2))
    all_pairs = {(family, component) for family in training.mixture.FAMILIES for component in training.mixture.COMPONENTS}
    focused_pairs = {(family, component) for family in training.FOCUS for component in training.mixture.COMPONENTS}
    for subset, expected, cycle_length in ((cases[::2], all_pairs, 48), (cases[1::2], focused_pairs, 8)):
        pairs = [(row["case"]["family"], row["case"]["component"]) for row in subset]
        for start in range(0, len(subset) - cycle_length + 1, cycle_length):
            assert set(pairs[start:start + cycle_length]) == expected
        counts = Counter(pairs)
        assert max(counts.values()) - min(counts.values()) <= 1
    assert len({row["case"]["seed"] for row in cases}) == 200


def test_episode_seed_and_private_shape_draws_remain_per_step():
    for row in training.training_cases():
        case = row["case"]
        seed = training.mixture.SEEDS["data_seed"] + 16 * row["step"]
        assert case["seed"] == seed
        chooser = random.Random(seed)
        shapes = training.mixture.COMPACT_SHAPES if case["family"] in training.mixture.NEURAL_FAMILIES else training.mixture.SPECTRAL_SHAPES
        assert case["shape"] == list(chooser.choice(shapes))
        assert case["regime"] == chooser.choice(("legacy", "area_scaled"))
        assert case["requested_missingness"] == chooser.choice((.2, .6))


def test_recipe_only_changes_prespecified_prior_sampling():
    base = {"version": 10, "steps": 400, "parent_checkpoint": "selected_v8", "input_sha256": {"input": "fixed"},
            "source_sha256": {"source": "fixed"}, "experiment": "original", "structured_policy": deepcopy(training.CONTROL_POLICY),
            "training_structured_cases": [], "validation_cases": [{"seed": 123}], "loss": {"weight": 1.}}
    candidate = training.make_config(base)
    training.validate_pair(candidate, base)
    for key, value in (("steps", 200), ("validation_cases", [{"seed": 124}]), ("loss", {"weight": 2.}),
                       ("parent_checkpoint", "unselected_tail"), ("input_sha256", {}), ("source_sha256", {}),
                       ("family_rehearsal", {}), ("training_structured_cases", []), ("structured_policy", {})):
        changed = deepcopy(candidate)
        changed[key] = value
        with pytest.raises(ValueError):
            training.validate_pair(changed, base)


def test_process_local_policy_restores_after_exception():
    original = training.mixture.POLICY
    with pytest.raises(RuntimeError):
        with training.policy_runtime():
            assert training.mixture.POLICY == training.POLICY
            assert training.mixture.POLICY["retention"] == original["retention"]
            raise RuntimeError("test")
    assert training.mixture.POLICY is original
