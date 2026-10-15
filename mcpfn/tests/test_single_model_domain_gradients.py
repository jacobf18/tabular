from copy import deepcopy
import math

import pytest
import torch

import probe_single_model_domain_gradients as probe


def explicit_clip(vector):
    return vector * min(1., probe.CLIP / (float(torch.linalg.vector_norm(vector)) + 1e-6))


@pytest.mark.parametrize("imputation,rating", [([10., 0.], [0., 1.]), ([10., 0.], [-1., 1.]),
                                               ([.1, .2], [.2, .1]), ([0., 0.], [0., 0.]), ([1., 0.], [-1., 0.])])
def test_statistics_match_explicit_vectors(imputation, rating):
    first, second = torch.tensor(imputation, dtype=torch.float64), torch.tensor(rating, dtype=torch.float64)
    result = probe.geometry(first, second)
    vectors = {"joint_clip": explicit_clip((first + second) / 2),
               "domain_clip_then_mean": (explicit_clip(first) + explicit_clip(second)) / 2}
    for name, vector in vectors.items():
        stats = result["methods"][name]
        assert stats["direction_norm"] == pytest.approx(float(vector.norm()), abs=1e-8)
        assert stats["imputation_dot"] == pytest.approx(float(vector @ first), abs=1e-8)
        assert stats["rating_dot"] == pytest.approx(float(vector @ second), abs=1e-8)
        assert stats["direction_norm"] <= .5 + 1e-8


def test_gradient_capture_preserves_weights_and_requires_every_parameter():
    model = torch.nn.Linear(3, 2)
    original = deepcopy(model.state_dict())
    model(torch.ones(2, 3)).sum().backward()
    vector = probe.capture(model)
    assert vector.dtype == torch.float32 and vector.numel() == 8
    assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())
    model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match="Missing training gradient"):
        probe.capture(model)


def test_nonfinite_gradient_and_mismatched_inventory_are_rejected():
    with pytest.raises(FloatingPointError):
        probe.geometry(torch.tensor([float("nan")]), torch.tensor([1.]))
    with pytest.raises(ValueError):
        probe.geometry(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        probe.geometry(torch.ones(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError):
        probe.geometry(torch.ones(0), torch.ones(0))


@pytest.mark.parametrize("norm", [-1., float("nan"), float("inf")])
def test_invalid_clipping_norm_is_rejected(norm):
    with pytest.raises(ValueError):
        probe.coefficient(norm)


def test_cases_cover_complete_structured_cross_product_and_original_domains():
    cases = probe.specifications()
    assert len(cases) == 64
    structured = [case["imputation"]["structured"] for case in cases if case["group"] == "structured"]
    assert {(case["family"], case["component"]) for case in structured} == {
        (family, component) for family in probe.mixture.FAMILIES for component in probe.mixture.COMPONENTS}
    assert len(structured) == 48
    legacy = [case for case in cases if case["group"] == "legacy"]
    assert len(legacy) == 16
    assert {case["imputation"]["component"] for case in legacy} == set(range(8))
    assert {tuple(case["rating"]["shape"]) for case in cases} == {(1000, 1000), (2000, 1000)}
    seeds = [case["seed"] for case in structured] + [case["imputation"]["seed"] for case in legacy]
    seeds += [case["rating"]["seed"] for case in cases]
    assert len(set(seeds)) == 128
    for case in structured:
        shapes = probe.mixture.COMPACT_SHAPES if case["family"] in probe.mixture.NEURAL_FAMILIES else probe.mixture.SPECTRAL_SHAPES
        assert tuple(case["shape"]) in shapes


def test_summary_retains_undefined_cosines_and_every_family():
    records = []
    for case in probe.specifications():
        specification = case["imputation"].get("structured", {})
        records.append({**case, "metadata": {"imputation": specification},
                        "geometry": probe.geometry(torch.zeros(2), torch.zeros(2))})
    result = probe.summarize(records)
    assert result["all"]["cases"] == 64
    assert result["all"]["defined_cosines"] == 0
    assert result["all"]["median_cross_cosine"] is None
    assert all(result[family]["cases"] == 4 for family in probe.mixture.FAMILIES)
    assert not math.isnan(result["all"]["methods"]["joint_clip"]["median_rating_coefficient"])
    with pytest.raises(ValueError, match="all diagnostic groups"):
        probe.summarize([row for row in records if row["metadata"]["imputation"].get("family") != probe.mixture.FAMILIES[0]])
