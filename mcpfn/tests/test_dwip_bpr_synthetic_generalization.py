from types import SimpleNamespace

import pytest
import torch

from diagnose_dwip_bpr_synthetic_generalization import synthetic_context_references, synthetic_ranking_utilities, tensor_digest


def episode():
    relevance = torch.tensor([[[.8, .1, .2, 0.], [0., .5, .9, .3], [.2, .2, .2, .2]]])
    query = torch.ones_like(relevance, dtype=torch.bool)
    selected = torch.zeros_like(query)
    selected[:, :, 0] = True
    selected[:, 2, :] = True
    return SimpleNamespace(source=torch.full_like(relevance, torch.nan), query=query, rating_query=selected,
                           relevance_probabilities=relevance, global_col_support=torch.tensor([10, 2, 7, 1]),
                           metadata={"gradient_data": "synthetic"})


def test_ideal_synthetic_ranking_has_unit_expected_dcg_efficiency():
    data = episode()
    result = synthetic_ranking_utilities(1 + 4 * data.relevance_probabilities[data.query], data, top_k=2)
    assert result["ranking_users"] == 2
    assert result["expected_dcg_efficiency"] == pytest.approx(1.)
    assert result["expected_positive_mass_at_k"] == pytest.approx(1.2)
    assert result["ideal_expected_positive_mass_at_k"] == pytest.approx(1.2)
    assert result["expected_positive_mass_recall"] == pytest.approx((1 / 1.1 + 1.4 / 1.7) / 2)
    assert result["top_k_score_ceiling_fraction"] == 0


def test_inverted_ranking_is_worse_and_rating_only_users_do_not_contribute():
    data = episode()
    scores = 5 - 4 * data.relevance_probabilities[data.query]
    result = synthetic_ranking_utilities(scores, data, top_k=2)
    assert result["expected_dcg_efficiency"] < .5
    assert result["expected_positive_mass_at_k"] == pytest.approx(.2)
    scores[-4:] = 100
    assert synthetic_ranking_utilities(scores, data, top_k=2) == result


def test_clipping_and_stable_ties_are_used_on_synthetic_slates():
    data = episode()
    scores = torch.full((12,), 6.)
    scores[1] = 100
    result = synthetic_ranking_utilities(scores, data, top_k=1)
    assert result["expected_positive_mass_at_k"] == pytest.approx(.4)
    assert result["top_k_score_ceiling_fraction"] == 1


def test_missing_queries_have_no_probability_mass_or_topk_ceiling_slots():
    data = episode()
    data.query[:, :, 2:] = False
    data.rating_query &= data.query
    data.relevance_probabilities[:, :, 2:] = 1
    result = synthetic_ranking_utilities(torch.full((6,), 5.), data, top_k=10)
    assert result["ranking_users"] == 2
    assert result["expected_positive_mass_at_k"] == pytest.approx(.7)
    assert result["expected_positive_mass_recall"] == pytest.approx(1)
    assert result["top_k_score_ceiling_fraction"] == 1


@pytest.mark.parametrize("problem", ["real", "shape", "topk", "probability", "live_targets", "nonfinite", "zero_mass", "no_ranking_users", "batch", "mask_dtype", "rating_subset"])
def test_invalid_synthetic_cases_are_rejected(problem):
    data = episode()
    scores = torch.ones(12)
    top_k = 2
    if problem == "real":
        data.metadata["gradient_data"] = "real"
    elif problem == "shape":
        scores = scores[:-1]
    elif problem == "topk":
        top_k = 0
    elif problem == "probability":
        data.relevance_probabilities[0, 0, 0] = 2
    elif problem == "live_targets":
        data.relevance_probabilities.requires_grad_(True)
    elif problem == "nonfinite":
        scores[0] = torch.nan
    elif problem == "zero_mass":
        data.relevance_probabilities.zero_()
    elif problem == "no_ranking_users":
        data.rating_query.fill_(True)
    elif problem == "batch":
        data.source = data.source.expand(2, -1, -1)
    elif problem == "mask_dtype":
        data.query = data.query.float()
    else:
        data.query[0, 0, 0] = False
    with pytest.raises(ValueError):
        synthetic_ranking_utilities(scores, data, top_k=top_k)


def test_tensor_digest_preserves_nans_and_detects_input_mutation():
    data = episode()
    original = tensor_digest(vars(data))
    assert tensor_digest(vars(data)) == original
    data.relevance_probabilities[0, 0, 0] = .9
    assert tensor_digest(vars(data)) != original
    assert tensor_digest({"value": torch.tensor([0.])}) != tensor_digest({"value": torch.tensor([-0.])})


def test_count_references_use_only_masks_not_observed_values_and_do_not_mutate_inputs():
    data = episode()
    data.source[:, :, 1] = 1
    data.query[:, :, 1] = False
    data.rating_query &= data.query
    original = tensor_digest(vars(data))
    references = synthetic_context_references(data)
    assert tensor_digest(vars(data)) == original
    data.source[:, :, 1] = -100
    assert synthetic_context_references(data) == references
    assert references["local_item_context_zero_fraction"] == pytest.approx(.75)
    assert references["query_fraction_without_local_item_context"] == 1
    assert references["query_relevance_mass_without_local_item_context"] == 1


@pytest.mark.parametrize("counts", [None, torch.ones(3), torch.tensor([1., 2., 3., -1.]), torch.tensor([1., 2., 3., torch.nan])])
def test_count_references_reject_invalid_parent_counts(counts):
    data = episode()
    data.global_col_support = counts
    with pytest.raises(ValueError, match="parent item counts"):
        synthetic_context_references(data)
