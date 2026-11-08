import numpy as np
import pandas as pd
import pytest
import torch

from matrixformer_supervised import SupervisedMatrixFormer, TabularPreprocessor
from tabimpute.model.bar_distribution import FullSupportBarDistribution


class FakeNetwork:
    def __init__(self):
        self.inputs = []

    def __call__(self, values, return_embeddings):
        assert return_embeddings
        self.inputs.append(values.cpu().numpy().copy())
        return torch.nan_to_num(values)[..., None]

    def decoder(self, features):
        return torch.cat([features - 2, features + 1, features, features - 1], dim=-1)


def distribution():
    return FullSupportBarDistribution(torch.tensor([-2., -1., 0., 1., 2.]))


def test_train_only_numeric_transform_and_constants():
    train = pd.DataFrame({"number": [0., 1., 2., 3., np.nan], "constant": [5.] * 5,
                          "empty": [np.nan] * 5, "missing": [2., 2., np.nan, 2., 2.]})
    preprocessor = TabularPreprocessor().fit(train)
    quantiles = preprocessor.specifications_[0][2].quantiles_.copy()
    test = pd.DataFrame({"number": [1e12, np.inf, np.nan], "constant": [999.] * 3,
                         "empty": [8.] * 3, "missing": [2., np.nan, 2.]})
    result = preprocessor.transform(test)
    assert result.shape == (3, 2)
    assert result[0, 0] == 5 and np.isnan(result[1:, 0]).all()
    np.testing.assert_array_equal(quantiles, preprocessor.specifications_[0][2].quantiles_)
    assert preprocessor.dropped_columns_ == ["constant", "empty"]
    np.testing.assert_array_equal(result[:, 1], [-1, 1, -1])


def test_nominal_vocab_uses_observed_train_values_not_pandas_categories():
    values = pd.Categorical(["a", "b", "c", "a", None], categories=["a", "b", "c", "test_only"])
    preprocessor = TabularPreprocessor(max_categories=2).fit(pd.DataFrame({"category": values}))
    result = preprocessor.transform(pd.DataFrame({"category": ["a", "b", "c", "test_only", None]}))
    assert result.shape == (5, 5)
    np.testing.assert_array_equal(result.argmax(axis=1), [0, 1, 2, 4, 3])
    assert "str:'test_only'" not in preprocessor.specifications_[0][2]["seen"]
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(5))


def test_numeric_looking_nominal_and_unknown_binary_handling():
    train = pd.DataFrame({"code": [10, 20, 10, 20]})
    preprocessor = TabularPreprocessor().fit(train, ["code"])
    result = preprocessor.transform(pd.DataFrame({"code": [10, 20, 30]}))
    np.testing.assert_array_equal(result[:2, 0], [-1, 1])
    assert np.isnan(result[2, 0])


def test_unflagged_string_columns_are_categorical():
    train = pd.DataFrame({"age_band": ["30-40 years", "40-50 years", "30-40 years"],
                          "outcome": ["unknown", "success", "failure"]})
    preprocessor = TabularPreprocessor().fit(train, categorical_columns=[])
    result = preprocessor.transform(train)
    assert np.isfinite(result).all()
    assert set(preprocessor.inferred_categorical_columns_) == {"age_band", "outcome"}
    assert result.shape == (3, 6)


def test_iso_dates_get_train_fitted_numeric_calendar_features():
    train = pd.DataFrame({"date": ["2013-01-15", "2013-02-16", "2014-04-01", None]})
    preprocessor = TabularPreprocessor().fit(train, categorical_columns=[])
    result = preprocessor.transform(pd.DataFrame({"date": ["2013-01-15", "2029-12-31", "invalid", None]}))
    assert preprocessor.datetime_columns_ == ["date"]
    assert result.shape == (4, 5)
    assert np.isfinite(result[:2]).all() and np.isnan(result[2:]).all()
    assert np.max(np.abs(result[:2])) <= 5


@pytest.mark.parametrize("problem_type,target", [("regression", [100., 110., 120., 130.]),
                                                 ("binary", ["no", "yes", "no", "yes"]),
                                                 ("multiclass", ["cat", "dog", "owl", "cat"])])
def test_training_labels_only_and_every_query_target_masked(problem_type, target):
    network = FakeNetwork()
    predictor = SupervisedMatrixFormer(network, distribution(), problem_type, device="cpu", max_batch_size=2)
    train = pd.DataFrame({"feature": [0., 1., 2., 3.]})
    predictor.fit(train, pd.Series(target))
    query = pd.DataFrame({"feature": [4., 5., 6.]})
    result = predictor._predict(query)
    assert len(result) == 3 and np.isfinite(result).all()
    for matrix in network.inputs:
        assert np.isnan(matrix[:, -1, -predictor.target_columns_:]).all()
        assert np.isfinite(matrix[:, :-1, -predictor.target_columns_:]).all()
        assert matrix.shape[1] == len(train) + 1
    if problem_type == "regression":
        assert predictor.target_mean_ == 115
        assert predictor.target_scale_ == pytest.approx(np.std(target))
    else:
        np.testing.assert_allclose(result.sum(axis=1), 1)
        assert (result > 0).all() and (result < 1).all()


def test_predict_is_independent_of_unrelated_test_rows():
    network = FakeNetwork()
    predictor = SupervisedMatrixFormer(network, distribution(), "binary", device="cpu")
    predictor.fit(pd.DataFrame({"feature": [0., 1., 2., 3.]}), pd.Series([0, 1, 0, 1]))
    alone = predictor.predict_proba(pd.DataFrame({"feature": [1.5]}))
    first_matrix = network.inputs[-1][0].copy()
    combined = predictor.predict_proba(pd.DataFrame({"feature": [1.5, -1e10, 1e10]}))
    np.testing.assert_array_equal(first_matrix, network.inputs[-1][0])
    np.testing.assert_array_equal(alone[0], combined[0])


def test_context_selection_deterministic_preserves_rare_training_classes():
    train = pd.DataFrame({"feature": np.arange(100)})
    target = pd.Series(["common"] * 99 + ["rare"])
    predictors = [SupervisedMatrixFormer(FakeNetwork(), distribution(), "binary", context_rows=8, device="cpu").fit(train, target)
                  for _ in range(2)]
    np.testing.assert_array_equal(predictors[0].context_indices_, predictors[1].context_indices_)
    assert len(predictors[0].context_indices_) == 8 and 99 in predictors[0].context_indices_


def test_binary_probabilities_are_cdf_masses_not_means():
    predictor = SupervisedMatrixFormer(FakeNetwork(), distribution(), "binary", device="cpu")
    predictor.fit(pd.DataFrame({"feature": [0., 1., 2., 3.]}), pd.Series([0, 1, 0, 1]))
    probability = predictor.predict_proba(pd.DataFrame({"feature": [1.5]}))[0, 1]
    expected = torch.softmax(torch.tensor([-2., 1., 0., -1.]), dim=0)[2:].sum().item()
    assert probability == pytest.approx(expected, abs=1e-7)


def test_predict_rejects_changed_feature_order():
    preprocessor = TabularPreprocessor().fit(pd.DataFrame({"first": [1, 2], "second": [2, 3]}))
    with pytest.raises(ValueError, match="column order"):
        preprocessor.transform(pd.DataFrame({"second": [4], "first": [5]}))
