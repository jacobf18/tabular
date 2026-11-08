"""Leakage-safe, inductive supervised prediction through frozen matrix completion."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch


def category_tokens(series):
    return series.astype(object).map(lambda value: None if pd.isna(value) else
                                    f"{type(value).__name__}:{value!r}")


def is_datetime_column(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    observed = series.dropna().astype(str)
    return bool(len(observed) and observed.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?").all()
                and pd.to_datetime(observed, errors="coerce", format="ISO8601", utc=True).notna().all())


def datetime_features(series, name):
    parsed = pd.to_datetime(series, errors="coerce", format="ISO8601", utc=True)
    days = (parsed.astype("int64") / (86400 * 1e9)).where(parsed.notna(), np.nan)
    month = parsed.dt.month.astype(float)
    weekday = parsed.dt.dayofweek.astype(float)
    return pd.DataFrame({f"{name}:days": days,
                         f"{name}:month_sin": np.sin(2 * np.pi * month / 12),
                         f"{name}:month_cos": np.cos(2 * np.pi * month / 12),
                         f"{name}:weekday_sin": np.sin(2 * np.pi * weekday / 7),
                         f"{name}:weekday_cos": np.cos(2 * np.pi * weekday / 7)})


class TabularPreprocessor:
    """Train-fitted numerical Gaussianization and non-ordinal nominal encoding."""

    def __init__(self, max_categories=32):
        if max_categories < 2:
            raise ValueError("Require at least two categories")
        self.max_categories = max_categories

    def fit(self, features, categorical_columns=None):
        if not features.columns.is_unique:
            raise ValueError("Feature names must be unique")
        self.columns_ = list(features.columns)
        explicit_categories = set(categorical_columns or [])
        self.datetime_columns_ = [name for name in features if name not in explicit_categories and is_datetime_column(features[name])]
        inferred_categories = {name for name in features if name not in self.datetime_columns_
                               and (not pd.api.types.is_numeric_dtype(features[name]) or pd.api.types.is_bool_dtype(features[name]))}
        self.categorical_columns_ = explicit_categories | inferred_categories
        self.inferred_categorical_columns_ = sorted(inferred_categories - explicit_categories)
        if not self.categorical_columns_.issubset(self.columns_):
            raise ValueError("Unknown categorical feature names")
        self.specifications_ = []
        self.output_names_ = []
        self.dropped_columns_ = []
        for name in self.columns_:
            series = features[name]
            if name in self.datetime_columns_:
                child = TabularPreprocessor(self.max_categories).fit(datetime_features(series, name), [])
                self.specifications_.append((name, "date", child))
                self.output_names_.extend(child.output_names_)
            elif name in self.categorical_columns_:
                tokens = category_tokens(series)
                counts = tokens.dropna().value_counts()
                if len(counts) == 0 or (len(counts) == 1 and not tokens.isna().any()):
                    self.dropped_columns_.append(name)
                    continue
                ordered = sorted(counts.index, key=lambda token: (-counts[token], token))
                if len(ordered) <= 2:
                    self.specifications_.append((name, "binary", ordered))
                    self.output_names_.append(f"{name}:binary")
                else:
                    kept = ordered[:self.max_categories]
                    specification = {"kept": kept, "seen": set(ordered), "rare": len(ordered) > len(kept)}
                    self.specifications_.append((name, "nominal", specification))
                    labels = kept + (["OTHER"] if specification["rare"] else []) + ["MISSING", "UNKNOWN"]
                    self.output_names_.extend(f"{name}:{label}" for label in labels)
            else:
                values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
                finite = np.isfinite(values)
                if len(np.unique(values[finite])) <= 1:
                    if finite.any() and (~finite).any():
                        self.specifications_.append((name, "missingness", None))
                        self.output_names_.append(f"{name}:missingness")
                    else:
                        self.dropped_columns_.append(name)
                    continue
                transformer = QuantileTransformer(n_quantiles=min(1024, int(finite.sum())),
                                                  output_distribution="normal", subsample=None, random_state=0)
                transformer.fit(values[finite, None])
                self.specifications_.append((name, "numeric", transformer))
                self.output_names_.append(str(name))
        if not self.output_names_:
            self.output_names_ = ["constant_context"]
        return self

    def transform(self, features):
        if list(features.columns) != self.columns_:
            raise ValueError("Prediction columns or column order differ from training")
        blocks = []
        for name, kind, specification in self.specifications_:
            if kind == "date":
                block = specification.transform(datetime_features(features[name], name))
            elif kind in ["numeric", "missingness"]:
                values = pd.to_numeric(features[name], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
                finite = np.isfinite(values)
                if kind == "missingness":
                    block = np.where(finite, -1.0, 1.0)[:, None]
                else:
                    block = np.full((len(features), 1), np.nan)
                    if finite.any():
                        block[finite] = np.clip(specification.transform(values[finite, None]), -5, 5)
            else:
                tokens = category_tokens(features[name])
                if kind == "binary":
                    mapping = {token: -1.0 if index == 0 else 1.0 for index, token in enumerate(specification)}
                    block = tokens.map(mapping).to_numpy(dtype=np.float64)[:, None]
                    if len(specification) == 1:
                        block[tokens.isna().to_numpy()] = 1.0
                else:
                    mapping = {token: index for index, token in enumerate(specification["kept"])}
                    rare_index = len(mapping) if specification["rare"] else None
                    missing_index = len(mapping) + int(specification["rare"])
                    unknown_index = missing_index + 1
                    indices = []
                    for token in tokens:
                        if token is None:
                            index = missing_index
                        elif token in mapping:
                            index = mapping[token]
                        elif token in specification["seen"]:
                            index = rare_index
                        else:
                            index = unknown_index
                        indices.append(index)
                    block = np.zeros((len(features), unknown_index + 1))
                    block[np.arange(len(features)), indices] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(features), 1), dtype=np.float32)

    def summary(self):
        return {"input_features": len(self.columns_), "encoded_features": len(self.output_names_),
                "categorical_columns": sorted(self.categorical_columns_), "dropped_columns": self.dropped_columns_,
                "datetime_columns": self.datetime_columns_, "dtype_inferred_categorical_columns": self.inferred_categorical_columns_,
                "feature_blocks": [{"name": name, "kind": kind,
                                    "training_categories": len(specification["seen"]) if kind == "nominal" else
                                    len(specification) if kind == "binary" else None}
                                   for name, kind, specification in self.specifications_]}


class SupervisedMatrixFormer:
    """One query per independent matrix; other test rows never enter its context."""

    def __init__(self, model, distribution, problem_type, context_rows=1023, max_batch_cells=131072,
                 max_batch_size=8, seed=0, device="cuda"):
        if problem_type not in ["regression", "binary", "multiclass"]:
            raise ValueError(problem_type)
        if context_rows < 2 or max_batch_cells < 1 or max_batch_size < 1:
            raise ValueError("Invalid inference budget")
        self.model = model
        self.distribution = distribution
        self.problem_type = problem_type
        self.context_rows = context_rows
        self.max_batch_cells = max_batch_cells
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.device = device

    def fit(self, features, target, categorical_columns=None):
        if len(features) != len(target) or len(features) < 2 or pd.isna(target).any():
            raise ValueError("Training rows and finite/nonmissing labels are required")
        self.preprocessor_ = TabularPreprocessor().fit(features, categorical_columns)
        transformed = self.preprocessor_.transform(features)
        if self.problem_type == "regression":
            values = np.asarray(target, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite regression label")
            self.target_mean_ = float(values.mean())
            self.target_scale_ = float(values.std()) or 1.0
            encoded_target = ((values - self.target_mean_) / self.target_scale_)[:, None]
            self.classes_ = None
        else:
            self.classes_ = np.asarray(sorted(pd.unique(pd.Series(target).astype(object)), key=lambda value: (str(type(value)), str(value))))
            if len(self.classes_) < 2 or (self.problem_type == "binary" and len(self.classes_) != 2):
                raise ValueError("Unexpected number of training classes")
            class_indices = pd.Index(self.classes_).get_indexer(np.asarray(target))
            if self.problem_type == "binary":
                encoded_target = (2 * class_indices - 1)[:, None]
            else:
                encoded_target = 2 * np.eye(len(self.classes_))[class_indices] - 1
        self.target_columns_ = encoded_target.shape[1]
        random = np.random.default_rng(self.seed)
        if len(features) <= self.context_rows:
            indices = np.arange(len(features))
        elif self.classes_ is None:
            indices = np.sort(random.choice(len(features), self.context_rows, replace=False))
        else:
            if len(self.classes_) > self.context_rows:
                raise ValueError("Not enough context rows to retain every training class")
            anchors = np.array([random.choice(np.flatnonzero(class_indices == index)) for index in range(len(self.classes_))])
            remaining = np.setdiff1d(np.arange(len(features)), anchors)
            indices = np.sort(np.concatenate([anchors, random.choice(remaining, self.context_rows - len(anchors), replace=False)]))
        self.context_indices_ = indices
        self.context_ = np.concatenate([transformed[indices], encoded_target[indices]], axis=1).astype(np.float32)
        self.batch_size_ = min(self.max_batch_size, max(1, self.max_batch_cells // ((len(indices) + 1) * self.context_.shape[1])))
        return self

    def _query_matrices(self, transformed):
        matrix = np.empty((len(transformed), len(self.context_) + 1, self.context_.shape[1]), dtype=np.float32)
        matrix[:, :-1] = self.context_[None]
        matrix[:, -1, :-self.target_columns_] = transformed
        matrix[:, -1, -self.target_columns_:] = np.nan
        return matrix

    @torch.inference_mode()
    def _predict(self, features):
        transformed = self.preprocessor_.transform(features)
        predictions = []
        for start in range(0, len(features), self.batch_size_):
            matrix = self._query_matrices(transformed[start:start + self.batch_size_])
            values = torch.from_numpy(matrix).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                encoded = self.model(values, return_embeddings=True)
                query_features = encoded[:, -1, -self.target_columns_:].reshape(-1, encoded.shape[-1])
                logits = self.model.decoder(query_features).float()
            if self.problem_type == "regression":
                point = self.distribution.mean(logits).cpu().numpy().astype(np.float64)
                predictions.append(point * self.target_scale_ + self.target_mean_)
            else:
                threshold = torch.tensor([0.0], device=logits.device)
                if not self.distribution.borders[0] < 0 < self.distribution.borders[-1]:
                    raise ValueError("The binary decision threshold must lie inside the finite histogram borders")
                positive = (1 - self.distribution.cdf(logits, threshold).squeeze(-1)).clamp(1e-7, 1 - 1e-7)
                positive = positive.reshape(len(matrix), self.target_columns_).cpu().numpy().astype(np.float64)
                probabilities = np.concatenate([1 - positive, positive], axis=1) if self.problem_type == "binary" else positive / positive.sum(axis=1, keepdims=True)
                predictions.append(probabilities)
        if not predictions:
            return np.empty((0,)) if self.classes_ is None else np.empty((0, len(self.classes_)))
        result = np.concatenate(predictions)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite supervised predictions")
        return result

    def predict_proba(self, features):
        if self.classes_ is None:
            raise ValueError("Regression does not have class probabilities")
        return self._predict(features)

    def predict(self, features):
        prediction = self._predict(features)
        return prediction if self.classes_ is None else self.classes_[prediction.argmax(axis=1)]
