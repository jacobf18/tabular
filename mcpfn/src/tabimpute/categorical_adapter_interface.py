from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from tabimpute.prepreocess import (
    RandomColumnPermutation,
    RandomRowColumnPermutation,
    RandomRowPermutation,
    SequentialPreprocess,
)
from tabimpute.tabimpute_v2 import TabImputeV2


@dataclass
class SingleCategoricalColumnState:
    # Reusable adapter state for one categorical column.
    target_col: int
    categories: list[Any]
    category_to_index: dict[Any, int]
    adapter: nn.Module
    losses: list[float]


@dataclass
class CategoricalAdapterState:
    # Multi-column container: one lightweight encoder/decoder pair per categorical column.
    target_cols: list[int]
    column_states: dict[int, SingleCategoricalColumnState]


@dataclass
class _PreparedCategoricalColumn:
    target_col: int
    categories: list[Any]
    category_to_index: dict[Any, int]
    category_ids: np.ndarray
    observed_mask: np.ndarray


class _CategoricalColumnAdapter(nn.Module):
    # Lightweight trainable head: category -> pretrained embedding space -> category logits.
    def __init__(
        self,
        embedding_size: int,
        num_categories: int,
        hidden_size: int,
        encoder_init: Optional[torch.Tensor] = None,
        decoder_init: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.encoder = nn.Embedding(num_categories, embedding_size)
        if encoder_init is not None:
            with torch.no_grad():
                self.encoder.weight.copy_(encoder_init)

        self.decoder_norm = nn.LayerNorm(embedding_size)
        self.decoder_hidden = nn.Linear(embedding_size, hidden_size)
        self.decoder_activation = nn.GELU()
        self.decoder_out = nn.Linear(hidden_size, num_categories)

        if decoder_init is not None:
            # Reuse the first pretrained decoder projection as a sensible starting point.
            with torch.no_grad():
                self.decoder_hidden.weight.copy_(decoder_init.weight)
                self.decoder_hidden.bias.copy_(decoder_init.bias)

    def encode(self, category_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(category_ids)

    def decode(self, contextual_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.decoder_norm(contextual_embeddings)
        x = self.decoder_hidden(x)
        x = self.decoder_activation(x)
        return self.decoder_out(x)


class TabImputeCategoricalAdapter(TabImputeV2):
    """Test-time categorical adapters for one or more target columns.

    The pretrained TabImpute V2 backbone stays frozen. For each categorical
    column we build a small encoder/decoder pair that maps that column's values
    into the same embedding space used by the pretrained scalar feature encoder,
    then decodes contextual backbone embeddings back into that column's labels.

    Each categorical column gets its own paired encoder/decoder, so the values
    may be of arbitrary Python types as long as they are hashable.

    If ``preprocessors`` is set (same as ``TabImputeV2`` / ``ImputePFN``), after
    adapter training the inference step mirrors ``ImputePFN.impute``: each
    preprocessor gets an independent ``fit_transform`` on the normalized
    inference matrix, the frozen model + adapters run in that view, then
    ``inverse_transform`` and averaging (numeric in normalized space; categorical
    predictions are majority-voted across preprocessor draws).

    Assumptions
    -----------
    - Target categorical columns may contain arbitrary hashable values plus missing values.
    - All non-target columns must be numeric (or missing) so they can be passed
      through the pretrained numeric feature encoder.

    Hyperparameters: adapter heads are small; use AdamW learning rates on the order
    of 1e-3–1e-2 with enough steps (hundreds). Much smaller rates with few steps
    typically underfit versus a full one-hot TabImpute forward.
    """

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except TypeError:
            return False

    def _normalize_target_cols(self, X_obj: np.ndarray, target_cols: list[int]) -> list[int]:
        if len(target_cols) == 0:
            raise ValueError("target_cols must contain at least one categorical column.")
        n_cols = X_obj.shape[1]
        normalized = []
        for target_col in target_cols:
            if target_col < 0 or target_col >= n_cols:
                raise ValueError(
                    f"target_col {target_col} is out of bounds for {n_cols} columns."
                )
            normalized.append(int(target_col))
        deduped = sorted(set(normalized))
        if len(deduped) != len(normalized):
            raise ValueError("target_cols must not contain duplicates.")
        return deduped

    def _prepare_categorical_inputs(
        self,
        X: np.ndarray,
        target_cols: list[int],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[int, _PreparedCategoricalColumn],
    ]:
        # Work in object dtype so mixed numeric / string columns can be inspected safely.
        X_obj = np.asarray(X, dtype=object)
        if X_obj.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")
        n_rows, n_cols = X_obj.shape
        target_cols = self._normalize_target_cols(X_obj, target_cols)

        prepared_columns: dict[int, _PreparedCategoricalColumn] = {}
        for target_col in target_cols:
            observed_mask = np.array(
                [not self._is_missing_value(v) for v in X_obj[:, target_col]], dtype=bool
            )
            observed_values = X_obj[observed_mask, target_col].tolist()
            if len(observed_values) == 0:
                raise ValueError(
                    f"Target categorical column {target_col} has no observed values to adapt on."
                )

            # Preserve first-seen order so category IDs are stable and human-readable.
            categories = list(dict.fromkeys(observed_values))
            category_to_index = {category: idx for idx, category in enumerate(categories)}
            category_ids = np.full(n_rows, -1, dtype=np.int64)
            for row_idx in np.where(observed_mask)[0]:
                category_ids[row_idx] = category_to_index[X_obj[row_idx, target_col]]

            prepared_columns[target_col] = _PreparedCategoricalColumn(
                target_col=target_col,
                categories=categories,
                category_to_index=category_to_index,
                category_ids=category_ids,
                observed_mask=observed_mask,
            )

        X_numeric = np.empty((n_rows, n_cols), dtype=np.float32)
        categorical_cols = set(target_cols)
        for col_idx in range(n_cols):
            if col_idx in categorical_cols:
                prepared = prepared_columns[col_idx]
                # Categorical columns are represented by their own adapters. Here we
                # only preserve their observed/missing pattern for the frozen backbone.
                X_numeric[:, col_idx] = np.where(
                    prepared.observed_mask, 0.0, np.nan
                ).astype(np.float32)
                continue

            col = X_obj[:, col_idx]
            out_col = np.empty(n_rows, dtype=np.float32)
            for row_idx, value in enumerate(col):
                if self._is_missing_value(value):
                    out_col[row_idx] = np.nan
                else:
                    try:
                        out_col[row_idx] = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "All non-target columns must be numeric when using "
                            "TabImputeCategoricalAdapter."
                        ) from exc
            X_numeric[:, col_idx] = out_col

        means, stds = self._resolve_normalization_stats(X_numeric)
        # Keep placeholder categorical columns centered and unscaled; only their mask matters.
        for target_col in target_cols:
            means[target_col] = 0.0
            stds[target_col] = 1.0
        X_normalized = (X_numeric - means) / (stds + 1e-16)
        for target_col, prepared in prepared_columns.items():
            X_normalized[:, target_col] = np.where(
                prepared.observed_mask, 0.0, np.nan
            ).astype(np.float32)
        return (
            X_numeric.astype(np.float32),
            X_normalized.astype(np.float32),
            means.astype(np.float32),
            stds.astype(np.float32),
            prepared_columns,
        )

    def _build_adapter(
        self,
        categories: list[Any],
    ) -> _CategoricalColumnAdapter:
        # Match the pretrained backbone dimensionality exactly so each adapter can
        # drop category embeddings directly into the frozen feature encoder stream.
        embedding_size = int(self.model.feature_encoder.embedding_size)
        hidden_size = int(self.model.decoder.layers[0].out_features)
        model_dtype = next(self.model.parameters()).dtype
        device = torch.device(self.device)

        lo = float(self.borders[0].detach().cpu())
        hi = float(self.borders[-1].detach().cpu())
        if len(categories) == 1:
            anchors = np.array([(lo + hi) / 2.0], dtype=np.float32)
        else:
            anchors = np.linspace(lo, hi, len(categories), dtype=np.float32)
        anchor_tensor = torch.from_numpy(anchors).to(
            device=device, dtype=model_dtype
        ).unsqueeze(-1)
        with torch.no_grad():
            # Bootstrap category embeddings by passing numeric anchors through the
            # pretrained scalar encoder. This keeps the new encoder in-family with
            # the space the transformer already understands.
            encoder_init = (
                self.model.feature_encoder.observed_linear_layer(anchor_tensor)
                .to(dtype=torch.float32)
                .detach()
            )

        decoder_init = self.model.decoder.layers[0]
        adapter = _CategoricalColumnAdapter(
            embedding_size=embedding_size,
            num_categories=len(categories),
            hidden_size=hidden_size,
            encoder_init=encoder_init,
            decoder_init=decoder_init,
        )
        return adapter.to(device)

    def _build_adapters(
        self,
        prepared_columns: dict[int, _PreparedCategoricalColumn],
    ) -> dict[int, _CategoricalColumnAdapter]:
        return {
            target_col: self._build_adapter(prepared.categories)
            for target_col, prepared in prepared_columns.items()
        }

    @staticmethod
    def _spatial_row_col_maps_from_preprocessor(
        preprocessor: Any,
        n_rows: int,
        n_cols: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map preprocessed indices back to original row/column indices.

        For transforms where ``X_pre[i, j] = X[cum_r[i], cum_c[j]]`` (row/column
        permutations only), ``cum_r`` and ``cum_c`` record which original row/col
        each preprocessed slot corresponds to. Non-spatial preprocessors
        (e.g. whitening) leave the identity mapping.
        """
        cum_r = np.arange(n_rows, dtype=np.int64)
        cum_c = np.arange(n_cols, dtype=np.int64)

        def _compose(subs: list[Any]) -> None:
            nonlocal cum_r, cum_c
            for sub in subs:
                if isinstance(sub, RandomRowColumnPermutation):
                    cum_r = cum_r[np.asarray(sub.row_perm, dtype=np.int64)]
                    cum_c = cum_c[np.asarray(sub.col_perm, dtype=np.int64)]
                elif isinstance(sub, RandomRowPermutation):
                    cum_r = cum_r[np.asarray(sub.perm, dtype=np.int64)]
                elif isinstance(sub, RandomColumnPermutation):
                    cum_c = cum_c[np.asarray(sub.perm, dtype=np.int64)]

        if isinstance(preprocessor, SequentialPreprocess):
            _compose(list(preprocessor.preprocessors))
        else:
            _compose([preprocessor])
        return cum_r, cum_c

    @staticmethod
    def _remap_prepared_columns_for_permutation(
        prepared_columns: dict[int, _PreparedCategoricalColumn],
        cum_r: np.ndarray,
        cum_c: np.ndarray,
    ) -> dict[int, _PreparedCategoricalColumn]:
        """Rebuild prepared column metadata in permuted column / row order."""
        out: dict[int, _PreparedCategoricalColumn] = {}
        for old_col, prep in prepared_columns.items():
            new_col = int(np.flatnonzero(cum_c == old_col)[0])
            out[new_col] = _PreparedCategoricalColumn(
                target_col=new_col,
                categories=prep.categories,
                category_to_index=prep.category_to_index,
                category_ids=prep.category_ids[cum_r],
                observed_mask=prep.observed_mask[cum_r],
            )
        return out

    def _encode_with_categorical_adapters(
        self,
        X_normalized: torch.Tensor,
        prepared_columns: dict[int, _PreparedCategoricalColumn],
        category_input_observed: dict[int, torch.Tensor],
        adapters: dict[int, _CategoricalColumnAdapter],
    ) -> tuple[torch.Tensor, int, int]:
        feature_encoder = self.model.feature_encoder
        nan_mask = torch.isnan(X_normalized)
        x_clean = torch.where(nan_mask, torch.zeros_like(X_normalized), X_normalized)

        # Start from the normal numeric encoding path for every column.
        embedded_observed = feature_encoder.observed_linear_layer(x_clean.unsqueeze(-1))

        # Replace each target column's observed entries with its own learned
        # categorical embeddings. Each column has a dedicated encoder/decoder pair.
        for target_col, prepared in prepared_columns.items():
            category_ids = torch.from_numpy(prepared.category_ids).to(self.device).unsqueeze(0)
            target_embeddings = adapters[target_col].encode(category_ids.clamp_min(0)).to(
                dtype=embedded_observed.dtype
            )
            target_slot = embedded_observed[:, :, target_col, :]
            observed_target_mask = category_input_observed[target_col].unsqueeze(-1)
            embedded_observed[:, :, target_col, :] = torch.where(
                observed_target_mask, target_embeddings, target_slot
            )

        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed)
        batch_size, rows, cols = X_normalized.shape
        mask_expanded = feature_encoder.mask_token.expand(
            batch_size, rows, cols, feature_encoder.embedding_size
        )
        embedded = torch.where(nan_mask_expanded, mask_expanded, embedded_observed)

        missingness_ids = nan_mask.to(torch.long)
        embedded = embedded + feature_encoder.missingness_embedding(missingness_ids)

        if (
            feature_encoder.row_embedding is not None
            and feature_encoder.column_embedding is not None
        ):
            embedded = feature_encoder.row_embedding(embedded)
            embedded = feature_encoder.column_embedding(embedded)

        # Rebuild the exact token layout expected by the pretrained transformer:
        # corner CLS tokens, row CLS tokens, column CLS tokens, then data tokens.
        k_rows = int(feature_encoder.num_cls)
        k_cols = int(feature_encoder.num_cls)
        feature_encoder.last_num_cls_rows = k_rows
        feature_encoder.last_num_cls_cols = k_cols

        device = X_normalized.device
        dtype = embedded.dtype
        row_indices = torch.arange(k_cols, device=device, dtype=torch.long)
        col_indices = torch.arange(k_rows, device=device, dtype=torch.long)

        row_cls_tokens = feature_encoder.row_cls_embedding(row_indices).to(dtype)
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(
            batch_size, rows, k_cols, feature_encoder.embedding_size
        )

        col_cls_tokens = feature_encoder.col_cls_embedding(col_indices).to(dtype)
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(
            batch_size, k_rows, cols, feature_encoder.embedding_size
        )

        row_corner = row_cls_tokens.unsqueeze(0).expand(
            k_rows, k_cols, feature_encoder.embedding_size
        )
        col_corner = col_cls_tokens.unsqueeze(1).expand(
            k_rows, k_cols, feature_encoder.embedding_size
        )
        corner = (row_corner + col_corner).unsqueeze(0).expand(
            batch_size, k_rows, k_cols, feature_encoder.embedding_size
        )

        top_block = torch.cat([corner, col_cls], dim=2)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        encoded = torch.cat([top_block, bottom_block], dim=1)
        encoded = feature_encoder.embedding_dropout(encoded)
        return encoded, k_rows, k_cols

    def _forward_contextual_embeddings(
        self,
        X_normalized: np.ndarray,
        prepared_columns: dict[int, _PreparedCategoricalColumn],
        category_input_observed: dict[int, np.ndarray],
        adapters: dict[int, _CategoricalColumnAdapter],
    ) -> torch.Tensor:
        # Run one frozen-backbone forward pass with custom embeddings for every
        # categorical column and return the contextual data-token embeddings.
        model_dtype = next(self.model.parameters()).dtype
        X_tensor = torch.from_numpy(X_normalized).to(self.device).unsqueeze(0).to(model_dtype)
        category_observed_t = {
            target_col: torch.from_numpy(mask).to(self.device).unsqueeze(0)
            for target_col, mask in category_input_observed.items()
        }

        encoded, num_cls_rows, num_cls_cols = self._encode_with_categorical_adapters(
            X_tensor,
            prepared_columns=prepared_columns,
            category_input_observed=category_observed_t,
            adapters=adapters,
        )
        for block in self.model.transformer_blocks:
            encoded = block(encoded, num_cls_rows=num_cls_rows, num_cls_cols=num_cls_cols)
        return encoded[:, num_cls_rows:, num_cls_cols:, :]

    def _forward_categorical_logits(
        self,
        contextual: torch.Tensor,
        prepared_columns: dict[int, _PreparedCategoricalColumn],
        adapters: dict[int, _CategoricalColumnAdapter],
    ) -> dict[int, torch.Tensor]:
        return {
            target_col: adapters[target_col].decode(contextual[:, :, target_col, :].float())
            for target_col in prepared_columns
        }

    def train_column_adapters(
        self,
        X: np.ndarray,
        target_cols: list[int],
        *,
        max_steps: int = 200,
        mask_prob: float = 0.35,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_rows: Optional[int] = None,
        random_state: int = 0,
        verbose: bool = False,
    ) -> CategoricalAdapterState:
        _, X_normalized, _, _, prepared_columns = self._prepare_categorical_inputs(
            X, target_cols
        )
        adapters = self._build_adapters(prepared_columns)
        rng = np.random.default_rng(random_state)

        # Freeze the pretrained model; only the small categorical adapters train.
        previous_requires_grad = [param.requires_grad for param in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        optimizer = torch.optim.AdamW(
            [param for adapter in adapters.values() for param in adapter.parameters()],
            lr=lr,
            weight_decay=weight_decay,
        )
        losses_by_col: dict[int, list[float]] = {target_col: [] for target_col in prepared_columns}
        try:
            for adapter in adapters.values():
                adapter.train()

            if any(len(prepared.categories) > 1 for prepared in prepared_columns.values()):
                if verbose:
                    pbar = tqdm(
                        total=max_steps,
                        desc=f"Training column adapters for {len(prepared_columns)} cols",
                    )
                for _ in range(max_steps):
                    if verbose:
                        pbar.update(1)
                    input_observed_by_col: dict[int, np.ndarray] = {}
                    masked_rows_by_col: dict[int, np.ndarray] = {}
                    active_cols: list[int] = []
                    X_step = X_normalized.copy()

                    for target_col, prepared in prepared_columns.items():
                        observed_rows = np.where(prepared.observed_mask)[0]
                        input_observed = prepared.observed_mask.copy()

                        if len(prepared.categories) > 1 and observed_rows.size > 0:
                            if batch_rows is not None and batch_rows < observed_rows.size:
                                selected_rows = rng.choice(
                                    observed_rows, size=batch_rows, replace=False
                                )
                            else:
                                selected_rows = observed_rows

                            masked_rows = np.zeros_like(prepared.observed_mask)
                            draw = rng.random(selected_rows.size) < mask_prob
                            masked_selected = selected_rows[draw]
                            if masked_selected.size == 0:
                                masked_selected = np.array(
                                    [selected_rows[rng.integers(0, selected_rows.size)]]
                                )
                            masked_rows[masked_selected] = True
                            input_observed[masked_rows] = False
                            masked_rows_by_col[target_col] = masked_rows
                            active_cols.append(target_col)
                        else:
                            masked_rows_by_col[target_col] = np.zeros_like(
                                prepared.observed_mask
                            )

                        # Hide a random subset of observed labels and train the adapter to
                        # recover them from context, mimicking true test-time imputation.
                        X_step[~input_observed, target_col] = np.nan
                        input_observed_by_col[target_col] = input_observed

                    if not active_cols:
                        break

                    contextual = self._forward_contextual_embeddings(
                        X_step,
                        prepared_columns=prepared_columns,
                        category_input_observed=input_observed_by_col,
                        adapters=adapters,
                    )
                    logits_by_col = self._forward_categorical_logits(
                        contextual,
                        prepared_columns=prepared_columns,
                        adapters=adapters,
                    )
                    losses = []
                    for target_col in active_cols:
                        masked_rows = masked_rows_by_col[target_col]
                        target = torch.from_numpy(
                            prepared_columns[target_col].category_ids[masked_rows]
                        ).to(self.device)
                        loss = F.cross_entropy(
                            logits_by_col[target_col][0, masked_rows].float(),
                            target,
                        )
                        losses.append(loss)
                        losses_by_col[target_col].append(float(loss.detach().cpu()))

                    total_loss = torch.stack(losses).mean()
                    if verbose:
                        pbar.set_postfix(loss=float(total_loss.detach().cpu()))
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            for target_col, prepared in prepared_columns.items():
                if len(prepared.categories) == 1 and len(losses_by_col[target_col]) == 0:
                    losses_by_col[target_col].append(0.0)
        finally:
            for param, requires_grad in zip(self.model.parameters(), previous_requires_grad):
                param.requires_grad_(requires_grad)

        for adapter in adapters.values():
            adapter.eval()
        column_states = {
            target_col: SingleCategoricalColumnState(
                target_col=target_col,
                categories=prepared.categories,
                category_to_index=prepared.category_to_index,
                adapter=adapters[target_col],
                losses=losses_by_col[target_col],
            )
            for target_col, prepared in prepared_columns.items()
        }
        return CategoricalAdapterState(
            target_cols=sorted(column_states),
            column_states=column_states,
        )

    def train_column_adapter(
        self,
        X: np.ndarray,
        target_col: int,
        *,
        max_steps: int = 200,
        mask_prob: float = 0.35,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_rows: Optional[int] = None,
        random_state: int = 0,
    ) -> SingleCategoricalColumnState:
        state = self.train_column_adapters(
            X,
            [target_col],
            max_steps=max_steps,
            mask_prob=mask_prob,
            lr=lr,
            weight_decay=weight_decay,
            batch_rows=batch_rows,
            random_state=random_state,
        )
        return state.column_states[target_col]

    def _coerce_state_for_columns(
        self,
        state: CategoricalAdapterState | SingleCategoricalColumnState,
    ) -> CategoricalAdapterState:
        if isinstance(state, SingleCategoricalColumnState):
            return CategoricalAdapterState(
                target_cols=[state.target_col],
                column_states={state.target_col: state},
            )
        return state

    def impute_categorical_columns(
        self,
        X: np.ndarray,
        target_cols: list[int],
        *,
        state: Optional[CategoricalAdapterState] = None,
        train_adapter: bool = True,
        max_steps: int = 200,
        mask_prob: float = 0.35,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_rows: Optional[int] = None,
        random_state: int = 0,
        return_probabilities: bool = False,
        verbose: bool = False,
    ) -> (
        tuple[np.ndarray, CategoricalAdapterState]
        | tuple[np.ndarray, CategoricalAdapterState, dict[int, np.ndarray]]
    ):
        if state is None and not train_adapter:
            raise ValueError("state must be provided when train_adapter=False.")

        X_obj = np.asarray(X, dtype=object)
        X_numeric, X_normalized, means, stds, prepared_columns = (
            self._prepare_categorical_inputs(X_obj, target_cols)
        )
        if state is None:
            state = self.train_column_adapters(
                X_obj,
                target_cols=target_cols,
                max_steps=max_steps,
                mask_prob=mask_prob,
                lr=lr,
                weight_decay=weight_decay,
                batch_rows=batch_rows,
                random_state=random_state,
                verbose=verbose,
            )

        expected_cols = sorted(prepared_columns)
        if sorted(state.target_cols) != expected_cols:
            raise ValueError(
                f"Adapter state columns {sorted(state.target_cols)} do not match requested columns {expected_cols}."
            )

        for target_col, prepared in prepared_columns.items():
            column_state = state.column_states.get(target_col)
            if column_state is None:
                raise ValueError(f"Missing adapter state for target_col={target_col}.")
            if column_state.categories != prepared.categories:
                raise ValueError(
                    f"Observed categories do not match the adapter state for column {target_col}."
                )

        X_infer = X_normalized.copy()
        input_observed_by_col = {
            target_col: prepared.observed_mask.copy()
            for target_col, prepared in prepared_columns.items()
        }
        # At inference we expose only the truly observed category entries and ask
        # each adapter to classify the missing rows in its own column.
        for target_col, prepared in prepared_columns.items():
            X_infer[~prepared.observed_mask, target_col] = np.nan

        categorical_cols = set(prepared_columns)
        numeric_orig_cols = [
            c for c in range(X_obj.shape[1]) if c not in categorical_cols
        ]
        preprocessors_list = getattr(self, "preprocessors", None) or []
        n_pre = len(preprocessors_list)

        X_imputed = X_obj.copy()
        probs_by_col: dict[int, np.ndarray] = {}

        if n_pre == 0:
            adapters = {
                target_col: state.column_states[target_col].adapter
                for target_col in prepared_columns
            }
            with torch.no_grad():
                contextual = self._forward_contextual_embeddings(
                    X_infer,
                    prepared_columns=prepared_columns,
                    category_input_observed=input_observed_by_col,
                    adapters=adapters,
                )
                logits_by_col = self._forward_categorical_logits(
                    contextual,
                    prepared_columns=prepared_columns,
                    adapters=adapters,
                )
                probs_by_col = {
                    target_col: torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                    for target_col, logits in logits_by_col.items()
                }
                numeric_logits = self._postprocess_logits(self.model.decoder(contextual))
                numeric_medians = (
                    self.bar_distribution.median(logits=numeric_logits)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            for col_idx in numeric_orig_cols:
                missing_rows = np.where(np.isnan(X_numeric[:, col_idx]))[0]
                if missing_rows.size == 0:
                    continue
                denormalized = numeric_medians[:, col_idx] * stds[col_idx] + means[col_idx]
                for row_idx in missing_rows:
                    X_imputed[row_idx, col_idx] = float(denormalized[row_idx])

            for target_col, prepared in prepared_columns.items():
                pred_ids = probs_by_col[target_col].argmax(axis=-1)
                missing_rows = np.where(~prepared.observed_mask)[0]
                for row_idx in missing_rows:
                    X_imputed[row_idx, target_col] = state.column_states[
                        target_col
                    ].categories[int(pred_ids[row_idx])]
        else:
            # Same ensemble pattern as ``ImputePFN.impute`` / ``TabImputeV2.impute``:
            # independent preprocessor draws, forward in transformed space, inverse,
            # then average in the normalized (pre-inverse) layout of ``X_infer``.
            acc_norm = np.zeros_like(X_infer, dtype=np.float64)
            cat_votes: dict[tuple[int, int], list[Any]] = defaultdict(list)
            probs_acc: Optional[dict[int, np.ndarray]] = None
            if return_probabilities:
                probs_acc = {
                    old_col: np.zeros(
                        (
                            X_infer.shape[0],
                            len(state.column_states[old_col].categories),
                        ),
                        dtype=np.float64,
                    )
                    for old_col in prepared_columns
                }

            for pre in preprocessors_list:
                X_pre = pre.fit_transform(np.asarray(X_infer, dtype=np.float64))
                cum_r, cum_c = self._spatial_row_col_maps_from_preprocessor(
                    pre, X_infer.shape[0], X_infer.shape[1]
                )
                prepared_pre = self._remap_prepared_columns_for_permutation(
                    prepared_columns, cum_r, cum_c
                )
                input_observed_pre = {
                    col: prepared_pre[col].observed_mask.copy() for col in prepared_pre
                }
                adapters_perm = {
                    int(np.flatnonzero(cum_c == old_col)[0]): state.column_states[
                        old_col
                    ].adapter
                    for old_col in prepared_columns
                }
                with torch.no_grad():
                    contextual = self._forward_contextual_embeddings(
                        X_pre.astype(np.float32),
                        prepared_columns=prepared_pre,
                        category_input_observed=input_observed_pre,
                        adapters=adapters_perm,
                    )
                    logits_by_col = self._forward_categorical_logits(
                        contextual,
                        prepared_columns=prepared_pre,
                        adapters=adapters_perm,
                    )
                    probs_by_pre = {
                        col: torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                        for col, logits in logits_by_col.items()
                    }
                    numeric_logits = self._postprocess_logits(
                        self.model.decoder(contextual)
                    )
                    numeric_medians_pre = (
                        self.bar_distribution.median(logits=numeric_logits)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                X_hat_pre = np.asarray(X_pre, dtype=np.float64).copy()
                for orig_c in numeric_orig_cols:
                    j = int(np.flatnonzero(cum_c == orig_c)[0])
                    miss = np.isnan(X_hat_pre[:, j])
                    X_hat_pre[miss, j] = numeric_medians_pre[miss, j].astype(np.float64)

                X_back = pre.inverse_transform(X_hat_pre)
                acc_norm += np.asarray(X_back, dtype=np.float64)

                for old_col, prep in prepared_columns.items():
                    new_col = int(np.flatnonzero(cum_c == old_col)[0])
                    probs = probs_by_pre[new_col]
                    cats = state.column_states[old_col].categories
                    if return_probabilities and probs_acc is not None:
                        for orig_row in range(X_infer.shape[0]):
                            r = int(np.flatnonzero(cum_r == orig_row)[0])
                            probs_acc[old_col][orig_row] += probs[r]
                    for orig_row in np.where(~prep.observed_mask)[0]:
                        r = int(np.flatnonzero(cum_r == orig_row)[0])
                        cat_votes[(int(orig_row), old_col)].append(
                            cats[int(probs[r].argmax())]
                        )

            acc_norm /= n_pre
            if return_probabilities and probs_acc is not None:
                for old_col in prepared_columns:
                    probs_acc[old_col] /= n_pre
                probs_by_col = probs_acc

            for col_idx in numeric_orig_cols:
                missing_rows = np.where(np.isnan(X_numeric[:, col_idx]))[0]
                if missing_rows.size == 0:
                    continue
                denormalized = acc_norm[:, col_idx] * stds[col_idx] + means[col_idx]
                for row_idx in missing_rows:
                    X_imputed[row_idx, col_idx] = float(denormalized[row_idx])

            for target_col, prepared in prepared_columns.items():
                for row_idx in np.where(~prepared.observed_mask)[0]:
                    votes = cat_votes[(int(row_idx), target_col)]
                    if votes:
                        X_imputed[row_idx, target_col] = Counter(votes).most_common(1)[0][
                            0
                        ]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_probabilities:
            return X_imputed, state, probs_by_col
        return X_imputed, state

    def impute_categorical_column(
        self,
        X: np.ndarray,
        target_col: int,
        *,
        state: Optional[CategoricalAdapterState | SingleCategoricalColumnState] = None,
        train_adapter: bool = True,
        max_steps: int = 200,
        mask_prob: float = 0.35,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_rows: Optional[int] = None,
        random_state: int = 0,
        return_probabilities: bool = False,
        verbose: bool = False,
    ) -> (
        tuple[np.ndarray, SingleCategoricalColumnState]
        | tuple[np.ndarray, SingleCategoricalColumnState, np.ndarray]
    ):
        multi_state = None if state is None else self._coerce_state_for_columns(state)
        result = self.impute_categorical_columns(
            X,
            [target_col],
            state=multi_state,
            train_adapter=train_adapter,
            max_steps=max_steps,
            mask_prob=mask_prob,
            lr=lr,
            weight_decay=weight_decay,
            batch_rows=batch_rows,
            random_state=random_state,
            return_probabilities=return_probabilities,
            verbose=verbose,
        )
        if return_probabilities:
            X_imputed, state_out, probs_by_col = result
            return X_imputed, state_out.column_states[target_col], probs_by_col[target_col]
        X_imputed, state_out = result
        return X_imputed, state_out.column_states[target_col]


__all__ = [
    "SingleCategoricalColumnState",
    "CategoricalAdapterState",
    "TabImputeCategoricalAdapter",
]


if __name__ == "__main__":
    from tabimpute.model.bar_distribution import FullSupportBarDistribution
    from tabimpute.model.model_new_stable import TabImputeModel

    torch.manual_seed(0)
    np.random.seed(0)

    # Build a tiny in-memory backbone so this smoke test does not depend on
    # external checkpoints. We only need the pretrained-style module structure,
    # not a trained model, to validate the adapter mechanics.
    smoke = TabImputeCategoricalAdapter.__new__(TabImputeCategoricalAdapter)
    smoke.device = "cuda"
    smoke.model = TabImputeModel(
        embedding_size=32,
        num_attention_heads=4,
        mlp_hidden_size=64,
        num_layers=2,
        num_outputs=17,
        num_cls=2,
        use_rope=True,
        rope_base=10000.0,
        rope_fraction=0.5,
        use_absolute_positional_embeddings=False,
        positional_damping_factor=0.1,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        drop_path_rate=0.0,
        residual_scale_init=0.1,
        rms_norm_eps=1e-6,
        embedding_dropout=0.0,
    ).to("cuda")
    smoke.model.eval()
    smoke.borders = torch.linspace(-3.0, 3.0, 17)
    smoke.bar_distribution = FullSupportBarDistribution(borders=smoke.borders.to(smoke.device))
    smoke.postprocessor = None
    smoke.postprocessor_kwargs = {}

    # Two categorical columns of different Python types:
    # - column 1: strings
    # - column 3: integers
    X = np.array(
        [
            [0.1, "red", 1.2, 10],
            [0.3, "blue", np.nan, 20],
            [np.nan, None, 0.4, 10],
            [0.7, "red", -1.0, None],
            [1.1, "green", 0.2, 20],
            [0.0, "blue", 0.8, None],
        ],
        dtype=object,
    )

    categorical_cols = [1, 3]
    state = smoke.train_column_adapters(
        X,
        target_cols=categorical_cols,
        max_steps=3,
        batch_rows=4,
        random_state=0,
    )
    assert state.target_cols == categorical_cols
    assert set(state.column_states) == set(categorical_cols)

    X_imputed, reused_state, probs = smoke.impute_categorical_columns(
        X,
        target_cols=categorical_cols,
        state=state,
        train_adapter=False,
        return_probabilities=True,
    )
    assert X_imputed.shape == X.shape
    assert reused_state.target_cols == categorical_cols
    assert set(probs) == set(categorical_cols)
    assert not smoke._is_missing_value(X_imputed[2, 0])
    assert not smoke._is_missing_value(X_imputed[1, 2])

    for target_col in categorical_cols:
        column_state = reused_state.column_states[target_col]
        observed_mask = np.array(
            [not smoke._is_missing_value(v) for v in X[:, target_col]], dtype=bool
        )
        missing_rows = np.where(~observed_mask)[0]

        assert probs[target_col].shape == (
            X.shape[0],
            len(column_state.categories),
        )
        for row_idx in np.where(observed_mask)[0]:
            assert X_imputed[row_idx, target_col] == X[row_idx, target_col]
        for row_idx in missing_rows:
            assert X_imputed[row_idx, target_col] in column_state.categories

    # Keep the single-column wrapper tested too.
    X_single, single_state, single_probs = smoke.impute_categorical_column(
        X,
        target_col=1,
        max_steps=2,
        random_state=1,
        return_probabilities=True,
    )
    assert X_single.shape == X.shape
    assert single_state.target_col == 1
    assert single_probs.shape[0] == X.shape[0]
    assert single_probs.shape[1] == len(single_state.categories)

    print("Categorical adapter smoke test passed.")
