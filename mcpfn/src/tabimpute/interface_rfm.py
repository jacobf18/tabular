"""ImputePFN variant that uses a kernel-ridge head (e.g. RFMHeadAGOP) on
transformer embeddings instead of the MLP bar-distribution decoder.

The base class handles everything around the inner model call: tiling,
standardization, NaN bookkeeping, the public ``.impute(X)`` API. We override
``_run_model_direct`` to route per-chunk inference through

    embeddings = self.model(X_chunk, return_embeddings=True)
    preds      = self.head.predict(embeddings, target, observed_mask, ...)

so registering this class in ``get_uci_errors.py`` writes ``{name}.npy``
files in the same format as every other baseline.

Optional per-matrix tuning. If ``per_matrix_steps > 0``, each call to
``impute(X)`` first runs a brief Adam loop on the head's two scalars
(``log_lambda``, ``log_gamma_scale``) using a held-out fraction of X's
OWN observed cells as the loss target. Modeled on the test-time
hyperparameter fitting that xRFM / MICE do per matrix. Matrix-specific
tuning is discarded between matrices.
"""
from __future__ import annotations

import contextlib
import copy
import io
import warnings

import numpy as np
import torch
import torch.nn as nn

from tabimpute.interface import ImputePFN


# Defaults matched to xrfm_wrapper.py (the bar plot's xRFM bar).
_DEFAULT_RFM_PARAMS = {
    "model": {
        "kernel": "l2_high_dim",
        "exponent": 1.0,
        "bandwidth": 10.0,
        "diag": True,
        "bandwidth_mode": "constant",
    },
    "fit": {
        "reg": 1e-3,
        "iters": 3,
        "early_stop_rfm": True,
        "verbose": False,
    },
}


class RFMImputePFN(ImputePFN):
    """ImputePFN with a swappable kernel head replacing the MLP decoder.

    Use with ``entry_wise_features=False`` and a direct-style backbone
    checkpoint (e.g. dwip_v2-*).
    """

    def __init__(
        self,
        *args,
        head: nn.Module,
        num_anchors: int = 1024,
        per_matrix_steps: int = 0,
        per_matrix_lr: float = 1e-2,
        per_matrix_val_frac: float = 0.1,
        per_matrix_patch_size: int = 50,
        per_matrix_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.entry_wise_features:
            raise ValueError(
                "RFMImputePFN only supports entry_wise_features=False "
                "(direct mat-to-mat backbone)."
            )
        self.head = head.to(self.device)
        self.num_anchors = int(num_anchors)

        # Snapshot the warm-start head state so per-matrix tuning resets
        # before each new matrix.
        self._initial_head_state = copy.deepcopy(self.head.state_dict())

        self.per_matrix_steps = int(per_matrix_steps)
        self.per_matrix_lr = float(per_matrix_lr)
        self.per_matrix_val_frac = float(per_matrix_val_frac)
        self.per_matrix_patch_size = int(per_matrix_patch_size)
        self.per_matrix_seed = int(per_matrix_seed)

        self.head.eval()

    # ----- the main inference path: backbone + head, base-class machinery -----

    def _run_head(self, x: torch.Tensor) -> torch.Tensor:
        observed = ~torch.isnan(x)
        target = torch.nan_to_num(x, nan=0.0)
        feats = self.model(x, return_embeddings=True)
        preds = self.head.predict(
            feats, target, observed, num_anchors=self.num_anchors,
        )
        out = torch.where(observed, target, preds)
        return out.to(torch.float32)

    def _run_model_direct(self, X_batch, X_last):
        with torch.no_grad():
            preds_batch = self._run_head(X_batch)
            preds_last = self._run_head(X_last) if X_last is not None else None
        return preds_batch, preds_last

    # -------- per-matrix tuning: Adam loop on the head's two scalars ---------

    def impute(self, X, *args, **kwargs):
        if self.per_matrix_steps > 0:
            self._tune_head_on_observed(X)
        return super().impute(X, *args, **kwargs)

    @torch.no_grad()
    def _reset_head_to_warmstart(self):
        self.head.load_state_dict(self._initial_head_state)

    def _tune_head_on_observed(self, X: np.ndarray) -> None:
        """Per-matrix Adam tuning of (log_lambda, log_gamma_scale).

        Samples random patches from X. Within each patch, randomly holds out
        a fraction of originally-observed cells as the tuning target. Loss
        is MSE on those held-out cells only.
        """
        # 1. Reset head to warm-start (so each new matrix starts fresh).
        self._reset_head_to_warmstart()
        self.head.train()

        # 2. Standardize the matrix the way ImputePFN.impute would.
        X = np.asarray(X, dtype=np.float64)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sigma = np.where(np.isnan(sigma), 1.0, sigma)
        X_std = (X - mu) / (sigma + 1e-16)

        R, C = X_std.shape
        rng = np.random.default_rng(self.per_matrix_seed)

        # 3. Optimizer over the head's two scalars.
        tunable = [self.head.log_lambda, self.head.log_gamma_scale]
        optimizer = torch.optim.Adam(tunable, lr=self.per_matrix_lr)

        device = self.device
        compute_dtype = torch.bfloat16 if "cuda" in device else torch.float32

        n_skipped = 0
        for step in range(self.per_matrix_steps):
            # 3a. Sample a random patch.
            max_r = min(R, self.per_matrix_patch_size)
            max_c = min(C, self.per_matrix_patch_size)
            rows = np.sort(rng.permutation(R)[:max_r])
            cols = np.sort(rng.permutation(C)[:max_c])
            x_patch = X_std[np.ix_(rows, cols)]                          # (max_r, max_c)

            observed_in_patch = ~np.isnan(x_patch)                       # bool
            obs_flat_idx = np.flatnonzero(observed_in_patch.ravel())
            if obs_flat_idx.size < 20:                                   # too sparse
                n_skipped += 1
                continue

            # 3b. Pick held-out cells.
            n_heldout = max(1, int(obs_flat_idx.size * self.per_matrix_val_frac))
            heldout_flat = rng.choice(obs_flat_idx, n_heldout, replace=False)

            # 3c. Construct tuning input: NaN at heldout + original-missing.
            x_tune = x_patch.copy()
            x_tune.ravel()[heldout_flat] = np.nan
            kept_observed = observed_in_patch.copy()
            kept_observed.ravel()[heldout_flat] = False                   # remove heldout

            # 3d. Forward through backbone (no grad, frozen).
            x_tune_t = torch.from_numpy(x_tune.astype(np.float32)).to(
                device=device, dtype=compute_dtype,
            ).unsqueeze(0)
            target_np = np.where(np.isnan(x_patch), 0.0, x_patch).astype(np.float32)
            target_t = torch.from_numpy(target_np).to(
                device=device, dtype=compute_dtype,
            ).unsqueeze(0)
            observed_t = torch.from_numpy(kept_observed).to(device=device).unsqueeze(0)

            with torch.no_grad():
                feats = self.model(x_tune_t, return_embeddings=True)

            # 3e. Head predicts at all not-observed positions (heldout + orig-missing).
            y_hat_full = self.head.predict(
                feats, target_t, observed_t,
                num_anchors=self.num_anchors,
                grad_enabled=True,
            )                                                            # (1, max_r, max_c) fp32

            # 3f. Loss only on heldout positions.
            y_hat_flat = y_hat_full[0].reshape(-1)
            heldout_idx_t = torch.from_numpy(heldout_flat).to(device=device).long()
            y_hat_heldout = y_hat_flat[heldout_idx_t]
            y_true_heldout = torch.from_numpy(
                x_patch.ravel()[heldout_flat].astype(np.float32)
            ).to(device=device)

            loss = ((y_hat_heldout - y_true_heldout) ** 2).mean()
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        self.head.eval()


class RFMLibImputePFN(ImputePFN):
    """ImputePFN variant that uses the xRFM library as a per-column test-time
    regressor on transformer embeddings.

    Pipeline per matrix:
      1. Backbone forward (tiled if needed) -> per-cell embeddings Phi of
         shape (R, C, D=768).
      2. For each column c with NaNs:
           features = Phi[:, c, :]                # (R, D)
           Fit xRFM library on
               X_train = features[observed_rows]
               y_train = X_std[observed_rows, c]
           Predict at features[missing_rows].
      3. De-standardize and return imputed matrix.

    No model training. Pure test-time per-matrix fit, identical in spirit to
    what xrfm_wrapper.py does on raw features — just operating on transformer
    embeddings instead.
    """

    def __init__(
        self,
        *args,
        rfm_params: dict | None = None,
        max_leaf_size: int = 60_000,
        refill_size: int = 1500,
        val_fraction: float = 0.2,
        random_state: int = 0,
        xrfm_kwargs: dict | None = None,
        xrfm_verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.entry_wise_features:
            raise ValueError(
                "RFMLibImputePFN requires entry_wise_features=False"
            )
        self.rfm_params = rfm_params if rfm_params is not None else _DEFAULT_RFM_PARAMS
        self.max_leaf_size = int(max_leaf_size)
        self.refill_size = int(refill_size)
        self.val_fraction = float(val_fraction)
        self.random_state = int(random_state)
        self.xrfm_kwargs = xrfm_kwargs or {}
        self.xrfm_verbose = bool(xrfm_verbose)

    @torch.no_grad()
    def _get_embeddings(self, X_std: np.ndarray) -> np.ndarray:
        """Tile-and-aggregate per-cell embeddings for the full matrix.

        Returns float32 ndarray of shape (R, C, D).
        Cells in overlapping tiles get embedding averaged across tiles.
        """
        R, C = X_std.shape
        compute_dtype = (
            torch.bfloat16 if "cuda" in str(self.device) else torch.float32
        )
        row_size = self.max_num_rows if self.max_num_rows else R
        col_size = self.max_num_cols if self.max_num_cols else C

        row_spans = self._get_chunk_spans(R, row_size)
        col_spans = self._get_chunk_spans(C, col_size)

        Phi_sum: np.ndarray | None = None
        counts = np.zeros((R, C), dtype=np.float32)

        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                X_chunk = X_std[r0:r1, c0:c1].astype(np.float32)
                x_t = (
                    torch.from_numpy(X_chunk)
                    .to(self.device)
                    .to(compute_dtype)
                    .unsqueeze(0)
                )
                feats = self.model(x_t, return_embeddings=True)  # (1, r, c, D)
                feats_np = feats[0].float().cpu().numpy()
                if Phi_sum is None:
                    D = feats_np.shape[-1]
                    Phi_sum = np.zeros((R, C, D), dtype=np.float32)
                Phi_sum[r0:r1, c0:c1] += feats_np
                counts[r0:r1, c0:c1] += 1.0

        if Phi_sum is None:
            raise RuntimeError("RFMLibImputePFN._get_embeddings produced no tiles.")
        counts_safe = np.maximum(counts, 1.0)
        return Phi_sum / counts_safe[:, :, None]

    def _fit_predict_column(
        self,
        features_obs: np.ndarray,
        y_obs: np.ndarray,
        features_query: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Fit xRFM on observed-row embeddings, predict at query rows."""
        # Lazy import so an xrfm install isn't required to import this module.
        from xrfm import xRFM

        n_obs = features_obs.shape[0]
        n_query = features_query.shape[0]
        if n_obs < 4 or n_query == 0:
            mean_y = float(y_obs.mean()) if n_obs > 0 else 0.0
            return np.full(n_query, mean_y, dtype=np.float32)

        n_val = max(1, int(round(n_obs * self.val_fraction)))
        n_val = min(n_val, n_obs - 1)
        idx = rng.permutation(n_obs)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        X_tr = features_obs[tr_idx].astype(np.float32, copy=False)
        y_tr = y_obs[tr_idx].astype(np.float32, copy=False)
        X_va = features_obs[val_idx].astype(np.float32, copy=False)
        y_va = y_obs[val_idx].astype(np.float32, copy=False)

        if self.xrfm_verbose:
            stdout_ctx = contextlib.nullcontext()
            stderr_ctx = contextlib.nullcontext()
        else:
            stdout_ctx = contextlib.redirect_stdout(io.StringIO())
            stderr_ctx = contextlib.redirect_stderr(io.StringIO())

        with stdout_ctx, stderr_ctx, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = xRFM(
                rfm_params=self.rfm_params,
                max_leaf_size=self.max_leaf_size,
                refill_size=self.refill_size,
                device=self.device,
                tuning_metric="mse",
                random_state=self.random_state,
                **self.xrfm_kwargs,
            )
            try:
                model.fit(X_tr, y_tr, X_va, y_va)
                pred = model.predict(features_query.astype(np.float32, copy=False))
            except Exception:
                # Fall back to mean on any xRFM internal failure.
                return np.full(n_query, float(y_obs.mean()), dtype=np.float32)

        pred = np.asarray(pred).reshape(-1).astype(np.float32)
        if pred.shape[0] != n_query:
            return np.full(n_query, float(y_obs.mean()), dtype=np.float32)
        return pred

    def impute(self, X: np.ndarray, *args, **kwargs):  # type: ignore[override]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X.shape}")
        R, C = X.shape

        # Standardize per column (same convention as ImputePFN.impute).
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sigma = np.where((sigma > 1e-12) & ~np.isnan(sigma), sigma, 1.0)
        X_std = (X - mu) / sigma

        # Get per-cell embeddings for the full matrix.
        Phi = self._get_embeddings(X_std)   # (R, C, D), float32

        # Column-wise xRFM fit + predict.
        rng = np.random.default_rng(self.random_state)
        X_imp = X_std.copy().astype(np.float32)

        cols_with_nan = [c for c in range(C) if np.isnan(X_std[:, c]).any()]
        for c in cols_with_nan:
            mask_c = np.isnan(X_std[:, c])
            obs_rows = ~mask_c
            miss_rows = mask_c
            if not miss_rows.any() or not obs_rows.any():
                continue

            features_obs = Phi[obs_rows, c, :]              # (n_obs, D)
            features_query = Phi[miss_rows, c, :]           # (n_miss, D)
            y_obs = X_std[obs_rows, c].astype(np.float32)

            pred = self._fit_predict_column(
                features_obs, y_obs, features_query, rng,
            )
            X_imp[miss_rows, c] = pred

        X_out = X_imp.astype(np.float64) * sigma + mu
        return X_out.astype(np.float32)


class MAGOPMatFormerImputer(ImputePFN):
    """MatFormer (MLP decoder kept) + per-matrix AGOP-style update of the
    FIRST layer of the backbone.

    Per UCI matrix at inference:
      1. Snapshot the first layer's weights (so we can reset between matrices).
      2. For each of `agop_iters` rounds:
         a. Sample a random 50x50 patch from the matrix.
         b. Forward through the full network (backbone + MLP decoder) -> y_hat.
            A forward hook captures h = output of the first layer (768-d per cell).
         c. From the patch's observed cells, sample K cells.
         d. For each sampled cell (r, c) compute  g_{r,c} = ∇_{h[r,c]} y_hat[r,c]
            via autograd.grad.
         e. M = (1/K) Σ g_{r,c} g_{r,c}^T   (D × D, D = 768).
         f. Normalize M ← M / max(|M|).
         g. sqrtM via SVD; pre-multiply the first layer's weights:
              W ← sqrtM @ W,   b ← sqrtM @ b.
      3. Run the full tiled MatFormer inference with updated first layer to
         predict missing cells.
      4. Restore the first layer to its initial weights.
    """

    def __init__(
        self,
        *args,
        agop_iters: int = 3,
        agop_num_samples: int = 64,
        agop_patch_rows: int = 50,
        agop_patch_cols: int = 50,
        agop_seed: int = 0,
        agop_max_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.entry_wise_features:
            raise ValueError(
                "MAGOPMatFormerImputer requires entry_wise_features=False"
            )
        # The first layer of the whole architecture: Linear(1, 768) inside
        # feature_encoder.observed_linear_layer.
        self.first_layer = self.model.feature_encoder.observed_linear_layer[0]
        # Snapshot initial weights (CPU clone) for per-matrix reset.
        self._first_W_init = self.first_layer.weight.detach().cpu().clone()
        self._first_b_init = (
            self.first_layer.bias.detach().cpu().clone()
            if self.first_layer.bias is not None else None
        )
        self.agop_iters = int(agop_iters)
        self.agop_num_samples = int(agop_num_samples)
        self.agop_patch_rows = int(agop_patch_rows)
        self.agop_patch_cols = int(agop_patch_cols)
        self.agop_seed = int(agop_seed)
        self.agop_max_scale = float(agop_max_scale)

    def _restore_first_layer(self):
        with torch.no_grad():
            self.first_layer.weight.data.copy_(
                self._first_W_init.to(self.first_layer.weight.device,
                                       dtype=self.first_layer.weight.dtype)
            )
            if self._first_b_init is not None:
                self.first_layer.bias.data.copy_(
                    self._first_b_init.to(self.first_layer.bias.device,
                                           dtype=self.first_layer.bias.dtype)
                )

    def _sample_patch(self, X_std: np.ndarray, rng: np.random.Generator):
        R, C = X_std.shape
        R_p = min(R, self.agop_patch_rows)
        C_p = min(C, self.agop_patch_cols)
        rows = np.sort(rng.permutation(R)[:R_p])
        cols = np.sort(rng.permutation(C)[:C_p])
        return rows, cols

    def _agop_update_one_iter(
        self,
        X_std: np.ndarray,
        rng: np.random.Generator,
    ) -> bool:
        """One AGOP iteration: sample a patch, compute per-cell gradients,
        form M, update first layer weights. Returns True if update happened,
        False if patch had no usable observed cells (no update).
        """
        rows, cols = self._sample_patch(X_std, rng)
        x_sub = X_std[np.ix_(rows, cols)]                # (R_p, C_p) with NaN
        observed = ~np.isnan(x_sub)
        obs_pos = np.argwhere(observed)                  # list of (r, c)
        if len(obs_pos) < 4:
            return False
        K = min(self.agop_num_samples, len(obs_pos))
        pick = rng.choice(len(obs_pos), K, replace=False)
        sampled = obs_pos[pick]                          # (K, 2)

        device = self.device
        compute_dtype = (
            torch.bfloat16 if "cuda" in str(device) else torch.float32
        )

        x_t = (
            torch.from_numpy(x_sub.astype(np.float32))
            .to(device).to(compute_dtype).unsqueeze(0)   # (1, R_p, C_p)
        )

        # Set up the forward hook to capture the first-layer output.
        captured: list[torch.Tensor] = []
        def _hook(_module, _input, output):
            captured.append(output)
        handle = self.first_layer.register_forward_hook(_hook)

        try:
            # Forward needs grad for autograd.grad to work through h.
            # Backbone + MLP decoder.
            torch.set_grad_enabled(True)
            self.model.eval()
            preds = self.model(x_t)                       # (1, R_p, C_p, num_outputs)
            # Cast to fp32 for the median + autograd to be stable.
            preds_f32 = preds.float()
            y_hat = self.bar_distribution.median(logits=preds_f32)  # (1, R_p, C_p)

            if not captured:
                handle.remove()
                return False
            h = captured[0]                               # (1, R_p, C_p, D)
            h_f32 = h.float()                             # for stable autograd

            # Per-cell gradient computation.
            D = h_f32.shape[-1]
            G = torch.zeros(K, D, device=device, dtype=torch.float32)
            for i in range(K):
                r_i, c_i = int(sampled[i, 0]), int(sampled[i, 1])
                scalar = y_hat[0, r_i, c_i]
                # gradient of this scalar w.r.t. h
                grad_h = torch.autograd.grad(
                    scalar, h, retain_graph=(i < K - 1), allow_unused=False
                )[0]                                       # (1, R_p, C_p, D)
                G[i] = grad_h[0, r_i, c_i].float()
        finally:
            handle.remove()
            torch.set_grad_enabled(False)

        # AGOP M = (1/K) G^T G
        M = (G.T @ G) / float(K)
        m_max = M.abs().max().clamp_min(1e-12)
        M_norm = M / m_max                                  # normalize
        # Scale so sqrt(M) has bounded operator norm.
        M_norm = M_norm * float(self.agop_max_scale)

        # sqrtM via symmetric eigendecomp (M is symmetric PSD by construction).
        # Use torch.linalg.eigh for numerical stability.
        try:
            evals, evecs = torch.linalg.eigh(M_norm)
        except Exception:
            return False
        evals = evals.clamp_min(0.0)
        sqrtM = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T  # (D, D)

        # Update first layer: W ← sqrtM @ W, b ← sqrtM @ b.
        with torch.no_grad():
            new_W = sqrtM.to(self.first_layer.weight.dtype) @ self.first_layer.weight.data
            self.first_layer.weight.data.copy_(new_W)
            if self.first_layer.bias is not None:
                new_b = (
                    sqrtM.to(self.first_layer.bias.dtype) @ self.first_layer.bias.data
                )
                self.first_layer.bias.data.copy_(new_b)

        return True

    def impute(self, X: np.ndarray, *args, **kwargs):  # type: ignore[override]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

        # Standardize per column.
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sigma = np.where((sigma > 1e-12) & ~np.isnan(sigma), sigma, 1.0)
        X_std = (X - mu) / sigma

        # Reset first layer to warm-start.
        self._restore_first_layer()

        # AGOP loop on STANDARDIZED matrix.
        rng = np.random.default_rng(self.agop_seed)
        for _ in range(self.agop_iters):
            self._agop_update_one_iter(X_std, rng)

        # Final tile-and-impute via ImputePFN's standard pipeline with the
        # updated first layer. We pass the ORIGINAL X (not X_std) since
        # ImputePFN.impute does its own standardization.
        with torch.no_grad():
            X_imputed = super().impute(X)

        # Restore first layer for the next matrix.
        self._restore_first_layer()

        return X_imputed


class DeltaMatFormerImputer(ImputePFN):
    """Per-matrix shift of Φ via gradient descent on a single shared Δ ∈ R^D.

    Per matrix:
      1. Compute per-cell embeddings Φ over the full matrix (tiled, no_grad).
      2. Initialize Δ = 0 (shape D, shared across all cells).
      3. For T steps:
           Phi_shifted = Phi + Δ (broadcast)
           y_hat = MLP_decoder_median(Phi_shifted)
           loss = MSE(y_hat[observed], X_std[observed])
           Adam step on Δ only.
      4. Final prediction: MLP_decoder_median(Phi + Δ) at MISSING cells.
      5. Δ is discarded; no model weights changed.
    """

    def __init__(
        self,
        *args,
        delta_steps: int = 20,
        delta_lr: float = 1e-2,
        delta_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.entry_wise_features:
            raise ValueError(
                "DeltaMatFormerImputer requires entry_wise_features=False"
            )
        if not hasattr(self.model, "decoder") or self.model.decoder is None:
            raise ValueError(
                "DeltaMatFormerImputer requires the model to have an MLP "
                "decoder (the checkpoint must include decoder weights)."
            )
        self.delta_steps = int(delta_steps)
        self.delta_lr = float(delta_lr)
        self.delta_seed = int(delta_seed)

    @torch.no_grad()
    def _get_full_phi(self, X_std: np.ndarray) -> torch.Tensor:
        """Tile-and-aggregate per-cell embeddings for the full matrix. Returns
        (R, C, D) tensor on device, fp32 (averaged across overlapping tiles).
        """
        R, C = X_std.shape
        compute_dtype = (
            torch.bfloat16 if "cuda" in str(self.device) else torch.float32
        )
        row_size = self.max_num_rows if self.max_num_rows else R
        col_size = self.max_num_cols if self.max_num_cols else C
        row_spans = self._get_chunk_spans(R, row_size)
        col_spans = self._get_chunk_spans(C, col_size)

        Phi_sum: np.ndarray | None = None
        counts = np.zeros((R, C), dtype=np.float32)
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                X_chunk = X_std[r0:r1, c0:c1].astype(np.float32)
                x_t = (
                    torch.from_numpy(X_chunk).to(self.device).to(compute_dtype).unsqueeze(0)
                )
                feats = self.model(x_t, return_embeddings=True)
                feats_np = feats[0].float().cpu().numpy()
                if Phi_sum is None:
                    D = feats_np.shape[-1]
                    Phi_sum = np.zeros((R, C, D), dtype=np.float32)
                Phi_sum[r0:r1, c0:c1] += feats_np
                counts[r0:r1, c0:c1] += 1.0
        if Phi_sum is None:
            raise RuntimeError("DeltaMatFormerImputer._get_full_phi produced no tiles.")
        counts_safe = np.maximum(counts, 1.0)
        Phi = Phi_sum / counts_safe[:, :, None]
        return torch.from_numpy(Phi).to(self.device).float()

    def impute(self, X: np.ndarray, *args, **kwargs):  # type: ignore[override]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

        # Standardize per column (matches ImputePFN's convention).
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sigma = np.where((sigma > 1e-12) & ~np.isnan(sigma), sigma, 1.0)
        X_std = (X - mu) / sigma                                # (R, C) with NaN at missing

        # Embeddings for the full matrix.
        Phi = self._get_full_phi(X_std)                          # (R, C, D), fp32, device

        # Init shared Δ (one D-dim vector, applied to all cells).
        D = Phi.shape[-1]
        torch.manual_seed(self.delta_seed)
        Delta = torch.zeros(D, device=self.device, dtype=torch.float32, requires_grad=True)

        # Observed mask + standardized targets (NaN-filled with 0 outside obs).
        observed_mask = torch.from_numpy(~np.isnan(X_std)).to(self.device)
        targets = torch.from_numpy(
            np.nan_to_num(X_std, nan=0.0).astype(np.float32)
        ).to(self.device)

        # Optimizer on Δ only.
        optimizer = torch.optim.Adam([Delta], lr=self.delta_lr)
        compute_dtype = (
            torch.bfloat16 if "cuda" in str(self.device) else torch.float32
        )

        for t in range(self.delta_steps):
            optimizer.zero_grad(set_to_none=True)
            Phi_shifted = (Phi + Delta[None, None, :]).to(compute_dtype).unsqueeze(0)  # (1, R, C, D)
            logits = self.model.decoder(Phi_shifted)             # (1, R, C, num_outputs)
            y_hat = self.bar_distribution.median(logits=logits.float()).squeeze(0)  # (R, C)
            err = (y_hat - targets) ** 2
            if observed_mask.sum() == 0:
                break
            loss = err[observed_mask].mean()
            if not torch.isfinite(loss):
                tqdm_write = None
                break
            loss.backward()
            # Guard: skip step if Δ grad is non-finite.
            if Delta.grad is not None and not torch.isfinite(Delta.grad).all():
                Delta.grad = None
                continue
            optimizer.step()

        # Final prediction with the optimized Δ.
        with torch.no_grad():
            Phi_shifted = (Phi + Delta[None, None, :]).to(compute_dtype).unsqueeze(0)
            logits = self.model.decoder(Phi_shifted)
            y_hat = (
                self.bar_distribution.median(logits=logits.float())
                .squeeze(0).cpu().numpy()
            )

        # Fill MISSING cells with predictions; observed cells keep their input values.
        miss_mask = np.isnan(X_std)
        X_imp = X_std.copy().astype(np.float32)
        X_imp[miss_mask] = y_hat[miss_mask]

        # De-standardize.
        X_out = X_imp.astype(np.float64) * sigma + mu
        return X_out.astype(np.float32)


class BoostMatFormerImputer(ImputePFN):
    """MatFormer's MLP prediction + xRFM column-wise residual correction.

    Per matrix:
      1. Run MatFormer (backbone + MLP decoder) → y_mlp at every cell.
      2. Compute residuals in standardized space at observed cells:
             r_std[obs, c] = X_std[obs, c] − y_mlp_std[obs, c]
      3. For each column c with NaNs:
           features_c = mean-fill + missingness-mask of OTHER columns
                        (same construction as xrfm_wrapper.py)
           Fit xRFM on (features_c[obs], r_std[obs, c]).
           Predict r_pred at features_c[missing].
      4. Final at missing cells:  y_final_std = y_mlp_std + alpha * r_pred,
         then de-standardize to original units.

    Floor: equivalent to MatFormer (when alpha=0 or xRFM predicts ~0 residual).
    Possible ceiling: better than either MatFormer or raw xRFM if their
    errors are partly uncorrelated.
    """

    def __init__(
        self,
        *args,
        rfm_params: dict | None = None,
        xrfm_max_leaf_size: int = 60_000,
        xrfm_refill_size: int = 1500,
        xrfm_val_fraction: float = 0.2,
        xrfm_random_state: int = 0,
        blend_alpha: float = 1.0,
        adaptive_alpha: bool = False,
        alpha_grid: list[float] | None = None,
        alpha_val_frac: float = 0.1,
        adaptive_alpha_threshold: float = 0.05,
        adaptive_alpha_method: str = "grid",
        alpha_shrinkage: float = 0.5,
        alpha_max_cap: float = 0.1,
        residual_learner: str = "xrfm",
        context_scale: float = 1.0,
        xrfm_verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.entry_wise_features:
            raise ValueError(
                "BoostMatFormerImputer requires entry_wise_features=False"
            )
        if not hasattr(self.model, "decoder") or self.model.decoder is None:
            raise ValueError(
                "BoostMatFormerImputer requires the model to have an MLP decoder."
            )
        self.rfm_params = rfm_params if rfm_params is not None else _DEFAULT_RFM_PARAMS
        self.xrfm_max_leaf_size = int(xrfm_max_leaf_size)
        self.xrfm_refill_size = int(xrfm_refill_size)
        self.xrfm_val_fraction = float(xrfm_val_fraction)
        self.xrfm_random_state = int(xrfm_random_state)
        self.blend_alpha = float(blend_alpha)
        self.adaptive_alpha = bool(adaptive_alpha)
        self.alpha_grid = (
            list(alpha_grid)
            if alpha_grid is not None
            else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        )
        self.alpha_val_frac = float(alpha_val_frac)
        self.adaptive_alpha_threshold = float(adaptive_alpha_threshold)
        self.adaptive_alpha_method = str(adaptive_alpha_method).lower()
        if self.adaptive_alpha_method not in {"grid", "bias_variance"}:
            raise ValueError(
                f"adaptive_alpha_method={adaptive_alpha_method!r} not in "
                "{'grid', 'bias_variance'}"
            )
        self.alpha_shrinkage = float(alpha_shrinkage)
        self.alpha_max_cap = float(alpha_max_cap)
        self.residual_learner = str(residual_learner).lower()
        if self.residual_learner not in {
            "xrfm", "xgboost", "lightgbm", "random_forest", "kernel_ridge", "linear"
        }:
            raise ValueError(
                f"Unknown residual_learner={self.residual_learner}; expected one of "
                "{xrfm, xgboost, lightgbm, random_forest, kernel_ridge, linear}."
            )
        self.context_scale = float(context_scale)
        self.xrfm_verbose = bool(xrfm_verbose)

    @staticmethod
    def _standardize_context(ctx: np.ndarray) -> np.ndarray:
        """Z-score each column of a context matrix so xRFM sees comparable
        per-dim scales. Empty / zero-variance columns are passed through as 0."""
        ctx = np.asarray(ctx, dtype=np.float32)
        if ctx.ndim != 2 or ctx.size == 0:
            return ctx
        mu = np.nanmean(ctx, axis=0)
        sd = np.nanstd(ctx, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where((sd > 1e-12) & np.isfinite(sd), sd, 1.0)
        return (ctx - mu) / sd

    @staticmethod
    def _mean_fill(X: np.ndarray) -> np.ndarray:
        out = X.copy().astype(np.float32)
        col_means = np.nanmean(out, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(out))
        out[inds] = np.take(col_means, inds[1])
        return out

    @staticmethod
    def _build_features(X_filled: np.ndarray, mask: np.ndarray,
                        target_col: int, d: int,
                        row_context: np.ndarray | None = None,
                        col_context_row: np.ndarray | None = None) -> np.ndarray:
        """Per-row feature vector for predicting column `target_col`.

        Base = [mean-filled other columns, missingness mask of other columns].
        Optional row_context (R, d_r) gets concatenated per-row.
        Optional col_context_row (d_c,) — context vector OF the target column —
        gets broadcast to every row. Useful for global learners; for per-column
        xRFM it's constant within the fit and not informative on its own, but
        we still allow it.
        """
        feats = np.concatenate([X_filled, mask.astype(np.float32)], axis=1)
        keep = np.ones(2 * d, dtype=bool)
        keep[target_col] = False
        keep[target_col + d] = False
        base = feats[:, keep]
        extras = []
        if row_context is not None:
            extras.append(row_context.astype(np.float32))
        if col_context_row is not None:
            cc = np.asarray(col_context_row, dtype=np.float32).reshape(1, -1)
            extras.append(np.broadcast_to(cc, (base.shape[0], cc.shape[1])).copy())
        if extras:
            base = np.concatenate([base, *extras], axis=1)
        return base

    def _fit_predict_residual(
        self,
        features_obs: np.ndarray,
        residual_obs: np.ndarray,
        features_query: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_obs = features_obs.shape[0]
        n_q = features_query.shape[0]
        if n_obs < 4 or n_q == 0:
            return np.zeros(n_q, dtype=np.float32)
        # Train/val split shared across all learners (for xRFM's early-stop
        # and for fair comparison when learners use it).
        n_val = max(1, int(round(n_obs * self.xrfm_val_fraction)))
        n_val = min(n_val, n_obs - 1)
        idx = rng.permutation(n_obs)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        X_tr = features_obs[tr_idx].astype(np.float32, copy=False)
        y_tr = residual_obs[tr_idx].astype(np.float32, copy=False)
        X_va = features_obs[val_idx].astype(np.float32, copy=False)
        y_va = residual_obs[val_idx].astype(np.float32, copy=False)
        X_q = features_query.astype(np.float32, copy=False)

        if self.xrfm_verbose:
            stdout_ctx = contextlib.nullcontext()
            stderr_ctx = contextlib.nullcontext()
        else:
            stdout_ctx = contextlib.redirect_stdout(io.StringIO())
            stderr_ctx = contextlib.redirect_stderr(io.StringIO())

        try:
            with stdout_ctx, stderr_ctx, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.residual_learner == "xrfm":
                    from xrfm import xRFM
                    model = xRFM(
                        rfm_params=self.rfm_params,
                        max_leaf_size=self.xrfm_max_leaf_size,
                        refill_size=self.xrfm_refill_size,
                        device=self.device,
                        tuning_metric="mse",
                        random_state=self.xrfm_random_state,
                    )
                    model.fit(X_tr, y_tr, X_va, y_va)
                    pred = model.predict(X_q)
                elif self.residual_learner == "xgboost":
                    import xgboost as xgb
                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        tree_method="hist",
                        device="cuda" if "cuda" in str(self.device) else "cpu",
                        early_stopping_rounds=20,
                        random_state=self.xrfm_random_state,
                        verbosity=0,
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    pred = model.predict(X_q)
                elif self.residual_learner == "lightgbm":
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=self.xrfm_random_state,
                        verbosity=-1,
                    )
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(20, verbose=False)],
                    )
                    pred = model.predict(X_q)
                elif self.residual_learner == "random_forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=200, max_depth=None,
                        random_state=self.xrfm_random_state, n_jobs=-1,
                    )
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_q)
                elif self.residual_learner == "kernel_ridge":
                    from sklearn.kernel_ridge import KernelRidge
                    model = KernelRidge(alpha=1e-2, kernel="laplacian", gamma=0.1)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_q)
                elif self.residual_learner == "linear":
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0, random_state=self.xrfm_random_state)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_q)
                else:
                    raise ValueError(f"unknown residual_learner={self.residual_learner}")
        except Exception:
            return np.zeros(n_q, dtype=np.float32)

        pred = np.asarray(pred).reshape(-1).astype(np.float32)
        if pred.shape[0] != n_q:
            return np.zeros(n_q, dtype=np.float32)
        return pred

    def impute(self, X: np.ndarray, *args,
               row_context: np.ndarray | None = None,
               col_context: np.ndarray | None = None,
               **kwargs):  # type: ignore[override]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X.shape}")
        R, C = X.shape

        # Validate and standardize+scale optional context arrays. Standardization
        # puts each context dim on stdev 1; multiplying by context_scale lifts
        # them so they contribute meaningfully to xRFM's diagonal-Laplace kernel
        # distance (which is otherwise dominated by the 2*(C-1) cell-value
        # features). context_scale = sqrt(2*(C-1) / d_context) gives roughly
        # equal weight; default 1.0 preserves the original drown-the-context
        # behavior.
        if row_context is not None:
            row_context = np.asarray(row_context, dtype=np.float32)
            if row_context.ndim != 2 or row_context.shape[0] != R:
                raise ValueError(
                    f"row_context must be (R={R}, d_r); got {row_context.shape}"
                )
            row_context = self._standardize_context(row_context) * self.context_scale
        if col_context is not None:
            col_context = np.asarray(col_context, dtype=np.float32)
            if col_context.ndim != 2 or col_context.shape[0] != C:
                raise ValueError(
                    f"col_context must be (C={C}, d_c); got {col_context.shape}"
                )
            col_context = self._standardize_context(col_context) * self.context_scale

        # 1. MatFormer's predictions at every cell.
        with torch.no_grad():
            _X_imputed_mlp, X_full = super().impute(X, return_full=True)
        X_full = np.asarray(X_full, dtype=np.float64)

        # 2. Standardize X and X_full with the same per-col stats.
        col_mean = np.nanmean(X, axis=0)
        col_std = np.nanstd(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        col_std = np.where(np.isnan(col_std), 1.0, col_std)
        col_std = np.where(col_std > 1e-12, col_std, 1.0)
        X_std = (X - col_mean) / col_std
        y_mlp_std = (X_full - col_mean) / col_std

        # 3. Standardized residuals at observed cells.
        residual_std = X_std - y_mlp_std    # NaN at missing positions

        # 4. xRFM residual correction per column with NaNs.
        rng = np.random.default_rng(self.xrfm_random_state)
        mask = np.isnan(X_std).astype(np.float32)
        X_filled = self._mean_fill(X_std)

        r_pred_full = np.zeros((R, C), dtype=np.float32)
        cols_with_nan = [c for c in range(C) if mask[:, c].any()]

        if self.adaptive_alpha:
            # Per-matrix α: hold out a fraction of observed cells per column
            # as α-val, fit xRFM on the rest, predict at val + missing,
            # then sweep α grid to find α* minimizing val RMSE.
            r_pred_val_per_col: dict[int, np.ndarray] = {}
            val_rows_per_col: dict[int, np.ndarray] = {}
            for c in cols_with_nan:
                obs_rows_c = np.where(mask[:, c] == 0.0)[0]
                miss_rows_c = np.where(mask[:, c] == 1.0)[0]
                if miss_rows_c.size == 0 or obs_rows_c.size < 10:
                    # Not enough to split; fit on all observed for this column.
                    features = self._build_features(
                        X_filled, mask, c, C,
                        row_context=row_context,
                        col_context_row=(col_context[c] if col_context is not None else None),
                    )
                    if obs_rows_c.size >= 4 and miss_rows_c.size > 0:
                        r_pred_miss = self._fit_predict_residual(
                            features[obs_rows_c],
                            residual_std[obs_rows_c, c].astype(np.float32),
                            features[miss_rows_c],
                            rng,
                        )
                        r_pred_full[miss_rows_c, c] = r_pred_miss
                    continue

                # Split observed rows into α-train and α-val.
                n_val = max(2, int(round(obs_rows_c.size * self.alpha_val_frac)))
                n_val = min(n_val, obs_rows_c.size - 4)  # keep ≥4 for xRFM
                perm = rng.permutation(obs_rows_c.size)
                val_rows = obs_rows_c[perm[:n_val]]
                train_rows = obs_rows_c[perm[n_val:]]

                features = self._build_features(
                    X_filled, mask, c, C,
                    row_context=row_context,
                    col_context_row=(col_context[c] if col_context is not None else None),
                )
                features_train = features[train_rows]
                target_train = residual_std[train_rows, c].astype(np.float32)
                features_query = np.concatenate(
                    [features[val_rows], features[miss_rows_c]], axis=0
                )
                r_pred = self._fit_predict_residual(
                    features_train, target_train, features_query, rng,
                )
                r_pred_val = r_pred[: val_rows.size]
                r_pred_miss = r_pred[val_rows.size:]

                r_pred_full[miss_rows_c, c] = r_pred_miss
                r_pred_val_per_col[c] = r_pred_val
                val_rows_per_col[c] = val_rows

            if self.adaptive_alpha_method == "bias_variance":
                # Plug-in α* from MSE decomposition:
                #   MSE_boost - MSE_mf = 2α·cov(e_mf, r_pred) + α²·E[r_pred²]
                # minimized at α* = -cov / E[r²]. Estimate cov and E[r²] on the
                # held-out val cells (which xRFM didn't train on), then shrink
                # and clip to be robust to val→test distribution shift.
                e_pool, r_pool = [], []
                for c, r_pred_val in r_pred_val_per_col.items():
                    val_rows = val_rows_per_col[c]
                    e_pool.append(y_mlp_std[val_rows, c] - X_std[val_rows, c])
                    r_pool.append(np.asarray(r_pred_val, dtype=np.float64))
                if e_pool:
                    e_arr = np.concatenate(e_pool)
                    r_arr = np.concatenate(r_pool)
                    E_r2 = float(np.mean(r_arr ** 2))
                    if E_r2 > 1e-12 and e_arr.size > 1:
                        cov_e_r = float(np.cov(e_arr, r_arr, ddof=0)[0, 1])
                        alpha_star_raw = -cov_e_r / E_r2
                    else:
                        cov_e_r = 0.0
                        alpha_star_raw = 0.0
                else:
                    cov_e_r = 0.0
                    alpha_star_raw = 0.0
                alpha_chosen = self.alpha_shrinkage * alpha_star_raw
                alpha_chosen = float(np.clip(alpha_chosen, 0.0, self.alpha_max_cap))
                effective_alpha = alpha_chosen
                self._last_alpha_scores = {
                    "alpha_star_raw": alpha_star_raw,
                    "cov_e_r": cov_e_r,
                    "E_r_pred_sq": E_r2 if e_pool else 0.0,
                }
                self._last_alpha_star = alpha_chosen
            else:
                # Sweep α grid on the aggregated α-val cells.
                alpha_scores: dict[float, float] = {}
                for alpha in self.alpha_grid:
                    total_sse = 0.0
                    total_n = 0
                    for c, r_pred_val in r_pred_val_per_col.items():
                        val_rows = val_rows_per_col[c]
                        y_pred_val_std = (
                            y_mlp_std[val_rows, c] + alpha * r_pred_val
                        )
                        y_true_val_std = X_std[val_rows, c]
                        total_sse += float(np.sum((y_pred_val_std - y_true_val_std) ** 2))
                        total_n += int(val_rows.size)
                    if total_n == 0:
                        continue
                    mse = total_sse / total_n
                    alpha_scores[alpha] = mse

                # "Smallest α with meaningful improvement" selection:
                # the val signal is biased toward high α (val cells were originally
                # observed, so xRFM extrapolates well to them but not to truly
                # missing cells). Counter that bias by preferring α=0 unless a
                # nonzero α yields val_MSE improvement > threshold of α=0's MSE,
                # and among qualifying α's, pick the SMALLEST.
                mse_at_zero = alpha_scores.get(0.0)
                best_alpha = 0.0
                if mse_at_zero is not None and mse_at_zero > 0:
                    threshold_mse = mse_at_zero * (1.0 - self.adaptive_alpha_threshold)
                    for alpha in sorted(self.alpha_grid):
                        if alpha == 0.0:
                            continue
                        if alpha_scores.get(alpha, float("inf")) <= threshold_mse:
                            best_alpha = alpha
                            break
                effective_alpha = best_alpha
                self._last_alpha_scores = alpha_scores
                self._last_alpha_star = best_alpha
        else:
            # Fixed α: fit xRFM on ALL observed cells per column.
            for c in cols_with_nan:
                obs_rows_c = mask[:, c] == 0.0
                miss_rows_c = ~obs_rows_c
                if not miss_rows_c.any():
                    continue
                features = self._build_features(
                    X_filled, mask, c, C,
                    row_context=row_context,
                    col_context_row=(col_context[c] if col_context is not None else None),
                )
                features_obs = features[obs_rows_c]
                features_miss = features[miss_rows_c]
                target_obs = residual_std[obs_rows_c, c].astype(np.float32)
                r_pred = self._fit_predict_residual(
                    features_obs, target_obs, features_miss, rng,
                )
                r_pred_full[miss_rows_c, c] = r_pred
            effective_alpha = self.blend_alpha

        # 5. Combine, then de-standardize.
        X_std_final = X_std.copy()
        miss_mask = np.isnan(X_std)
        X_std_final[miss_mask] = (
            y_mlp_std + effective_alpha * r_pred_full
        )[miss_mask]
        X_out = X_std_final * col_std + col_mean
        return X_out.astype(np.float32)
