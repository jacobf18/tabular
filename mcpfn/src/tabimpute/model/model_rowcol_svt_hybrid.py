from __future__ import annotations

import importlib.resources
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from tabimpute.model.model_new_stable import RMSNorm
from tabimpute.model.model_rowcol_latent_attention import (
    TabImputeModelRowColumnLatentAttention,
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


class TabImputeModelRowColumnSVTHybrid(TabImputeModelRowColumnLatentAttention):
    """Row/column encoder with differentiable SoftImpute/SVT refinement."""

    def __init__(
        self,
        *args,
        svt_init_rank: int = 10,
        svt_steps: int = 4,
        svt_max_rank: int | None = 20,
        svt_solver_precision: str = "inherit",
        svt_threshold_fraction_init: float = 0.02,
        svt_threshold_fraction_min: float = 1e-4,
        svt_threshold_fraction_max: float | None = None,
        svt_threshold_mode: str = "constant",
        svt_threshold_size_scale_init: float = 1.2,
        svt_update_damping: float = 1.0,
        svt_init_scale: float = 0.035,
        logit_sigma_init: float = 0.08,
        logit_sigma_min: float = 0.008,
        use_neural_residual: bool = False,
        neural_residual_rank: int | None = None,
        neural_residual_scale_init: float = 0.0,
        use_svt_feature_decoder: bool = False,
        use_svt_magnitude_channel: bool = False,
        use_svt_output_correction: bool = False,
        use_cell_query_decoder: bool = False,
        cell_query_width: int = 128,
        cell_query_heads: int = 4,
        cell_query_layers: int = 2,
        cell_query_chunk_size: int = 128,
        cell_query_checkpoint_chunks: bool = True,
        cell_query_axis_balanced: bool = False,
        svt_feature_decoder_scale_init: float = 0.1,
        svt_feature_decoder_hidden_size: int | None = None,
        use_observed_neighbor_features: bool = False,
        observed_neighbor_temperature: float = 0.20,
        observed_neighbor_topk: int = 64,
        use_observed_attention_decoder: bool = False,
        observed_attention_topk: int = 64,
        observed_attention_dim: int = 128,
        observed_attention_temperature_init: float = 1.0,
        observed_attention_scale_init: float = 0.05,
        observed_attention_hidden_size: int | None = None,
        use_empirical_bayes_decoder: bool = False,
        empirical_bayes_ridge_init: float = 0.02,
        empirical_bayes_ridge_min: float = 1e-5,
        empirical_bayes_shrinkage_init: float = 0.05,
        empirical_bayes_feature_scale_init: float = 0.10,
        empirical_bayes_hidden_size: int | None = None,
        empirical_bayes_solve_steps: int = 12,
        num_prior_adapters: int = 0,
        default_prior_index: int | None = None,
        prior_adapter_scale_init: float = 0.05,
        prior_adapter_calibration_scale_init: float = 0.02,
        prior_svt_steps: list[int] | tuple[int, ...] | None = None,
        svt_mixture_steps: list[int] | tuple[int, ...] | None = None,
        svt_mixture_hidden_size: int = 512,
        svt_mixture_temperature: float = 1.0,
        svt_mixture_initial_logit: float = 4.0,
        svt_mixture_other_logit: float = -2.0,
        svt_mixture_detach_refinement: bool = True,
        svt_mixture_spectral_feature_count: int = 0,
        svt_mixture_missingness_feature_count: int = 0,
        svt_mixture_spectrum_encoder_size: int = 0,
        svt_mixture_spectrum_encoder_layers: int = 1,
        svt_mixture_spectrum_encoder_heads: int = 4,
        svt_mixture_spectrum_encoder_dropout: float = 0.0,
        svt_adaptive_threshold: bool = False,
        svt_adaptive_threshold_hidden_size: int | None = None,
        svt_adaptive_threshold_min_scale: float = 0.25,
        svt_adaptive_threshold_max_scale: float = 4.0,
        svt_adaptive_threshold_init_scale: float = 1.0,
        svt_mixture_policy: str = "controller",
        svt_mixture_cv_fraction: float = 0.0,
        svt_mixture_cv_temperature: float = 0.01,
        svt_mixture_cv_seed: int = 17,
        svt_mixture_cv_min_count: int = 32,
        svt_mixture_cv_reencode: bool = False,
        svt_mixture_include_cv_features: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (use_svt_magnitude_channel or use_svt_output_correction or use_cell_query_decoder) and not use_svt_feature_decoder:
            raise ValueError("SVT magnitude/correction options require the feature decoder")
        self.use_svt_magnitude_channel = bool(use_svt_magnitude_channel)
        self.use_svt_output_correction = bool(use_svt_output_correction)
        self.use_cell_query_decoder = bool(use_cell_query_decoder)
        self.last_prediction_location = None
        self.last_svt_points = None
        if svt_init_rank <= 0:
            raise ValueError(f"svt_init_rank must be positive, got {svt_init_rank}.")
        if svt_steps < 0:
            raise ValueError(f"svt_steps must be non-negative, got {svt_steps}.")
        if svt_solver_precision not in {"inherit", "float32"}:
            raise ValueError(
                "svt_solver_precision must be 'inherit' or 'float32', got "
                f"{svt_solver_precision!r}."
            )
        if svt_max_rank is not None and svt_max_rank <= 0:
            raise ValueError(f"svt_max_rank must be positive, got {svt_max_rank}.")
        if not 0.0 < svt_update_damping <= 1.0:
            raise ValueError(
                "svt_update_damping must be in (0, 1], got "
                f"{svt_update_damping}."
            )
        if not 0.0 < svt_threshold_fraction_init < 1.0:
            raise ValueError(
                "svt_threshold_fraction_init must be in (0, 1), got "
                f"{svt_threshold_fraction_init}."
            )
        if not 0.0 <= svt_threshold_fraction_min < svt_threshold_fraction_init:
            raise ValueError(
                "svt_threshold_fraction_min must be non-negative and smaller "
                "than svt_threshold_fraction_init."
            )
        if svt_threshold_fraction_max is not None and (
            svt_threshold_fraction_max <= svt_threshold_fraction_min
        ):
            raise ValueError(
                "svt_threshold_fraction_max must be greater than "
                "svt_threshold_fraction_min."
            )
        if svt_threshold_mode not in {"constant", "inverse_size"}:
            raise ValueError(
                "svt_threshold_mode must be 'constant' or 'inverse_size', got "
                f"{svt_threshold_mode!r}."
            )
        if svt_threshold_size_scale_init <= 0.0:
            raise ValueError("svt_threshold_size_scale_init must be positive.")
        if logit_sigma_init <= logit_sigma_min:
            raise ValueError("logit_sigma_init must be greater than logit_sigma_min.")
        if neural_residual_rank is not None and neural_residual_rank <= 0:
            raise ValueError("neural_residual_rank must be positive.")
        if observed_neighbor_temperature <= 0.0:
            raise ValueError("observed_neighbor_temperature must be positive.")
        if observed_neighbor_topk <= 0:
            raise ValueError("observed_neighbor_topk must be positive.")
        if observed_attention_topk <= 0:
            raise ValueError("observed_attention_topk must be positive.")
        if observed_attention_dim <= 0:
            raise ValueError("observed_attention_dim must be positive.")
        if observed_attention_temperature_init <= 0.0:
            raise ValueError("observed_attention_temperature_init must be positive.")
        if use_observed_attention_decoder and not use_svt_feature_decoder:
            raise ValueError(
                "use_observed_attention_decoder requires use_svt_feature_decoder."
            )
        if empirical_bayes_ridge_init <= empirical_bayes_ridge_min:
            raise ValueError(
                "empirical_bayes_ridge_init must be greater than "
                "empirical_bayes_ridge_min."
            )
        if empirical_bayes_ridge_min <= 0.0:
            raise ValueError("empirical_bayes_ridge_min must be positive.")
        if not 0.0 < empirical_bayes_shrinkage_init < 1.0:
            raise ValueError("empirical_bayes_shrinkage_init must be in (0, 1).")
        if use_empirical_bayes_decoder and not use_svt_feature_decoder:
            raise ValueError(
                "use_empirical_bayes_decoder requires use_svt_feature_decoder."
            )
        if num_prior_adapters < 0:
            raise ValueError("num_prior_adapters must be non-negative.")
        if default_prior_index is not None and default_prior_index < 0:
            default_prior_index = None
        if (
            default_prior_index is not None
            and num_prior_adapters > 0
            and default_prior_index >= num_prior_adapters
        ):
            raise ValueError(
                "default_prior_index must be smaller than num_prior_adapters."
            )
        if prior_svt_steps is not None:
            if num_prior_adapters <= 0:
                raise ValueError("prior_svt_steps requires num_prior_adapters > 0.")
            if len(prior_svt_steps) != num_prior_adapters:
                raise ValueError(
                    "prior_svt_steps length must equal num_prior_adapters."
                )
            if any(int(steps) < 0 for steps in prior_svt_steps):
                raise ValueError("prior_svt_steps values must be non-negative.")
        if svt_mixture_steps is not None:
            if len(svt_mixture_steps) < 2:
                raise ValueError("svt_mixture_steps must contain at least two depths.")
            if any(int(steps) < 0 for steps in svt_mixture_steps):
                raise ValueError("svt_mixture_steps values must be non-negative.")
            if sorted(set(int(steps) for steps in svt_mixture_steps)) != [
                int(steps) for steps in svt_mixture_steps
            ]:
                raise ValueError("svt_mixture_steps must be sorted and unique.")
            if int(svt_mixture_steps[0]) != 0:
                raise ValueError("svt_mixture_steps must include 0 as the first depth.")
        if svt_mixture_temperature <= 0.0:
            raise ValueError("svt_mixture_temperature must be positive.")
        if svt_mixture_spectral_feature_count < 0:
            raise ValueError(
                "svt_mixture_spectral_feature_count must be non-negative."
            )
        if svt_mixture_missingness_feature_count < 0:
            raise ValueError(
                "svt_mixture_missingness_feature_count must be non-negative."
            )
        if svt_mixture_spectrum_encoder_size < 0:
            raise ValueError(
                "svt_mixture_spectrum_encoder_size must be non-negative."
            )
        if svt_mixture_spectrum_encoder_layers < 1:
            raise ValueError(
                "svt_mixture_spectrum_encoder_layers must be positive."
            )
        if svt_mixture_spectrum_encoder_heads < 1:
            raise ValueError("svt_mixture_spectrum_encoder_heads must be positive.")
        if (
            svt_mixture_spectrum_encoder_size > 0
            and svt_mixture_spectrum_encoder_size
            % svt_mixture_spectrum_encoder_heads
            != 0
        ):
            raise ValueError(
                "svt_mixture_spectrum_encoder_size must be divisible by "
                "svt_mixture_spectrum_encoder_heads."
            )
        if svt_mixture_spectrum_encoder_dropout < 0.0:
            raise ValueError(
                "svt_mixture_spectrum_encoder_dropout must be non-negative."
            )
        if svt_adaptive_threshold_min_scale <= 0.0:
            raise ValueError("svt_adaptive_threshold_min_scale must be positive.")
        if svt_adaptive_threshold_max_scale <= svt_adaptive_threshold_min_scale:
            raise ValueError(
                "svt_adaptive_threshold_max_scale must be greater than "
                "svt_adaptive_threshold_min_scale."
            )
        if not (
            svt_adaptive_threshold_min_scale
            <= svt_adaptive_threshold_init_scale
            <= svt_adaptive_threshold_max_scale
        ):
            raise ValueError(
                "svt_adaptive_threshold_init_scale must lie between min and max."
            )
        if svt_mixture_policy not in {"controller", "cv", "controller_plus_cv"}:
            raise ValueError(
                "svt_mixture_policy must be 'controller', 'cv', or "
                f"'controller_plus_cv', got {svt_mixture_policy!r}."
            )
        if not 0.0 <= svt_mixture_cv_fraction < 1.0:
            raise ValueError("svt_mixture_cv_fraction must be in [0, 1).")
        if svt_mixture_cv_temperature <= 0.0:
            raise ValueError("svt_mixture_cv_temperature must be positive.")
        if svt_mixture_cv_min_count < 1:
            raise ValueError("svt_mixture_cv_min_count must be positive.")

        self.svt_init_rank = int(svt_init_rank)
        self.svt_steps = int(svt_steps)
        self.svt_solver_precision = svt_solver_precision
        self.svt_max_rank = None if svt_max_rank is None else int(svt_max_rank)
        self.svt_threshold_fraction_min = float(svt_threshold_fraction_min)
        self.svt_threshold_fraction_max = (
            None
            if svt_threshold_fraction_max is None
            else float(svt_threshold_fraction_max)
        )
        self.svt_threshold_mode = svt_threshold_mode
        self.svt_update_damping = float(svt_update_damping)
        self.logit_sigma_min = float(logit_sigma_min)
        self.use_neural_residual = bool(use_neural_residual)
        self.use_svt_feature_decoder = bool(use_svt_feature_decoder)
        self.use_observed_neighbor_features = bool(use_observed_neighbor_features)
        self.observed_neighbor_temperature = float(observed_neighbor_temperature)
        self.observed_neighbor_topk = int(observed_neighbor_topk)
        self.use_observed_attention_decoder = bool(use_observed_attention_decoder)
        self.observed_attention_topk = int(observed_attention_topk)
        self.observed_attention_dim = int(observed_attention_dim)
        self.observed_attention_hidden_size = (
            int(observed_attention_hidden_size)
            if observed_attention_hidden_size is not None
            else self.embedding_size
        )
        self.use_empirical_bayes_decoder = bool(use_empirical_bayes_decoder)
        self.empirical_bayes_ridge_min = float(empirical_bayes_ridge_min)
        self.empirical_bayes_hidden_size = (
            int(empirical_bayes_hidden_size)
            if empirical_bayes_hidden_size is not None
            else self.embedding_size
        )
        self.empirical_bayes_solve_steps = int(empirical_bayes_solve_steps)
        self.neural_residual_rank = (
            int(neural_residual_rank)
            if neural_residual_rank is not None
            else self.svt_init_rank
        )
        self.svt_feature_decoder_hidden_size = (
            int(svt_feature_decoder_hidden_size)
            if svt_feature_decoder_hidden_size is not None
            else self.embedding_size
        )
        self.num_prior_adapters = int(num_prior_adapters)
        self.default_prior_index = default_prior_index
        self.prior_svt_steps = (
            None
            if prior_svt_steps is None
            else tuple(int(steps) for steps in prior_svt_steps)
        )
        self.svt_mixture_steps = (
            None
            if svt_mixture_steps is None
            else tuple(int(steps) for steps in svt_mixture_steps)
        )
        self.svt_mixture_temperature = float(svt_mixture_temperature)
        self.svt_mixture_detach_refinement = bool(svt_mixture_detach_refinement)
        self.svt_mixture_spectral_feature_count = int(
            svt_mixture_spectral_feature_count
        )
        self.svt_mixture_missingness_feature_count = int(
            svt_mixture_missingness_feature_count
        )
        self.svt_mixture_spectrum_encoder_size = int(
            svt_mixture_spectrum_encoder_size
        )
        self.svt_mixture_spectrum_encoder_layers = int(
            svt_mixture_spectrum_encoder_layers
        )
        self.svt_mixture_spectrum_encoder_heads = int(
            svt_mixture_spectrum_encoder_heads
        )
        self.svt_mixture_spectrum_encoder_dropout = float(
            svt_mixture_spectrum_encoder_dropout
        )
        self.svt_adaptive_threshold = bool(svt_adaptive_threshold)
        self.svt_adaptive_threshold_hidden_size = (
            None
            if svt_adaptive_threshold_hidden_size is None
            else int(svt_adaptive_threshold_hidden_size)
        )
        self.svt_adaptive_threshold_min_scale = float(
            svt_adaptive_threshold_min_scale
        )
        self.svt_adaptive_threshold_max_scale = float(
            svt_adaptive_threshold_max_scale
        )
        self.svt_adaptive_threshold_init_scale = float(
            svt_adaptive_threshold_init_scale
        )
        self.svt_mixture_policy = svt_mixture_policy
        self.svt_mixture_cv_fraction = float(svt_mixture_cv_fraction)
        self.svt_mixture_cv_temperature = float(svt_mixture_cv_temperature)
        self.svt_mixture_cv_seed = int(svt_mixture_cv_seed)
        self.svt_mixture_cv_min_count = int(svt_mixture_cv_min_count)
        self.svt_mixture_cv_reencode = bool(svt_mixture_cv_reencode)
        self.svt_mixture_include_cv_features = bool(svt_mixture_include_cv_features)
        self.last_svt_mixture_weights: torch.Tensor | None = None
        self.last_svt_mixture_cv_rmse: torch.Tensor | None = None
        self.last_svt_mixture_snapshots: torch.Tensor | None = None
        self.last_svt_mixture_snapshot_points: torch.Tensor | None = None
        self.last_svt_mixture_snapshot_batch_idx: torch.Tensor | None = None
        self.last_svt_controller_features: torch.Tensor | None = None
        self.last_svt_controller_matrix_stats: torch.Tensor | None = None
        self.last_svt_controller_missingness_stats: torch.Tensor | None = None
        self.last_svt_controller_cv_features: torch.Tensor | None = None
        self.last_svt_controller_spectral_stats: torch.Tensor | None = None
        self.last_svt_controller_spectrum_encoder_stats: torch.Tensor | None = None
        self.last_svt_mixture_controller_logits: torch.Tensor | None = None
        self.last_svt_adaptive_threshold_scale: torch.Tensor | None = None
        self.last_observed_attention_row_support: torch.Tensor | None = None
        self.last_observed_attention_col_support: torch.Tensor | None = None
        self.last_observed_attention_temperature: torch.Tensor | None = None
        self.last_empirical_bayes_ridge: torch.Tensor | None = None
        self.last_empirical_bayes_shrinkage: torch.Tensor | None = None
        self.last_empirical_bayes_row_observed_fraction: torch.Tensor | None = None
        self.last_empirical_bayes_col_observed_fraction: torch.Tensor | None = None
        self.decoder_feature_size = int(
            getattr(self.output_decoder, "feature_size", self.embedding_size)
        )

        self.row_factor_head = nn.Linear(self.embedding_size, self.svt_init_rank)
        if self.use_cell_query_decoder:
            from tabimpute.model.cell_query_decoder import CellQueryDecoder
            with torch.random.fork_rng(devices=[]):
                self.cell_query_decoder = CellQueryDecoder(
                    self.embedding_size, self.decoder_feature_size, cell_query_width,
                    cell_query_heads, cell_query_layers, cell_query_chunk_size,
                    cell_query_checkpoint_chunks, cell_query_axis_balanced)
        # Optional zero-init branches must not perturb initialization/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            if self.use_svt_magnitude_channel:
                self.svt_magnitude_proj = nn.Linear(1, self.decoder_feature_size, bias=False)
                nn.init.zeros_(self.svt_magnitude_proj.weight)
            if self.use_svt_output_correction:
                self.svt_output_correction = nn.Linear(self.decoder_feature_size, 1)
                nn.init.zeros_(self.svt_output_correction.weight)
                nn.init.zeros_(self.svt_output_correction.bias)
        self.col_factor_head = nn.Linear(self.embedding_size, self.svt_init_rank)
        self.row_bias_head = nn.Linear(self.embedding_size, 1)
        self.col_bias_head = nn.Linear(self.embedding_size, 1)
        self.prediction_scale = nn.Parameter(torch.ones(()))
        self.prediction_bias = nn.Parameter(torch.zeros(()))
        self.bias_scale = nn.Parameter(torch.zeros(()))
        if self.use_neural_residual:
            self.residual_row_factor_head = nn.Linear(
                self.embedding_size,
                self.neural_residual_rank,
            )
            self.residual_col_factor_head = nn.Linear(
                self.embedding_size,
                self.neural_residual_rank,
            )
            self.neural_residual_scale = nn.Parameter(
                torch.tensor(float(neural_residual_scale_init), dtype=torch.float32)
            )
        if self.use_svt_feature_decoder:
            decoder_feature_count = 6 + (
                12 if self.use_observed_neighbor_features else 0
            )
            self.svt_feature_decoder_proj = nn.Sequential(
                nn.Linear(decoder_feature_count, self.svt_feature_decoder_hidden_size),
                nn.SiLU(),
                nn.Linear(
                    self.svt_feature_decoder_hidden_size,
                    self.decoder_feature_size,
                ),
            )
            self.svt_feature_decoder_norm = RMSNorm(self.decoder_feature_size)
            self.svt_feature_decoder_scale = nn.Parameter(
                torch.tensor(float(svt_feature_decoder_scale_init), dtype=torch.float32)
            )
        if self.use_observed_attention_decoder:
            self.observed_attention_row_query = nn.Linear(
                self.embedding_size,
                self.observed_attention_dim,
                bias=False,
            )
            self.observed_attention_row_key = nn.Linear(
                self.embedding_size,
                self.observed_attention_dim,
                bias=False,
            )
            self.observed_attention_col_query = nn.Linear(
                self.embedding_size,
                self.observed_attention_dim,
                bias=False,
            )
            self.observed_attention_col_key = nn.Linear(
                self.embedding_size,
                self.observed_attention_dim,
                bias=False,
            )
            self.observed_attention_value_bias = nn.Sequential(
                nn.Linear(4, self.observed_attention_hidden_size),
                nn.SiLU(),
                nn.Linear(self.observed_attention_hidden_size, 1),
            )
            self.observed_attention_scalar_proj = nn.Sequential(
                nn.Linear(10, self.observed_attention_hidden_size),
                nn.SiLU(),
                nn.Linear(
                    self.observed_attention_hidden_size,
                    self.decoder_feature_size,
                ),
            )
            self.observed_attention_context_proj = nn.Linear(
                2 * self.embedding_size,
                self.decoder_feature_size,
            )
            self.observed_attention_context_norm = RMSNorm(self.decoder_feature_size)
            self.observed_attention_scale = nn.Parameter(
                torch.tensor(float(observed_attention_scale_init), dtype=torch.float32)
            )
            self.observed_attention_log_temperature = nn.Parameter(
                torch.tensor(
                    math.log(float(observed_attention_temperature_init)),
                    dtype=torch.float32,
                )
            )
        if self.use_empirical_bayes_decoder:
            self.empirical_bayes_feature_proj = nn.Sequential(
                nn.Linear(10, self.empirical_bayes_hidden_size),
                nn.SiLU(),
                nn.Linear(
                    self.empirical_bayes_hidden_size,
                    self.decoder_feature_size,
                ),
            )
            self.empirical_bayes_feature_scale = nn.Parameter(
                torch.tensor(float(empirical_bayes_feature_scale_init), dtype=torch.float32)
            )
            self.empirical_bayes_ridge_unconstrained = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(
                        float(empirical_bayes_ridge_init)
                        - float(empirical_bayes_ridge_min)
                    ),
                    dtype=torch.float32,
                )
            )
            self.empirical_bayes_shrinkage_logit = nn.Parameter(
                torch.tensor(
                    _inverse_sigmoid(float(empirical_bayes_shrinkage_init)),
                    dtype=torch.float32,
                )
            )
        if self.num_prior_adapters > 0:
            self.prior_row_shift = nn.Embedding(
                self.num_prior_adapters,
                self.embedding_size,
            )
            self.prior_col_shift = nn.Embedding(
                self.num_prior_adapters,
                self.embedding_size,
            )
            self.prior_latent_scale = nn.Parameter(
                torch.tensor(float(prior_adapter_scale_init), dtype=torch.float32)
            )
            self.prior_point_scale = nn.Embedding(self.num_prior_adapters, 1)
            self.prior_point_bias = nn.Embedding(self.num_prior_adapters, 1)
            self.prior_threshold_log_scale = nn.Embedding(self.num_prior_adapters, 1)
            self.prior_logit_sigma_log_scale = nn.Embedding(self.num_prior_adapters, 1)
            self.prior_calibration_scale = nn.Parameter(
                torch.tensor(
                    float(prior_adapter_calibration_scale_init),
                    dtype=torch.float32,
                )
            )
        if self.svt_mixture_steps is not None:
            stats_size = 9
            stats_size += self.svt_mixture_missingness_feature_count
            if self.svt_mixture_include_cv_features:
                stats_size += 3 * len(self.svt_mixture_steps) + 3
            if self.svt_mixture_spectral_feature_count > 0:
                stats_size += 2 * self.svt_mixture_spectral_feature_count + 3
            if self.svt_mixture_spectrum_encoder_size > 0:
                stats_size += 2 * self.svt_mixture_spectrum_encoder_size
                self.svt_mixture_spectrum_token_proj = nn.Linear(
                    5,
                    self.svt_mixture_spectrum_encoder_size,
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.svt_mixture_spectrum_encoder_size,
                    nhead=self.svt_mixture_spectrum_encoder_heads,
                    dim_feedforward=4 * self.svt_mixture_spectrum_encoder_size,
                    dropout=self.svt_mixture_spectrum_encoder_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.svt_mixture_spectrum_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=self.svt_mixture_spectrum_encoder_layers,
                )
                self.svt_mixture_spectrum_cls = nn.Parameter(
                    torch.zeros(
                        1,
                        1,
                        self.svt_mixture_spectrum_encoder_size,
                    )
                )
                self.svt_mixture_spectrum_summary_norm = nn.LayerNorm(
                    2 * self.svt_mixture_spectrum_encoder_size
                )
            controller_input_size = 4 * self.embedding_size + stats_size
            self.svt_mixture_controller = nn.Sequential(
                nn.Linear(controller_input_size, int(svt_mixture_hidden_size)),
                nn.SiLU(),
                nn.Linear(int(svt_mixture_hidden_size), len(self.svt_mixture_steps)),
            )
            if self.svt_adaptive_threshold:
                threshold_hidden_size = int(
                    self.svt_adaptive_threshold_hidden_size
                    or svt_mixture_hidden_size
                )
                self.svt_threshold_controller = nn.Sequential(
                    nn.Linear(controller_input_size, threshold_hidden_size),
                    nn.SiLU(),
                    nn.Linear(threshold_hidden_size, 1),
                )
        self.svt_threshold_unconstrained = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid(
                    svt_threshold_fraction_init - svt_threshold_fraction_min
                ),
                dtype=torch.float32,
            )
        )
        if self.svt_threshold_mode == "inverse_size":
            self.svt_threshold_size_scale_unconstrained = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(svt_threshold_size_scale_init),
                    dtype=torch.float32,
                )
            )
        self.logit_sigma_unconstrained = nn.Parameter(
            torch.tensor(
                _inverse_softplus(logit_sigma_init - logit_sigma_min),
                dtype=torch.float32,
            )
        )

        nn.init.normal_(self.row_factor_head.weight, mean=0.0, std=svt_init_scale)
        nn.init.normal_(self.col_factor_head.weight, mean=0.0, std=svt_init_scale)
        nn.init.zeros_(self.row_factor_head.bias)
        nn.init.zeros_(self.col_factor_head.bias)
        nn.init.zeros_(self.row_bias_head.weight)
        nn.init.zeros_(self.row_bias_head.bias)
        nn.init.zeros_(self.col_bias_head.weight)
        nn.init.zeros_(self.col_bias_head.bias)
        if self.use_neural_residual:
            nn.init.normal_(
                self.residual_row_factor_head.weight,
                mean=0.0,
                std=svt_init_scale,
            )
            nn.init.normal_(
                self.residual_col_factor_head.weight,
                mean=0.0,
                std=svt_init_scale,
            )
            nn.init.zeros_(self.residual_row_factor_head.bias)
            nn.init.zeros_(self.residual_col_factor_head.bias)
        if self.use_svt_feature_decoder:
            for module in self.svt_feature_decoder_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.use_observed_attention_decoder:
            for module in (
                self.observed_attention_row_query,
                self.observed_attention_row_key,
                self.observed_attention_col_query,
                self.observed_attention_col_key,
                self.observed_attention_context_proj,
            ):
                nn.init.xavier_uniform_(module.weight)
            for module in (
                self.observed_attention_value_bias,
                self.observed_attention_scalar_proj,
            ):
                for child in module.modules():
                    if isinstance(child, nn.Linear):
                        nn.init.xavier_uniform_(child.weight)
                        nn.init.zeros_(child.bias)
        if self.use_empirical_bayes_decoder:
            for module in self.empirical_bayes_feature_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        if self.num_prior_adapters > 0:
            nn.init.zeros_(self.prior_row_shift.weight)
            nn.init.zeros_(self.prior_col_shift.weight)
            nn.init.zeros_(self.prior_point_scale.weight)
            nn.init.zeros_(self.prior_point_bias.weight)
            nn.init.zeros_(self.prior_threshold_log_scale.weight)
            nn.init.zeros_(self.prior_logit_sigma_log_scale.weight)
        if self.svt_mixture_steps is not None:
            final_layer = self.svt_mixture_controller[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, float(svt_mixture_other_logit))
            final_layer.bias.data[0] = float(svt_mixture_initial_logit)
            if self.svt_adaptive_threshold:
                threshold_final_layer = self.svt_threshold_controller[-1]
                nn.init.zeros_(threshold_final_layer.weight)
                init_fraction = (
                    (
                        self.svt_adaptive_threshold_init_scale
                        - self.svt_adaptive_threshold_min_scale
                    )
                    / (
                        self.svt_adaptive_threshold_max_scale
                        - self.svt_adaptive_threshold_min_scale
                    )
                )
                nn.init.constant_(
                    threshold_final_layer.bias,
                    _inverse_sigmoid(init_fraction),
                )

        borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
        with importlib.resources.as_file(borders_path) as path:
            borders = torch.load(path, map_location="cpu")
        if borders.numel() - 1 != self.num_outputs:
            raise ValueError(
                f"num_outputs={self.num_outputs} does not match borders "
                f"({borders.numel() - 1})."
            )
        bucket_means = borders[:-1] + 0.5 * (borders[1:] - borders[:-1])
        self.register_buffer("bucket_means", bucket_means.float(), persistent=False)
        if self.use_svt_output_correction:
            from tabimpute.model.bar_distribution import FullSupportBarDistribution
            means = FullSupportBarDistribution(borders)._full_bucket_means
            self.register_buffer("output_bucket_means", means.float(), persistent=False)

    def _prior_ids(
        self,
        src: torch.Tensor,
        prior_index: int | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.num_prior_adapters <= 0:
            return None
        if prior_index is None:
            if self.default_prior_index is None:
                return None
            prior_index = self.default_prior_index
        if isinstance(prior_index, int):
            ids = torch.full(
                (src.shape[0],),
                int(prior_index),
                device=src.device,
                dtype=torch.long,
            )
        else:
            ids = prior_index.to(device=src.device, dtype=torch.long).view(-1)
            if ids.numel() == 1 and src.shape[0] != 1:
                ids = ids.expand(src.shape[0])
        if ids.shape != (src.shape[0],):
            raise ValueError(
                f"prior_index must have shape ({src.shape[0]},), got {tuple(ids.shape)}"
            )
        if bool((ids < 0).any()) or bool((ids >= self.num_prior_adapters).any()):
            raise ValueError(
                "prior_index values must be in "
                f"[0, {self.num_prior_adapters - 1}]."
            )
        return ids

    def _apply_prior_latent_adapter(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        prior_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prior_ids is None:
            return row_state, col_state
        scale = self.prior_latent_scale.to(dtype=row_state.dtype)
        row_shift = self.prior_row_shift(prior_ids).to(dtype=row_state.dtype)
        col_shift = self.prior_col_shift(prior_ids).to(dtype=col_state.dtype)
        row_state = row_state + scale * row_shift.unsqueeze(1)
        col_state = col_state + scale * col_shift.unsqueeze(1)
        return row_state, col_state

    def _initial_grid(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_factors = self.row_factor_head(row_state).float()
        col_factors = self.col_factor_head(col_state).float()
        row_bias = self.row_bias_head(row_state).squeeze(-1).float()
        col_bias = self.col_bias_head(col_state).squeeze(-1).float()
        point = torch.einsum("brk,bck->brc", row_factors, col_factors)
        point = self.prediction_scale.float() * point + self.prediction_bias.float()
        point = point + self.bias_scale.float() * (
            row_bias.unsqueeze(2) + col_bias.unsqueeze(1)
        )
        if prior_ids is not None:
            scale = self.prior_calibration_scale.float()
            point_scale = self.prior_point_scale(prior_ids).float().view(-1, 1, 1)
            point_bias = self.prior_point_bias(prior_ids).float().view(-1, 1, 1)
            point = point * (1.0 + scale * point_scale) + scale * point_bias
        return point

    def _threshold_fraction(
        self,
        src: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
        threshold_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.svt_threshold_mode == "inverse_size":
            dim = float(min(src.shape[-2], src.shape[-1]))
            value = (
                F.softplus(self.svt_threshold_size_scale_unconstrained.float())
                / dim
            )
            value = torch.clamp(value, min=self.svt_threshold_fraction_min)
        else:
            value = (
                self.svt_threshold_fraction_min
                + torch.sigmoid(self.svt_threshold_unconstrained.float())
            )
        if self.svt_threshold_fraction_max is not None:
            value = torch.clamp(value, max=self.svt_threshold_fraction_max)
        if prior_ids is not None:
            multiplier = torch.exp(
                self.prior_threshold_log_scale(prior_ids).float().view(-1, 1)
            )
            value = value.view(1, 1) * multiplier
        if threshold_scale is not None:
            value = value * threshold_scale.float()
            value = torch.clamp(
                value,
                min=self.svt_threshold_fraction_min,
                max=(
                    self.svt_threshold_fraction_max
                    if self.svt_threshold_fraction_max is not None
                    else float("inf")
                ),
            )
        return value

    def _matrix_stats(self, src: torch.Tensor) -> torch.Tensor:
        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (~torch.isnan(src)).float()
        count = observed.sum(dim=(1, 2)).clamp_min(1.0)
        density = observed.mean(dim=(1, 2), keepdim=False)
        mean = clean.sum(dim=(1, 2)) / count
        centered = (clean - mean.view(-1, 1, 1)) * observed
        std = (centered.square().sum(dim=(1, 2)) / count).sqrt()
        row_counts = observed.sum(dim=2)
        col_counts = observed.sum(dim=1)
        row_count_mean = row_counts.mean(dim=1) / max(float(src.shape[2]), 1.0)
        row_count_std = row_counts.std(dim=1, unbiased=False) / max(float(src.shape[2]), 1.0)
        col_count_mean = col_counts.mean(dim=1) / max(float(src.shape[1]), 1.0)
        col_count_std = col_counts.std(dim=1, unbiased=False) / max(float(src.shape[1]), 1.0)
        shape = src.new_tensor(
            [src.shape[1] / 1000.0, src.shape[2] / 1000.0],
            dtype=torch.float32,
        ).view(1, 2).expand(src.shape[0], -1)
        return torch.cat(
            [
                density.view(-1, 1).float(),
                mean.view(-1, 1).float(),
                std.view(-1, 1).float(),
                row_count_mean.view(-1, 1).float(),
                row_count_std.view(-1, 1).float(),
                col_count_mean.view(-1, 1).float(),
                col_count_std.view(-1, 1).float(),
                shape,
            ],
            dim=1,
        )

    @staticmethod
    def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_centered = left - left.mean(dim=1, keepdim=True)
        right_centered = right - right.mean(dim=1, keepdim=True)
        numerator = (left_centered * right_centered).mean(dim=1)
        left_std = left_centered.square().mean(dim=1).sqrt()
        right_std = right_centered.square().mean(dim=1).sqrt()
        corr = numerator / (left_std * right_std).clamp_min(1e-6)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _missingness_stats(self, src: torch.Tensor) -> torch.Tensor:
        feature_count = self.svt_mixture_missingness_feature_count
        if feature_count <= 0:
            return src.new_empty((src.shape[0], 0), dtype=torch.float32)

        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (~torch.isnan(src)).float()
        row_rate = observed.mean(dim=2)
        col_rate = observed.mean(dim=1)
        quantiles = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            device=src.device,
            dtype=torch.float32,
        )
        row_quantiles = torch.quantile(row_rate.float(), quantiles, dim=1).transpose(0, 1)
        col_quantiles = torch.quantile(col_rate.float(), quantiles, dim=1).transpose(0, 1)
        row_cv = row_rate.std(dim=1, unbiased=False) / row_rate.mean(dim=1).clamp_min(1e-6)
        col_cv = col_rate.std(dim=1, unbiased=False) / col_rate.mean(dim=1).clamp_min(1e-6)
        row_low = (row_rate < 0.05).float().mean(dim=1)
        col_low = (col_rate < 0.05).float().mean(dim=1)

        row_count = observed.sum(dim=2).clamp_min(1.0)
        col_count = observed.sum(dim=1).clamp_min(1.0)
        row_mean = clean.sum(dim=2) / row_count
        col_mean = clean.sum(dim=1) / col_count
        row_centered = (clean - row_mean.unsqueeze(2)) * observed
        col_centered = (clean - col_mean.unsqueeze(1)) * observed
        row_std = (row_centered.square().sum(dim=2) / row_count).sqrt()
        col_std = (col_centered.square().sum(dim=1) / col_count).sqrt()

        mask_centered = observed - observed.mean(dim=(1, 2), keepdim=True)
        singular_values = torch.linalg.svdvals(mask_centered.float())
        total_energy = singular_values.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
        top = singular_values[:, :4]
        if top.shape[1] < 4:
            top = F.pad(top, (0, 4 - top.shape[1]))
        mask_spectrum = top / singular_values[:, :1].clamp_min(1e-8)
        mask_energy = top.square() / total_energy

        raw = torch.cat(
            [
                row_quantiles,
                col_quantiles,
                row_cv.view(-1, 1),
                col_cv.view(-1, 1),
                row_low.view(-1, 1),
                col_low.view(-1, 1),
                self._safe_corr(row_rate, row_mean).view(-1, 1),
                self._safe_corr(col_rate, col_mean).view(-1, 1),
                self._safe_corr(row_rate, row_std).view(-1, 1),
                self._safe_corr(col_rate, col_std).view(-1, 1),
                mask_spectrum,
                mask_energy,
            ],
            dim=1,
        ).float()
        if raw.shape[1] < feature_count:
            raw = F.pad(raw, (0, feature_count - raw.shape[1]))
        return raw[:, :feature_count]

    def _controller_singular_values(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
    ) -> torch.Tensor:
        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (~torch.isnan(src)).float()
        missing = 1.0 - observed
        x = observed * clean + missing * initial_grid.detach().float()
        x = x - x.mean(dim=(1, 2), keepdim=True)
        return torch.linalg.svdvals(x.float())

    def _spectral_stats_from_singular_values(
        self,
        src: torch.Tensor,
        singular_values: torch.Tensor,
    ) -> torch.Tensor:
        feature_count = self.svt_mixture_spectral_feature_count
        if feature_count <= 0:
            return src.new_empty((src.shape[0], 0), dtype=torch.float32)

        k = min(feature_count, singular_values.shape[-1])
        top = singular_values[:, :k]
        if k < feature_count:
            top = F.pad(top, (0, feature_count - k))

        all_energy = singular_values.square()
        total_energy = all_energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
        top_energy = top.square()
        top_energy_fraction = top_energy.cumsum(dim=1) / total_energy
        normalized_top = top / top[:, :1].clamp_min(1e-6)
        probabilities = all_energy / total_energy
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
            dim=1,
            keepdim=True,
        )
        effective_rank = entropy.exp() / max(float(singular_values.shape[-1]), 1.0)
        first_energy_fraction = all_energy[:, :1] / total_energy
        retained_energy_fraction = top_energy.sum(dim=1, keepdim=True) / total_energy
        return torch.cat(
            [
                normalized_top,
                top_energy_fraction,
                effective_rank,
                first_energy_fraction,
                retained_energy_fraction,
            ],
            dim=1,
        ).float()

    def _spectral_stats(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
    ) -> torch.Tensor:
        singular_values = self._controller_singular_values(src, initial_grid)
        return self._spectral_stats_from_singular_values(src, singular_values)

    def _spectrum_encoder_stats_from_singular_values(
        self,
        src: torch.Tensor,
        singular_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = self.svt_mixture_spectrum_encoder_size
        if hidden_size <= 0:
            return src.new_empty((src.shape[0], 0), dtype=torch.float32)

        feature_count = max(self.svt_mixture_spectral_feature_count, 16)
        k = min(feature_count, singular_values.shape[-1])
        top = singular_values[:, :k]
        if k < feature_count:
            top = F.pad(top, (0, feature_count - k))

        total_energy = singular_values.square().sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        normalized = top / top[:, :1].clamp_min(1e-6)
        log_normalized = normalized.clamp_min(1e-8).log()
        previous = F.pad(normalized[:, :-1], (1, 0), value=1.0)
        log_drop = (normalized.clamp_min(1e-8) / previous.clamp_min(1e-8)).log()
        energy_fraction = top.square() / total_energy
        cumulative_energy = energy_fraction.cumsum(dim=1).clamp(max=1.0)
        index = torch.linspace(
            0.0,
            1.0,
            feature_count,
            device=src.device,
            dtype=torch.float32,
        ).view(1, feature_count).expand(src.shape[0], -1)
        token_features = torch.stack(
            [
                log_normalized,
                log_drop,
                energy_fraction,
                cumulative_energy,
                index,
            ],
            dim=-1,
        )
        projection = self.svt_mixture_spectrum_token_proj
        token_features = token_features.to(dtype=projection.weight.dtype)
        tokens = projection(token_features)
        cls = self.svt_mixture_spectrum_cls.expand(src.shape[0], -1, -1).to(
            dtype=tokens.dtype
        )
        encoded = self.svt_mixture_spectrum_encoder(torch.cat([cls, tokens], dim=1))
        summary = torch.cat(
            [encoded[:, 0], encoded[:, 1:].mean(dim=1)],
            dim=1,
        )
        return self.svt_mixture_spectrum_summary_norm(summary).float()

    def _spectrum_encoder_stats(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
    ) -> torch.Tensor:
        singular_values = self._controller_singular_values(src, initial_grid)
        return self._spectrum_encoder_stats_from_singular_values(src, singular_values)

    def _svt_iterates(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None,
        steps: tuple[int, ...],
        observed_mask: torch.Tensor | None = None,
        threshold_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (
            (~torch.isnan(src)).float()
            if observed_mask is None
            else observed_mask.float()
        )
        missing = 1.0 - observed
        x0 = observed * clean + missing * initial_grid.float()
        snapshots = {0: x0}
        x = x0
        max_steps = max(steps)
        # Casting the SVD input alone does not protect reconstruction from AMP.
        with (
            torch.autocast(device_type=src.device.type, enabled=False)
            if self.svt_solver_precision == "float32"
            else nullcontext()
        ):
            for step_index in range(1, max_steps + 1):
                u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
                rank = s.shape[-1] if self.svt_max_rank is None else min(self.svt_max_rank, s.shape[-1])
                u_k = u[..., :, :rank]
                s_k = s[..., :rank]
                vh_k = vh[..., :rank, :]
                tau = self._threshold_fraction(
                    src,
                    prior_ids,
                    threshold_scale,
                ) * s[..., :1].detach()
                s_shrunk = torch.relu(s_k - tau)
                low_rank = (u_k * s_shrunk.unsqueeze(-2)) @ vh_k
                updated_missing = (
                    (1.0 - self.svt_update_damping) * x
                    + self.svt_update_damping * low_rank
                )
                x = observed * clean + missing * updated_missing
                if step_index in steps:
                    snapshots[step_index] = x
        return torch.stack([snapshots[step] for step in steps], dim=1)

    def _svt_mixture_controller_logits(
        self,
        src: torch.Tensor,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            features = self._svt_controller_features(
                src,
                row_state,
                col_state,
                initial_grid,
                prior_ids,
            )
        first_layer = self.svt_mixture_controller[0]
        features = features.to(dtype=first_layer.weight.dtype)
        logits = self.svt_mixture_controller(features).float()
        self.last_svt_mixture_controller_logits = logits.detach()
        return logits

    def _svt_controller_features(
        self,
        src: torch.Tensor,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_mean = row_state.float().mean(dim=1)
        col_mean = col_state.float().mean(dim=1)
        row_std = row_state.float().std(dim=1, unbiased=False)
        col_std = col_state.float().std(dim=1, unbiased=False)
        stats = self._matrix_stats(src)
        missingness_stats = self._missingness_stats(src)
        cv_features = self._svt_mixture_cv_features(src, initial_grid, prior_ids)
        if (
            self.svt_mixture_spectral_feature_count > 0
            or self.svt_mixture_spectrum_encoder_size > 0
        ):
            singular_values = self._controller_singular_values(src, initial_grid)
            spectral_stats = self._spectral_stats_from_singular_values(
                src,
                singular_values,
            )
            spectrum_encoder_stats = self._spectrum_encoder_stats_from_singular_values(
                src,
                singular_values,
            )
        else:
            spectral_stats = src.new_empty((src.shape[0], 0), dtype=torch.float32)
            spectrum_encoder_stats = src.new_empty(
                (src.shape[0], 0),
                dtype=torch.float32,
            )
        features = torch.cat(
            [
                row_mean,
                col_mean,
                row_std,
                col_std,
                stats,
                missingness_stats,
                cv_features,
                spectral_stats,
                spectrum_encoder_stats,
            ],
            dim=1,
        )
        self.last_svt_controller_matrix_stats = stats.detach()
        self.last_svt_controller_missingness_stats = missingness_stats.detach()
        self.last_svt_controller_cv_features = cv_features.detach()
        self.last_svt_controller_spectral_stats = spectral_stats.detach()
        self.last_svt_controller_spectrum_encoder_stats = (
            spectrum_encoder_stats.detach()
        )
        self.last_svt_controller_features = features.detach()
        return features

    def _svt_adaptive_threshold_scale(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.svt_adaptive_threshold:
            self.last_svt_adaptive_threshold_scale = None
            return None
        first_layer = self.svt_threshold_controller[0]
        features = features.to(dtype=first_layer.weight.dtype)
        raw = self.svt_threshold_controller(features).float()
        scale_range = (
            self.svt_adaptive_threshold_max_scale
            - self.svt_adaptive_threshold_min_scale
        )
        scale = self.svt_adaptive_threshold_min_scale + scale_range * torch.sigmoid(
            raw
        )
        self.last_svt_adaptive_threshold_scale = scale
        return scale

    def _svt_mixture_cv_mask(self, src: torch.Tensor) -> torch.Tensor:
        observed = ~torch.isnan(src)
        if self.svt_mixture_cv_fraction <= 0.0:
            return torch.zeros_like(observed)
        rows = torch.arange(src.shape[1], device=src.device, dtype=torch.long)
        cols = torch.arange(src.shape[2], device=src.device, dtype=torch.long)
        row_hash = rows.view(1, -1, 1) * 1009
        col_hash = cols.view(1, 1, -1) * 9176
        hashed = (row_hash + col_hash + self.svt_mixture_cv_seed * 6113) % 10000
        threshold = int(round(self.svt_mixture_cv_fraction * 10000))
        return observed & (hashed < threshold)

    def _svt_mixture_cv_rmse(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validation = self._svt_mixture_cv_mask(src)
        observed = ~torch.isnan(src)
        fit_observed = observed & ~validation
        with torch.no_grad():
            fit_src = src
            fit_initial = initial_grid.detach()
            fit_mean = 0.0
            fit_std = 1.0
            if self.svt_mixture_cv_reencode:
                fit_src = src.masked_fill(~fit_observed, float("nan")).float()
                count = fit_observed.sum(dim=1, keepdim=True).clamp_min(1)
                fit_mean = fit_src.nan_to_num().sum(dim=1, keepdim=True) / count
                centered = torch.where(fit_observed, fit_src - fit_mean, 0.0)
                fit_std = (centered.square().sum(dim=1, keepdim=True) / count).sqrt()
                fit_std = torch.where(fit_std > 1e-6, fit_std, 1.0)
                fit_src = ((fit_src - fit_mean) / fit_std).to(src.dtype)
                # The held-out values must not enter encoding or normalization.
                modes = [(module, module.training) for module in self.modules()]
                captures = [(module, module.capture_attention_weights,
                             module.last_attention_weights, module.last_attention_stats)
                            for module in self.modules() if hasattr(module, "capture_attention_weights")]
                self.eval()
                self._in_cv_encoding = True
                try:
                    for module, _, _, _ in captures:
                        module.capture_attention_weights = False
                    fit_rows, fit_cols = self.encode_latents(fit_src)
                    fit_rows, fit_cols = self._apply_prior_latent_adapter(
                        fit_rows, fit_cols, prior_ids
                    )
                    fit_initial = self._initial_grid(fit_rows, fit_cols, prior_ids)
                finally:
                    self._in_cv_encoding = False
                    for module, training in modes:
                        module.training = training
                    for module, enabled, weights, stats in captures:
                        module.capture_attention_weights = enabled
                        module.last_attention_weights = weights
                        module.last_attention_stats = stats
            snapshots = self._svt_iterates(
                fit_src,
                fit_initial,
                prior_ids,
                self.svt_mixture_steps,
                observed_mask=fit_observed,
            )
            if self.svt_mixture_cv_reencode:
                snapshots = snapshots * fit_std.unsqueeze(1) + fit_mean.unsqueeze(1)
            clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
            validation_float = validation.float()
            counts = validation_float.sum(dim=(1, 2)).clamp_min(1.0)
            sq_error = (snapshots - clean.unsqueeze(1)).square()
            mse = (sq_error * validation_float.unsqueeze(1)).sum(dim=(2, 3))
            mse = mse / counts.unsqueeze(1)
            rmse = mse.clamp_min(0.0).sqrt()
            enough_values = counts >= float(self.svt_mixture_cv_min_count)
            rmse = torch.where(enough_values.unsqueeze(1), rmse, torch.zeros_like(rmse))
        self.last_svt_mixture_cv_rmse = rmse.detach()
        return rmse, counts.detach()

    def _svt_mixture_cv_features(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not self.svt_mixture_include_cv_features
            or self.svt_mixture_steps is None
            or self.svt_mixture_cv_fraction <= 0.0
        ):
            return src.new_empty((src.shape[0], 0), dtype=torch.float32)
        rmse, counts = self._svt_mixture_cv_rmse(src, initial_grid, prior_ids)
        rmse = rmse.float()
        centered = rmse - rmse.mean(dim=1, keepdim=True)
        cv_probs = torch.softmax(
            -rmse / float(self.svt_mixture_cv_temperature),
            dim=-1,
        )
        top_values = torch.topk(-rmse, k=min(2, rmse.shape[1]), dim=1).values
        if top_values.shape[1] > 1:
            margin = top_values[:, :1] - top_values[:, 1:2]
        else:
            margin = top_values.new_zeros((top_values.shape[0], 1))
        validation_density = counts.view(-1, 1) / max(
            float(src.shape[1] * src.shape[2]),
            1.0,
        )
        features = torch.cat(
            [
                rmse,
                centered,
                cv_probs,
                validation_density,
                rmse.min(dim=1, keepdim=True).values,
                margin,
            ],
            dim=1,
        ).float()
        self.last_svt_controller_cv_features = features.detach()
        return features

    def _svt_mixture_cv_logits(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.svt_mixture_include_cv_features and self.last_svt_mixture_cv_rmse is not None:
            rmse = self.last_svt_mixture_cv_rmse
        else:
            rmse, _ = self._svt_mixture_cv_rmse(src, initial_grid, prior_ids)
        return -rmse / self.svt_mixture_cv_temperature

    def _svt_mixture_weights(
        self,
        src: torch.Tensor,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        controller_logits = self._svt_mixture_controller_logits(
            src,
            row_state,
            col_state,
            initial_grid,
            prior_ids,
            features,
        )
        logits = controller_logits / self.svt_mixture_temperature
        if (
            self.svt_mixture_policy in {"cv", "controller_plus_cv"}
            and self.svt_mixture_cv_fraction > 0.0
        ):
            cv_logits = self._svt_mixture_cv_logits(src, initial_grid, prior_ids)
            if self.svt_mixture_policy == "cv":
                logits = cv_logits
            else:
                logits = logits + cv_logits
        elif not self.svt_mixture_include_cv_features:
            self.last_svt_mixture_cv_rmse = None
        self.last_svt_mixture_log_probs = torch.log_softmax(logits.float(), dim=-1)
        return self.last_svt_mixture_log_probs.exp().to(dtype=row_state.dtype)

    def _svt_refine(
        self,
        src: torch.Tensor,
        initial_grid: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
        threshold_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (~torch.isnan(src)).float()
        missing = 1.0 - observed
        x = observed * clean + missing * initial_grid.float()
        if self.prior_svt_steps is not None and prior_ids is not None:
            step_counts = torch.tensor(
                self.prior_svt_steps,
                device=src.device,
                dtype=torch.long,
            )[prior_ids]
            max_steps = int(step_counts.max().item())
            output = x.clone()
        else:
            step_counts = None
            max_steps = self.svt_steps
            output = None
        with (
            torch.autocast(device_type=src.device.type, enabled=False)
            if self.svt_solver_precision == "float32"
            else nullcontext()
        ):
            for step_index in range(1, max_steps + 1):
                u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
                rank = s.shape[-1] if self.svt_max_rank is None else min(self.svt_max_rank, s.shape[-1])
                u_k = u[..., :, :rank]
                s_k = s[..., :rank]
                vh_k = vh[..., :rank, :]
                tau = self._threshold_fraction(
                    src,
                    prior_ids,
                    threshold_scale,
                ) * s[..., :1].detach()
                s_shrunk = torch.relu(s_k - tau)
                low_rank = (u_k * s_shrunk.unsqueeze(-2)) @ vh_k
                updated_missing = (
                    (1.0 - self.svt_update_damping) * x
                    + self.svt_update_damping * low_rank
                )
                x = observed * clean + missing * updated_missing
                if step_counts is not None:
                    done = step_counts == step_index
                    if bool(done.any()):
                        output = torch.where(done.view(-1, 1, 1), x, output)
        if step_counts is not None:
            return output
        return x

    def _point_grid(
        self,
        src: torch.Tensor,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.svt_mixture_steps is not None:
            initial_grid = self._initial_grid(row_state, col_state, prior_ids)
            controller_features = self._svt_controller_features(
                src,
                row_state,
                col_state,
                initial_grid,
                prior_ids,
            )
            threshold_scale = self._svt_adaptive_threshold_scale(controller_features)
            if self.svt_mixture_detach_refinement:
                with torch.no_grad():
                    detached_snapshots = self._svt_iterates(
                        src,
                        initial_grid.detach(),
                        prior_ids,
                        self.svt_mixture_steps,
                        threshold_scale=(
                            None
                            if threshold_scale is None
                            else threshold_scale.detach()
                        ),
                    )
                snapshots = torch.cat(
                    [
                        initial_grid.float().unsqueeze(1),
                        detached_snapshots[:, 1:],
                    ],
                    dim=1,
                )
            else:
                snapshots = self._svt_iterates(
                    src,
                    initial_grid,
                    prior_ids,
                    self.svt_mixture_steps,
                    threshold_scale=threshold_scale,
                )
            self.last_svt_mixture_snapshots = snapshots.detach()
            weights = self._svt_mixture_weights(
                src,
                row_state,
                col_state,
                initial_grid,
                prior_ids,
                features=controller_features,
            )
            self.last_svt_mixture_weights = weights
            return (snapshots * weights.float().view(weights.shape[0], -1, 1, 1)).sum(dim=1)
        self.last_svt_mixture_snapshots = None
        self.last_svt_adaptive_threshold_scale = None
        self.last_svt_controller_features = None
        self.last_svt_controller_matrix_stats = None
        self.last_svt_controller_missingness_stats = None
        self.last_svt_controller_cv_features = None
        self.last_svt_controller_spectral_stats = None
        self.last_svt_controller_spectrum_encoder_stats = None
        self.last_svt_mixture_controller_logits = None
        point = self._svt_refine(
            src,
            self._initial_grid(row_state, col_state, prior_ids),
            prior_ids,
        )
        if self.use_neural_residual:
            row_factors = self.residual_row_factor_head(row_state).float()
            col_factors = self.residual_col_factor_head(col_state).float()
            residual = torch.einsum("brk,bck->brc", row_factors, col_factors)
            point = point + self.neural_residual_scale.float() * residual
        return point

    def _point_pairs(
        self,
        point_grid: torch.Tensor,
        decode_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            return point_grid.new_empty((0,)), batch_idx
        return point_grid[batch_idx, row_idx, col_idx], batch_idx

    def _logits_from_point(
        self,
        point: torch.Tensor,
        prior_ids: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        centers = self.bucket_means.to(device=point.device, dtype=torch.float32)
        sigma = F.softplus(self.logit_sigma_unconstrained).float()
        sigma = sigma + self.logit_sigma_min
        if prior_ids is not None:
            sigma_multiplier = torch.exp(
                self.prior_logit_sigma_log_scale(prior_ids).float().view(-1)
            )
            if batch_idx is not None:
                sigma = sigma * sigma_multiplier[batch_idx].view(*point.shape, 1)
            else:
                view_shape = (prior_ids.shape[0],) + (1,) * point.ndim
                sigma = sigma * sigma_multiplier.view(view_shape)
        logits = -0.5 * (
            (centers.view(*((1,) * point.ndim), -1) - point.unsqueeze(-1)) / sigma
        ).square()
        logits = logits - logits.amax(dim=-1, keepdim=True)
        return logits.to(dtype=point.dtype)

    def _svt_point_features(self, point: torch.Tensor) -> torch.Tensor:
        point = point.float()
        clipped = point.clamp(-6.0, 6.0)
        return torch.stack(
            [
                clipped,
                clipped.abs(),
                clipped.square(),
                torch.tanh(clipped),
                torch.sin(clipped),
                torch.cos(clipped),
            ],
            dim=-1,
        )

    def _masked_topk_average(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        logits: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.float()
        values = values.float()
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        weights = torch.exp(
            masked_logits - masked_logits.amax(dim=dim, keepdim=True)
        )
        weights = weights * mask.float()
        denom = weights.sum(dim=dim, keepdim=True)
        pred = (weights * values).sum(dim=dim) / denom.squeeze(dim).clamp_min(1.0e-6)
        has_value = denom.squeeze(dim) > 0.0
        return pred, has_value

    def _observed_neighbor_point_features(self, src: torch.Tensor) -> torch.Tensor:
        observed = ~torch.isnan(src)
        clean = torch.nan_to_num(src.float(), nan=0.0)
        obs = observed.float()
        eps = 1.0e-6
        batch_size, rows, cols = clean.shape
        temp = self.observed_neighbor_temperature

        global_count = obs.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        global_mean = (clean * obs).sum(dim=(1, 2), keepdim=True) / global_count
        row_count = obs.sum(dim=2, keepdim=True).clamp_min(1.0)
        col_count = obs.sum(dim=1, keepdim=True).clamp_min(1.0)
        row_mean = (clean * obs).sum(dim=2, keepdim=True) / row_count
        col_mean = (clean * obs).sum(dim=1, keepdim=True) / col_count
        row_mean_grid = row_mean.expand(batch_size, rows, cols)
        col_mean_grid = col_mean.expand(batch_size, rows, cols)
        global_mean_grid = global_mean.expand(batch_size, rows, cols)

        row_centered = torch.where(observed, clean - row_mean, torch.zeros_like(clean))
        row_norm = row_centered.square().sum(dim=2).sqrt().clamp_min(eps)
        row_sim = (row_centered @ row_centered.transpose(1, 2)) / (
            row_norm[:, :, None] * row_norm[:, None, :]
        )
        row_sim = row_sim.masked_fill(
            torch.eye(rows, device=src.device, dtype=torch.bool)[None], -1.0e9
        )
        row_k = min(self.observed_neighbor_topk, rows)
        row_top, row_idx = row_sim.topk(row_k, dim=2)
        row_gather_idx = row_idx.unsqueeze(-1).expand(batch_size, rows, row_k, cols)
        row_values = clean[:, None].expand(batch_size, rows, rows, cols).gather(
            2, row_gather_idx
        )
        row_mask = observed[:, None].expand(batch_size, rows, rows, cols).gather(
            2, row_gather_idx
        )
        row_logits = (row_top / temp).unsqueeze(-1).expand_as(row_values)
        row_neighbor, row_has = self._masked_topk_average(
            row_values,
            row_mask,
            row_logits,
            dim=2,
        )
        row_neighbor = torch.where(row_has, row_neighbor, col_mean_grid)

        col_centered = torch.where(observed, clean - col_mean, torch.zeros_like(clean))
        col_norm = col_centered.square().sum(dim=1).sqrt().clamp_min(eps)
        col_sim = (col_centered.transpose(1, 2) @ col_centered) / (
            col_norm[:, :, None] * col_norm[:, None, :]
        )
        col_sim = col_sim.masked_fill(
            torch.eye(cols, device=src.device, dtype=torch.bool)[None], -1.0e9
        )
        col_k = min(self.observed_neighbor_topk, cols)
        col_top, col_idx = col_sim.topk(col_k, dim=2)
        col_gather_idx = col_idx[:, None].expand(batch_size, rows, cols, col_k)
        col_values = clean[:, :, None].expand(batch_size, rows, cols, cols).gather(
            3, col_gather_idx
        )
        col_mask = observed[:, :, None].expand(batch_size, rows, cols, cols).gather(
            3, col_gather_idx
        )
        col_logits = (col_top / temp)[:, None].expand_as(col_values)
        col_neighbor, col_has = self._masked_topk_average(
            col_values,
            col_mask,
            col_logits,
            dim=3,
        )
        col_neighbor = torch.where(col_has, col_neighbor, row_mean_grid)

        neighbor_mean = 0.5 * (row_neighbor + col_neighbor)
        row_support = row_has.float()
        col_support = col_has.float()
        features = torch.stack(
            [
                row_neighbor,
                col_neighbor,
                neighbor_mean,
                row_mean_grid,
                col_mean_grid,
                global_mean_grid,
                row_neighbor - col_neighbor,
                neighbor_mean - global_mean_grid,
                row_neighbor.abs(),
                col_neighbor.abs(),
                row_support,
                col_support,
            ],
            dim=-1,
        )
        return features.clamp(-10.0, 10.0)

    @staticmethod
    def _observed_attention_value_features(values: torch.Tensor) -> torch.Tensor:
        values = values.float().clamp(-10.0, 10.0)
        return torch.stack(
            [
                values,
                values.abs(),
                values.square().clamp(max=100.0),
                torch.tanh(values),
            ],
            dim=-1,
        )

    @staticmethod
    def _masked_attention_weights(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_logits = logits.float().masked_fill(~mask, -1.0e9)
        weights = torch.exp(masked_logits - masked_logits.amax(dim=dim, keepdim=True))
        weights = weights * mask.float()
        denom = weights.sum(dim=dim, keepdim=True)
        weights = weights / denom.clamp_min(1.0e-6)
        return weights, denom.squeeze(dim) > 0.0

    def _observed_attention_bias(self, features: torch.Tensor) -> torch.Tensor:
        first_layer = self.observed_attention_value_bias[0]
        features = features.to(dtype=first_layer.weight.dtype)
        # This pointwise MLP otherwise materializes a huge all-cell hidden tensor.
        flat = features.reshape(-1, features.shape[-1])
        if flat.shape[0] <= 65536:
            return self.observed_attention_value_bias(features).float().squeeze(-1)
        from torch.utils.checkpoint import checkpoint
        pieces = []
        for part in flat.split(65536):
            if self.training and torch.is_grad_enabled():
                result = checkpoint(self.observed_attention_value_bias, part, use_reentrant=False)
            else:
                result = self.observed_attention_value_bias(part)
            pieces.append(result.float().squeeze(-1))
        return torch.cat(pieces).reshape(features.shape[:-1])

    def _observed_attention_context(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        src: torch.Tensor,
        decode_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        observed = ~torch.isnan(src)
        clean = torch.nan_to_num(src.float(), nan=0.0)
        batch_size, rows, cols = clean.shape
        temperature = self.observed_attention_log_temperature.float().exp().clamp_min(
            1.0e-4
        )
        self.last_observed_attention_temperature = temperature.detach()

        row_q = F.normalize(
            self.observed_attention_row_query(row_state).float(),
            dim=-1,
        )
        row_k = F.normalize(
            self.observed_attention_row_key(row_state).float(),
            dim=-1,
        )
        row_sim = torch.bmm(row_q, row_k.transpose(1, 2)) / temperature
        row_sim = row_sim.masked_fill(
            torch.eye(rows, device=src.device, dtype=torch.bool)[None],
            -1.0e9,
        )
        row_topk = min(self.observed_attention_topk, rows)
        row_top, row_idx = row_sim.topk(row_topk, dim=2)
        row_valid = row_top > -1.0e8
        row_gather_idx = row_idx.unsqueeze(-1).expand(batch_size, rows, row_topk, cols)
        row_values = clean[:, None].expand(batch_size, rows, rows, cols).gather(
            2,
            row_gather_idx,
        )
        row_mask = observed[:, None].expand(batch_size, rows, rows, cols).gather(
            2,
            row_gather_idx,
        )
        row_mask = row_mask & row_valid.unsqueeze(-1)
        row_value_features = self._observed_attention_value_features(row_values)
        row_logits = row_top.unsqueeze(-1) + self._observed_attention_bias(
            row_value_features
        )
        row_weights, row_has = self._masked_attention_weights(
            row_logits,
            row_mask,
            dim=2,
        )
        row_state_sel = row_state[:, None].expand(
            batch_size,
            rows,
            rows,
            self.embedding_size,
        ).gather(
            2,
            row_idx.unsqueeze(-1).expand(
                batch_size,
                rows,
                row_topk,
                self.embedding_size,
            ),
        )
        row_context = torch.einsum(
            "brkc,brkd->brcd",
            row_weights,
            row_state_sel.float(),
        )
        row_scalar_inputs = torch.cat(
            [row_value_features, row_mask.float().unsqueeze(-1)],
            dim=-1,
        )
        row_scalar = torch.einsum(
            "brkc,brkcf->brcf",
            row_weights,
            row_scalar_inputs,
        )

        col_q = F.normalize(
            self.observed_attention_col_query(col_state).float(),
            dim=-1,
        )
        col_k = F.normalize(
            self.observed_attention_col_key(col_state).float(),
            dim=-1,
        )
        col_sim = torch.bmm(col_q, col_k.transpose(1, 2)) / temperature
        col_sim = col_sim.masked_fill(
            torch.eye(cols, device=src.device, dtype=torch.bool)[None],
            -1.0e9,
        )
        col_topk = min(self.observed_attention_topk, cols)
        col_top, col_idx = col_sim.topk(col_topk, dim=2)
        col_valid = col_top > -1.0e8
        col_gather_idx = col_idx[:, None].expand(batch_size, rows, cols, col_topk)
        col_values = clean[:, :, None].expand(batch_size, rows, cols, cols).gather(
            3,
            col_gather_idx,
        )
        col_mask = observed[:, :, None].expand(batch_size, rows, cols, cols).gather(
            3,
            col_gather_idx,
        )
        col_mask = col_mask & col_valid[:, None]
        col_value_features = self._observed_attention_value_features(col_values)
        col_logits = col_top[:, None] + self._observed_attention_bias(
            col_value_features
        )
        col_weights, col_has = self._masked_attention_weights(
            col_logits,
            col_mask,
            dim=3,
        )
        col_state_sel = col_state[:, None].expand(
            batch_size,
            cols,
            cols,
            self.embedding_size,
        ).gather(
            2,
            col_idx.unsqueeze(-1).expand(
                batch_size,
                cols,
                col_topk,
                self.embedding_size,
            ),
        )
        col_context = torch.einsum(
            "brck,bckd->brcd",
            col_weights,
            col_state_sel.float(),
        )
        col_scalar_inputs = torch.cat(
            [col_value_features, col_mask.float().unsqueeze(-1)],
            dim=-1,
        )
        col_scalar = torch.einsum(
            "brck,brckf->brcf",
            col_weights,
            col_scalar_inputs,
        )

        context_weight = self.observed_attention_context_proj.weight.dtype
        latent_context = self.observed_attention_context_proj(
            torch.cat([row_context, col_context], dim=-1).to(dtype=context_weight)
        )
        scalar_first_layer = self.observed_attention_scalar_proj[0]
        scalar_context = self.observed_attention_scalar_proj(
            torch.cat([row_scalar, col_scalar], dim=-1).to(
                dtype=scalar_first_layer.weight.dtype
            )
        )
        context = self.observed_attention_context_norm(
            (latent_context + scalar_context).to(dtype=row_state.dtype)
        )

        self.last_observed_attention_row_support = (
            row_has.float().mean(dim=(1, 2)).detach()
        )
        self.last_observed_attention_col_support = (
            col_has.float().mean(dim=(1, 2)).detach()
        )
        if decode_mask is not None:
            batch_idx, row_idx_mask, col_idx_mask = decode_mask.nonzero(as_tuple=True)
            return context[batch_idx, row_idx_mask, col_idx_mask]
        return context

    def _empirical_bayes_grid(self, src: torch.Tensor) -> dict[str, torch.Tensor]:
        observed = ~torch.isnan(src)
        clean = torch.nan_to_num(src.float(), nan=0.0)
        obs = observed.float()
        batch_size, rows, cols = clean.shape

        col_count = obs.sum(dim=1).clamp_min(1.0)
        col_mean = (clean * obs).sum(dim=1) / col_count
        centered = torch.where(
            observed,
            clean - col_mean.unsqueeze(1),
            torch.zeros_like(clean),
        )
        col_var = centered.square().sum(dim=1) / (col_count - 1.0).clamp_min(1.0)
        col_std = col_var.sqrt().clamp_min(1.0e-4)
        standardized = centered / col_std.unsqueeze(1)

        gram = torch.bmm(standardized.transpose(1, 2), standardized)
        corr = gram / float(max(rows, 1))
        corr_diag = corr.diagonal(dim1=1, dim2=2).clamp_min(1.0e-4)
        corr = corr / (
            corr_diag.sqrt().unsqueeze(2) * corr_diag.sqrt().unsqueeze(1)
        ).clamp_min(1.0e-4)
        corr = 0.5 * (corr + corr.transpose(1, 2))
        eye = torch.eye(cols, device=src.device, dtype=torch.float32).unsqueeze(0)
        shrinkage = torch.sigmoid(self.empirical_bayes_shrinkage_logit.float())
        covariance = (1.0 - shrinkage) * corr + shrinkage * eye
        covariance = covariance.masked_fill(eye.bool(), 1.0)

        ridge = self.empirical_bayes_ridge_min + F.softplus(
            self.empirical_bayes_ridge_unconstrained.float()
        )
        self.last_empirical_bayes_ridge = ridge.detach()
        self.last_empirical_bayes_shrinkage = shrinkage.detach()

        residual = (clean - col_mean.unsqueeze(1)) * obs
        solution = torch.zeros_like(residual)
        conjugate_residual = residual
        direction = conjugate_residual

        def system_matvec(vector: torch.Tensor) -> torch.Tensor:
            observed_vector = vector * obs
            covariance_part = torch.einsum(
                "bcj,brj->brc",
                covariance,
                observed_vector,
            )
            return obs * covariance_part + ridge * observed_vector + (1.0 - obs) * vector

        residual_square = conjugate_residual.square().sum(dim=2, keepdim=True)
        for _ in range(max(self.empirical_bayes_solve_steps, 0)):
            system_direction = system_matvec(direction)
            denom = (direction * system_direction).sum(dim=2, keepdim=True).clamp_min(
                1.0e-8
            )
            step_size = residual_square / denom
            solution = solution + step_size * direction
            conjugate_residual = conjugate_residual - step_size * system_direction
            next_residual_square = conjugate_residual.square().sum(
                dim=2,
                keepdim=True,
            )
            direction = conjugate_residual + (
                next_residual_square / residual_square.clamp_min(1.0e-8)
            ) * direction
            residual_square = next_residual_square
        correction = torch.einsum("bcj,brj->brc", covariance, solution)
        pred = col_mean.unsqueeze(1) + correction

        row_count = obs.sum(dim=2).clamp_min(1.0)
        row_mean = (clean * obs).sum(dim=2) / row_count
        row_fraction = obs.mean(dim=2)
        col_fraction = obs.mean(dim=1)
        self.last_empirical_bayes_row_observed_fraction = (
            row_fraction.mean(dim=1).detach()
        )
        self.last_empirical_bayes_col_observed_fraction = (
            col_fraction.mean(dim=1).detach()
        )
        return {
            "pred": pred.clamp(-20.0, 20.0),
            "col_mean": col_mean,
            "row_mean": row_mean,
            "row_fraction": row_fraction,
            "col_fraction": col_fraction,
        }

    def _empirical_bayes_point_features(
        self,
        src: torch.Tensor,
        point_grid: torch.Tensor,
        decode_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        empirical = self._empirical_bayes_grid(src)
        pred = empirical["pred"]
        col_mean = empirical["col_mean"].unsqueeze(1).expand_as(pred)
        row_mean = empirical["row_mean"].unsqueeze(2).expand_as(pred)
        row_fraction = empirical["row_fraction"].unsqueeze(2).expand_as(pred)
        col_fraction = empirical["col_fraction"].unsqueeze(1).expand_as(pred)
        point = point_grid.float()
        features = torch.stack(
            [
                pred,
                pred - point,
                pred - col_mean,
                pred - row_mean,
                pred.abs(),
                (pred - point).abs(),
                point - col_mean,
                point - row_mean,
                row_fraction,
                col_fraction,
            ],
            dim=-1,
        ).clamp(-20.0, 20.0)
        if decode_mask is not None:
            batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
            return features[batch_idx, row_idx, col_idx]
        return features

    def _decode_with_svt_features(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        src: torch.Tensor,
        point_grid: torch.Tensor,
        decode_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if decode_mask is not None:
            if decode_mask.shape != point_grid.shape:
                raise ValueError(
                    f"decode_mask shape {tuple(decode_mask.shape)} must match "
                    f"point grid shape {tuple(point_grid.shape)}"
                )
            batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
            if batch_idx.numel() == 0:
                return row_state.new_empty((0, self.num_outputs))
            rows = row_state[batch_idx, row_idx]
            cols = col_state[batch_idx, col_idx]
            point = point_grid[batch_idx, row_idx, col_idx]
            features = self.output_decoder.features_for_pairs(rows, cols)
        else:
            point = point_grid
            features = self.output_decoder.grid_features(row_state, col_state)
        if self.use_observed_neighbor_features:
            neighbor_features = self._observed_neighbor_point_features(src)
            if decode_mask is not None:
                neighbor_features = neighbor_features[batch_idx, row_idx, col_idx]
            decoder_inputs = torch.cat(
                [
                    self._svt_point_features(point),
                    neighbor_features,
                ],
                dim=-1,
            )
        else:
            decoder_inputs = self._svt_point_features(point)
        point_features = self.svt_feature_decoder_proj(
            decoder_inputs.to(device=features.device, dtype=features.dtype)
        )
        features = features + self.svt_feature_decoder_scale.to(features.dtype) * (
            point_features
        )
        if self.use_svt_magnitude_channel:
            magnitude = (point.float().sign() * point.float().abs().log1p()).unsqueeze(-1)
            features = features + self.svt_magnitude_proj(magnitude.to(features.dtype))
        if self.use_empirical_bayes_decoder:
            empirical_features = self._empirical_bayes_point_features(
                src,
                point_grid,
                decode_mask,
            )
            empirical_context = self.empirical_bayes_feature_proj(
                empirical_features.to(device=features.device, dtype=features.dtype)
            )
            features = features + self.empirical_bayes_feature_scale.to(
                features.dtype
            ) * empirical_context
        if self.use_observed_attention_decoder:
            attention_context = self._observed_attention_context(
                row_state,
                col_state,
                src,
                decode_mask,
            )
            features = features + self.observed_attention_scale.to(features.dtype) * (
                attention_context.to(device=features.device, dtype=features.dtype)
            )
        if self.use_cell_query_decoder:
            if decode_mask is not None:
                indices = (batch_idx, row_idx, col_idx)
            else:
                linear = torch.arange(point_grid.numel(), device=src.device)
                nrows, ncols = src.shape[1:]
                indices = (linear // (nrows * ncols), (linear // ncols) % nrows, linear % ncols)
            features = features + self.cell_query_decoder(features, row_state, col_state, src, indices)
        features = self.svt_feature_decoder_norm(features)
        logits = self.output_decoder.decode_features(features)
        self.last_svt_points = point.detach().float()
        if self.use_svt_output_correction:
            # Translation preserves the entire density, with unit Jacobian.
            # Centering makes the initial mean exactly SVT despite asymmetric bins.
            with torch.autocast(device_type=features.device.type, enabled=False):
                correction = F.linear(features.float(), self.svt_output_correction.weight.float(),
                                      self.svt_output_correction.bias.float()).squeeze(-1)
                base_mean = logits.float().softmax(-1) @ self.output_bucket_means.float()
                self.last_prediction_location = point.float() + correction - base_mean
        return logits

    def forward(
        self,
        src: torch.Tensor,
        return_embeddings: bool = False,
        decode_mask: torch.Tensor | None = None,
        prior_index: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_prediction_location = None
        self.last_svt_points = None
        prior_ids = self._prior_ids(src, prior_index)
        row_state, col_state = self.encode_latents(src)
        row_state, col_state = self._apply_prior_latent_adapter(
            row_state,
            col_state,
            prior_ids,
        )
        if return_embeddings:
            return self.output_decoder.grid_features(row_state, col_state)

        point_grid = self._point_grid(src, row_state, col_state, prior_ids)
        if self.use_svt_feature_decoder:
            if decode_mask is not None:
                batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
                snapshots = self.last_svt_mixture_snapshots
                if snapshots is not None and batch_idx.numel() > 0:
                    snapshot_points = snapshots[batch_idx, :, row_idx, col_idx]
                    self.last_svt_mixture_snapshot_points = snapshot_points.detach()
                    self.last_svt_mixture_snapshot_batch_idx = batch_idx.detach()
                else:
                    self.last_svt_mixture_snapshot_points = None
                    self.last_svt_mixture_snapshot_batch_idx = None
            else:
                self.last_svt_mixture_snapshot_points = None
                self.last_svt_mixture_snapshot_batch_idx = None
            return self._decode_with_svt_features(
                row_state,
                col_state,
                src,
                point_grid,
                decode_mask,
            )
        if decode_mask is not None:
            if decode_mask.shape != src.shape:
                raise ValueError(
                    f"decode_mask shape {tuple(decode_mask.shape)} must match "
                    f"input shape {tuple(src.shape)}"
                )
            point_pairs, batch_idx = self._point_pairs(point_grid, decode_mask)
            snapshots = self.last_svt_mixture_snapshots
            if snapshots is not None:
                _, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
                snapshot_points = snapshots[batch_idx, :, row_idx, col_idx]
                self.last_svt_mixture_snapshot_points = snapshot_points.detach()
                self.last_svt_mixture_snapshot_batch_idx = batch_idx.detach()
            else:
                self.last_svt_mixture_snapshot_points = None
                self.last_svt_mixture_snapshot_batch_idx = None
            return self._logits_from_point(point_pairs, prior_ids, batch_idx)
        self.last_svt_mixture_snapshot_points = None
        self.last_svt_mixture_snapshot_batch_idx = None
        return self._logits_from_point(point_grid, prior_ids)


class TabImputeModel(TabImputeModelRowColumnSVTHybrid):
    """Compatibility alias."""


__all__ = [
    "TabImputeModel",
    "TabImputeModelRowColumnSVTHybrid",
]
