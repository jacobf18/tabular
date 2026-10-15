#  Copyright (c) Prior Labs GmbH 2025.

# TODO: This module needs some tidying
from __future__ import annotations

import math
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


# TODO: Merge functionality from BarDistribution and FullSupportBarDistribution
class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor, *, ignore_nan_targets: bool = True):
        """Loss for a distribution over bars. The bars are defined by the borders.
        The loss is the negative log density of the distribution. The density is defined
        as a softmax over the logits, where the softmax is scaled by the width of the
        bars. This means that the density is 0 outside of the borders and the density
        is 1 on the borders.

        Args:
            borders:
                Here borders should start with min and end with max, where all values
                lie in (min,max) and are sorted
            ignore_nan_targets:
                if `True`, nan targets will be ignored,
                if `False`, an error will be raised.
        """
        super().__init__()
        assert len(borders.shape) == 1
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        self._refresh_cached_tensors()
        full_width = self.bucket_widths.sum()

        assert (1 - (full_width / (self.borders[-1] - self.borders[0]))).abs() < 1e-2, (
            f"diff: {full_width - (self.borders[-1] - self.borders[0])} with"
            f" {full_width} {self.borders[-1]} {self.borders[0]}"
        )

        # This also allows size zero buckets
        assert (self.bucket_widths >= 0.0).all(), "Please provide sorted borders!"

        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    def _set_or_register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor)

    def _refresh_cached_tensors(self) -> None:
        bucket_widths = self.borders[1:] - self.borders[:-1]
        self._set_or_register_buffer("_bucket_widths", bucket_widths)
        self._set_or_register_buffer("_log_bucket_widths", torch.log(bucket_widths))
        self._set_or_register_buffer("_bucket_means", self.borders[:-1] + bucket_widths / 2)
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        self._set_or_register_buffer(
            "_bucket_mean_of_square",
            (left_borders.square() + right_borders.square() + left_borders * right_borders) / 3.0,
        )

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self._bucket_widths

    @property
    def log_bucket_widths(self) -> torch.Tensor:
        return self._log_bucket_widths

    @property
    def bucket_means(self) -> torch.Tensor:
        return self._bucket_means

    @property
    def bucket_mean_of_square(self) -> torch.Tensor:
        return self._bucket_mean_of_square

    @property
    def num_bars(self) -> int:
        return len(self.borders) - 1

    def cdf(self, logits: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Calculates the cdf of the distribution described by the logits.
        The cdf is scaled by the width of the bars.

        Args:
            logits:
                tensor of shape (batch_size, ..., num_bars) with the logits describing
                the distribution
            ys:
                tensor of shape (batch_size, ..., n_ys to eval) or (n_ys to eval)
                with the targets.
        """
        if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
            # bring new borders to the same dim as logits up to the last dim
            ys = ys.repeat(logits.shape[:-1] + (1,))
        else:
            assert (
                ys.shape[:-1] == logits.shape[:-1]
            ), f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
        probs = torch.softmax(logits, dim=-1)
        buckets_of_ys = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)

        prob_so_far = torch.cumsum(probs, dim=-1) - probs
        prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)

        share_of_bucket_left = (
            (ys - self.borders[buckets_of_ys]) / self.bucket_widths[buckets_of_ys]
        ).clamp(0.0, 1.0)
        prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left

        prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

        # just to fix numerical inaccuracies, if we had *exact* computation above we
        # would not need the following:
        prob_left_of_ys[ys <= self.borders[0]] = 0.0
        prob_left_of_ys[ys >= self.borders[-1]] = 1.0
        assert not torch.isnan(prob_left_of_ys).any()

        return prob_left_of_ys.clip(0.0, 1.0)

    def get_probs_for_different_borders(
        self,
        logits: torch.Tensor,
        new_borders: torch.Tensor,
    ) -> torch.Tensor:
        """The logits describe the density of the distribution over the current
        self.borders.

        This function returns the logits if the self.borders were changed
        to new_borders. This is useful to average the logits of different models.
        """
        if (len(self.borders) == len(new_borders)) and (
            self.borders == new_borders
        ).all():
            return logits.softmax(-1)

        prob_left_of_borders = self.cdf(logits, new_borders)
        prob_left_of_borders[..., 0] = 0.0
        prob_left_of_borders[..., -1] = 1.0

        return (
            prob_left_of_borders[..., 1:] - prob_left_of_borders[..., :-1]
        ).clamp_min(0.0)

    def average_bar_distributions_into_this(
        self,
        list_of_bar_distributions: Sequence[BarDistribution],
        list_of_logits: Sequence[torch.Tensor],
        *,
        average_logits: bool = False,
    ) -> torch.Tensor:
        """:param list_of_bar_distributions:
        :param list_of_logits:
        :param average_logits:
        :return:
        """
        probs = torch.stack(
            [
                bar_dist.get_probs_for_different_borders(logits, self.borders)
                for bar_dist, logits in zip(list_of_bar_distributions, list_of_logits)
            ],
            dim=0,
        )

        if average_logits:
            probs = probs.log().mean(dim=0).softmax(-1)
        else:
            probs = probs.mean(dim=0)

        return probs.log()

    def __setstate__(self, state: dict) -> None:
        if "bucket_widths" in state:
            del state["bucket_widths"]
        super().__setstate__(state)
        self.__dict__.setdefault("append_mean_pred", False)
        if hasattr(self, "borders") and "_bucket_widths" not in self._buffers:
            self._refresh_cached_tensors()

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        # assert the borders are actually sorted
        assert (self.bucket_widths >= 0.0).all()
        target_sample = torch.searchsorted(self.borders, y)
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars
        return target_sample

    def ignore_init(self, y: torch.Tensor) -> torch.Tensor:
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any() and not self.ignore_nan_targets:
            raise ValueError(f"Found NaN in target {y}")

        # this is just a default value, it will be ignored anyway
        # Convert borders[0] to match y's dtype to avoid dtype mismatch
        y[ignore_loss_mask] = self.borders[0].to(y.dtype)
        return ignore_loss_mask

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # this is equivalent to log(p(y)) of the density p
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - self.log_bucket_widths

    def full_ce(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log_softmax(logits, -1)).sum(-1)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # gives the negative log density (the _loss_),
        # y: T x B, logits: T x B x self.num_bars
        y = y.reshape(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any():
            if not self.ignore_nan_targets:
                raise ValueError(f"Found NaN in target {y}")
            y = y.clone()
            y[ignore_loss_mask] = self.borders[0].to(y.dtype)
        target_sample = self.map_to_bucket_idx(y)
        assert (target_sample >= 0).all()
        assert (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"

        last_dim = logits.shape[-1]
        assert last_dim == self.num_bars, f"{last_dim} v {self.num_bars}"

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        # T x B
        nll_loss = -scaled_bucket_log_probs.gather(
            -1,
            target_sample[..., None],
        ).squeeze(-1)

        # TODO(eddiebergman): Verify if this is still relevant
        # TO BE REMOVED AFTER BO SUBMISSION
        if mean_prediction_logits is not None:
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )

        nll_loss[ignore_loss_mask] = 0.0
        return nll_loss

    # TODO(eddiebergman): Verify if this is still relevant
    def mean_loss(
        self,
        logits: torch.Tensor,
        mean_prediction_logits: torch.Tensor,
    ) -> torch.Tensor:  # TO BE REMOVED AFTER BO SUBMISSION
        scaled_mean_log_probs = self.compute_scaled_log_probs(mean_prediction_logits)
        if not self.training:
            pass
        assert len(logits.shape) == 3, len(logits.shape)
        assert len(scaled_mean_log_probs.shape) == 2, len(scaled_mean_log_probs.shape)

        means = self.mean(logits).detach()  # T x B

        # T x B
        target_mean = self.map_to_bucket_idx(means).clamp_(0, self.num_bars - 1)

        # 1 x B
        return -scaled_mean_log_probs.gather(1, target_mean.T).mean(1).unsqueeze(0)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, -1)
        return p @ self.bucket_means.to(device=logits.device, dtype=logits.dtype)

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    # TODO(eddiebergman): Check if still relevant
    def cdf_temporary(self, logits: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function.

        TODO: this already exists here, make sure to merge, at the moment still used.
        """
        probs = logits.softmax(-1)
        return -torch.cumsum(probs, -1).sum(axis=-1)  # type: ignore

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        """Implementation of the quantile function
        :param logits: Tensor of any shape, with the last dimension being logits
        :param left_prob: float: The probability mass to the left of the result.
        :return: Position with `left_prob` probability weight to the left.
        """
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1)
        idx = (
            torch.searchsorted(
                cumprobs,
                torch.full_like(cumprobs[..., :1], left_prob),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )  # this might not do the right for outliers
        cumprobs = torch.cat(
            [torch.zeros_like(cumprobs[..., :1]), cumprobs],
            -1,
        )

        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1,
            idx[..., None],
        ).squeeze(-1)

    def quantile(
        self,
        logits: torch.Tensor,
        center_prob: float = 0.682,
    ) -> torch.Tensor:
        side_probs = (1.0 - center_prob) / 2
        return torch.stack(
            (self.icdf(logits, side_probs), self.icdf(logits, 1.0 - side_probs)),
            -1,
        )

    def ucb(
        self,
        logits: torch.Tensor,
        best_f: float,  # noqa: ARG002
        rest_prob: float = (1 - 0.682) / 2,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """UCB utility. Rest Prob is the amount of utility above (below) the confidence
        interval that is ignored.

        Higher rest_prob is equivalent to lower beta in the standard GP-UCB formulation.

        Args:
            logits: Logits, as returned by the Transformer.
            rest_prob:
                The amount of utility above (below) the confidence interval that is
                ignored.

                The default is equivalent to using GP-UCB with `beta=1`.
                To get the corresponding `beta`, where `beta` is from
                the standard GP definition of UCB `ucb_utility = mean + beta * std`,
                you can use this computation:

                `beta = math.sqrt(2)*torch.erfinv(torch.tensor(2*(1-rest_prob)-1))`
            best_f: Unused
            maximize: Whether to maximize.
        """
        if maximize:
            rest_prob = 1 - rest_prob
        return self.icdf(logits, rest_prob)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)
        return self.bucket_means[mode_inds]

    def ei(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:  # logits: evaluation_points x batch x feature_dim
        bucket_diffs = self.bucket_widths
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)  # type: ignore

        best_f = best_f[..., None].repeat(*[1] * len(best_f.shape), logits.shape[-1])  # type: ignore
        clamped_best_f = best_f.clamp(self.borders[:-1], self.borders[1:])

        # > bucket_contributions =
        # >     (best_f[...,None] < self.borders[:-1]).float() * bucket_means
        # true bucket contributions
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def pi(
        self,
        logits: torch.Tensor,
        best_f: float | torch.Tensor,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:  # logits: evaluation_points x batch x feature_dim
        """Acquisition Function: Probability of Improvement.

        Args:
            logits: as returned by Transformer
            best_f: best evaluation so far (the incumbent)
            maximize: whether to maximize

        Returns:
            probability of improvement
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)  # type: ignore
        p = torch.softmax(logits, -1)
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / self.bucket_widths).clamp(  # type: ignore
            0.0,
            1.0,
        )
        return (p * factor).sum(-1)

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes E[x^2].

        Args:
            logits: Output of the model.

        Returns:
            mean of square
        """
        p = torch.softmax(logits, -1)
        return p @ self.bucket_mean_of_square.to(device=logits.device, dtype=logits.dtype)

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    # TODO: Move into standalone module for plotting
    def plot(
        self,
        logits: torch.Tensor,
        ax: plt.Axes | None = None,
        zoom_to_quantile: float | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plots the distribution."""
        import matplotlib.pyplot as plt

        logits = logits.squeeze()
        assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
        if ax is None:
            ax = plt.gca()
        if zoom_to_quantile is not None:
            lower_bounds, upper_bounds = self.quantile(
                logits,
                zoom_to_quantile,
            ).transpose(0, -1)
            lower_bound = lower_bounds.min().item()
            upper_bound = upper_bounds.max().item()
            ax.set_xlim(lower_bound, upper_bound)
            border_mask = (self.borders[:-1] >= lower_bound) & (
                self.borders[1:] <= upper_bound
            )
        else:
            border_mask = slice(None)
        p = torch.softmax(logits, -1) / self.bucket_widths
        ax.bar(
            self.borders[:-1][border_mask],
            p[border_mask],
            self.bucket_widths[border_mask],
            **kwargs,
        )
        return ax


class FullSupportBarDistribution(BarDistribution):
    def __init__(
        self,
        borders: torch.Tensor,
        **kwargs: Any,
    ):
        # here borders should start with min and end with max, where all values
        # lie in (min,max) and are sorted
        """:param borders:"""
        super().__init__(borders, **kwargs)
        self.assert_support(allow_zero_bucket_left=False)
        self._refresh_full_support_cached_tensors()

        losses_per_bucket = torch.zeros_like(self.bucket_widths)
        self.register_buffer("losses_per_bucket", losses_per_bucket)

    def _refresh_full_support_cached_tensors(self) -> None:
        self._refresh_cached_tensors()
        median_halfnormal_unit = self.bucket_widths.new_tensor(0.6744897501960817)
        side_scale_left = self.bucket_widths[0] / median_halfnormal_unit
        side_scale_right = self.bucket_widths[-1] / median_halfnormal_unit
        side_mean_left = side_scale_left * math.sqrt(2.0 / math.pi)
        side_mean_right = side_scale_right * math.sqrt(2.0 / math.pi)
        side_var_left = side_scale_left.square() * (1.0 - 2.0 / math.pi)
        side_var_right = side_scale_right.square() * (1.0 - 2.0 / math.pi)

        full_bucket_means = self.bucket_means.clone()
        full_bucket_means[0] = -side_mean_left + self.borders[1]
        full_bucket_means[-1] = side_mean_right + self.borders[-2]

        full_bucket_mean_of_square = self.bucket_mean_of_square.clone()
        full_bucket_mean_of_square[0] = side_var_left + full_bucket_means[0].square()
        full_bucket_mean_of_square[-1] = side_var_right + full_bucket_means[-1].square()

        self._set_or_register_buffer("_side_scale_left", side_scale_left)
        self._set_or_register_buffer("_side_scale_right", side_scale_right)
        self._set_or_register_buffer("_side_mean_left", side_mean_left)
        self._set_or_register_buffer("_side_mean_right", side_mean_right)
        self._set_or_register_buffer("_side_var_left", side_var_left)
        self._set_or_register_buffer("_side_var_right", side_var_right)
        self._set_or_register_buffer("_full_bucket_means", full_bucket_means)
        self._set_or_register_buffer("_full_bucket_mean_of_square", full_bucket_mean_of_square)

    @staticmethod
    def _halfnormal_log_prob(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale = scale.to(device=x.device, dtype=x.dtype)
        return (
            x.new_tensor(0.5 * math.log(2.0 / math.pi))
            - torch.log(scale)
            - x.square() / (2.0 * scale.square())
        )

    @staticmethod
    def _halfnormal_cdf(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale = scale.to(device=x.device, dtype=x.dtype)
        return torch.erf(x / (scale * math.sqrt(2.0)))

    def assert_support(self, *, allow_zero_bucket_left: bool = False) -> None:
        if allow_zero_bucket_left:
            assert (
                self.bucket_widths[-1] > 0
            ), f"Half Normal weight must be > 0 (got -1:{self.bucket_widths[-1]})."
            # This fixes the distribution if the half normal at zero is width zero
            if self.bucket_widths[0] == 0:
                self.borders[0] = self.borders[0] - 1
                self.bucket_widths[0] = 1.0
                self._refresh_full_support_cached_tensors()
        else:
            assert self.bucket_widths[0] > 0
            assert self.bucket_widths[-1] > 0

    @staticmethod
    def halfnormal_with_p_weight_before(
        range_max: float,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p),
        )
        return torch.distributions.HalfNormal(s)

    @override
    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mean_prediction_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the negative log density (the _loss_).

        y: T x B, logits: T x B x self.num_bars.

        :param logits: Tensor of shape T x B x self.num_bars
        :param y: Tensor of shape T x B
        :param mean_prediction_logits:
        :return:
        """
        assert self.num_bars > 1
        y = y.reshape(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask = torch.isnan(y)
        if ignore_loss_mask.any():
            if not self.ignore_nan_targets:
                raise ValueError(f"Found NaN in target {y}")
            y = y.clone()
            y[ignore_loss_mask] = self.borders[0].to(y.dtype)
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        assert (target_sample >= 0).all()
        assert (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        last_dim = logits.shape[-1]
        assert last_dim == self.num_bars, f"{last_dim} vs {self.num_bars}"
        # ignore all position with nan values

        flat_logits = logits.reshape(-1, last_dim)
        flat_targets = target_sample.reshape(-1)
        nll_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="none",
        ).view_as(target_sample)
        log_probs = -nll_loss - self.log_bucket_widths[target_sample]

        left_tail_mask = target_sample == 0
        if left_tail_mask.any():
            left_tail_x = (self.borders[1] - y[left_tail_mask]).clamp(min=0.00000001)
            log_probs[left_tail_mask] += self._halfnormal_log_prob(
                left_tail_x,
                self._side_scale_left,
            ) + self.log_bucket_widths[0]
        right_tail_mask = target_sample == self.num_bars - 1
        if right_tail_mask.any():
            right_tail_x = (y[right_tail_mask] - self.borders[-2]).clamp(min=0.00000001)
            log_probs[right_tail_mask] += self._halfnormal_log_prob(
                right_tail_x,
                self._side_scale_right,
            ) + self.log_bucket_widths[-1]

        nll_loss = -log_probs

        if mean_prediction_logits is not None:  # TO BE REMOVED AFTER BO PAPER IS DONE
            assert (
                not ignore_loss_mask.any()
            ), "Ignoring examples is not implemented with mean pred."
            if not torch.is_grad_enabled():
                pass
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, mean_prediction_logits)),
                0,
            )

        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0

        # # TODO: Check with samuel whether to keep
        # self.losses_per_bucket += (
        #     torch.scatter(
        #         self.losses_per_bucket,
        #         0,
        #         target_sample[~ignore_loss_mask].flatten(),
        #         nll_loss[~ignore_loss_mask].flatten().detach(),
        #     )
        #     / target_sample[~ignore_loss_mask].numel()
        # )

        return nll_loss

    def pdf(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Probability density function at y."""
        return torch.exp(self.forward(logits, y))

    def sample(self, logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Samples values from the distribution.

        Temperature t.
        """
        p_cdf = torch.rand(*logits.shape[:-1])
        return torch.tensor(
            [self.icdf(logits[i, :] / t, p) for i, p in enumerate(p_cdf.tolist())], # we need to fix this
        )

    @override
    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, -1)
        return p @ self._full_bucket_means.to(device=logits.device, dtype=logits.dtype)

    @override
    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes E[x^2].

        Args:
            logits: Output of the model.
        """
        p = torch.softmax(logits, -1)
        return p @ self._full_bucket_mean_of_square.to(device=logits.device, dtype=logits.dtype)

    @override
    def pi(
        self,
        logits: torch.Tensor,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """Acquisition Function: Probability of Improvement.

        Args:
            logits: as returned by Transformer (evaluation_points x batch x feature_dim)
            best_f: best evaluation so far (the incumbent)
            maximize: whether to maximize
        """
        # logits: evaluation_points x batch x feature_dim
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            # evaluation_points x batch
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)  # type: ignore

        assert best_f.shape == logits[..., 0].shape, (  # type: ignore
            f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"  # type: ignore
        )
        p = torch.softmax(logits, -1)  # evaluation_points x batch
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / self.bucket_widths).clamp(  # type: ignore
            0.0,
            1.0,
        )  # evaluation_points x batch x num_bars
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch
        factor[..., 0] = 0.0
        factor[..., 0][position_in_side_normals[0] > 0.0] = self._halfnormal_cdf(
            position_in_side_normals[0][position_in_side_normals[0] > 0.0],
            self._side_scale_left,
        )
        factor[..., -1] = 1.0
        factor[..., -1][position_in_side_normals[1] > 0.0] = 1.0 - self._halfnormal_cdf(
            position_in_side_normals[1][position_in_side_normals[1] > 0.0],
            self._side_scale_right,
        )
        return (p * factor).sum(-1)

    def ei_for_halfnormal(
        self,
        scale: float,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        """EI for a standard normal distribution with mean 0 and variance `scale` times 2.

        Which is the same as the half normal EI. Tested this with MC approximation:

        ```python
        ei_for_halfnormal = lambda scale, best_f: (torch.distributions.HalfNormal(torch.tensor(scale)).sample((10_000_000,))- best_f ).clamp(min=0.).mean()
        print([(ei_for_halfnormal(scale,best_f), FullSupportBarDistribution().ei_for_halfnormal(scale,best_f)) for scale in [0.1,1.,10.] for best_f in [.1,10.,4.]])
        ```
        """  # noqa: E501
        assert maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        scale = scale.to(device=best_f.device, dtype=best_f.dtype) if torch.is_tensor(scale) else best_f.new_tensor(scale)
        mean = torch.zeros((), device=best_f.device, dtype=best_f.dtype)
        u = (mean - best_f) / scale
        ucdf = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
        updf = torch.exp(-0.5 * u.square()) / math.sqrt(2.0 * math.pi)
        normal_ei = scale * (updf + u * ucdf)
        return 2 * normal_ei

    @override
    def ei(
        self,
        logits: torch.Tensor,
        best_f: torch.Tensor | float,
        *,
        maximize: bool = True,
    ) -> torch.Tensor:
        # logits: evaluation_points x batch x feature_dim
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        bucket_diffs = self.bucket_widths
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):  # type: ignore
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)  # type: ignore

        assert best_f.shape == logits[..., 0].shape, (  # type: ignore
            f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"  # type: ignore
        )

        best_f_per_logit = best_f[..., None].repeat(  # type: ignore
            *[1] * len(best_f.shape),  # type: ignore
            logits.shape[-1],
        )
        clamped_best_f = best_f_per_logit.clamp(self.borders[:-1], self.borders[1:])

        # true bucket contributions
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f_per_logit * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        # extra stuff for continuous
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch

        bucket_contributions[..., -1] = self.ei_for_halfnormal(
            self._side_scale_right,
            position_in_side_normals[1],
        )

        bucket_contributions[..., 0] = self.ei_for_halfnormal(
            self._side_scale_left,
            torch.zeros_like(position_in_side_normals[0]),
        ) - self.ei_for_halfnormal(self._side_scale_left, position_in_side_normals[0])

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)


def get_bucket_limits(
    num_outputs: int,
    full_range: tuple | None = None,
    ys: torch.Tensor | None = None,
    *,
    verbose: bool = False,  # noqa: ARG001
    widen_bucket_limits_factor: float | None = None,
) -> torch.Tensor:
    """Decide for a set of bucket limits based on a distritbution of ys.

    Args:
        num_outputs:
            This is only tested for num_outputs=1, but should work for larger
            num_outputs as well.
        full_range:
            If ys is not passed, this is the range of the ys that should be
            used to estimate the bucket limits.
        ys:
            If ys is passed, this is the ys that should be used to estimate the bucket
            limits. Do not pass full_range in this case.
        verbose: Unused
        widen_bucket_limits_factor:
            If set, the bucket limits are widened by this factor.
            This allows to have a slightly larger range than the actual data.
    """
    assert (ys is None) != (
        full_range is None
    ), "Either full_range or ys must be passed."

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        assert (
            len(ys) > num_outputs
        ), f"Number of ys :{len(ys)} must be larger than num_outputs: {num_outputs}"
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min()
            assert full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)  # type: ignore

        ys_sorted, ys_order = ys.sort(0)  # type: ignore
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],  # type: ignore
            0,
        )
        if widen_bucket_limits_factor is not None:
            bucket_limits = bucket_limits * widen_bucket_limits_factor

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs  # type: ignore
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,  # type: ignore
                torch.tensor(full_range[1]).unsqueeze(0),  # type: ignore
            ],
            0,
        )

    assert len(bucket_limits) - 1 == num_outputs, (
        f"len(bucket_limits) - 1 == {len(bucket_limits) - 1}"
        f" != {num_outputs} == num_outputs"
    )

    if not widen_bucket_limits_factor or widen_bucket_limits_factor == 1.0:
        assert (
            full_range[0] == bucket_limits[0]  # type: ignore
        ), f"{full_range[0]} != {bucket_limits[0]}"  # type: ignore
        assert (
            full_range[-1] == bucket_limits[-1]  # type: ignore
        ), f"{full_range[-1]} != {bucket_limits[-1]}"  # type: ignore

    return bucket_limits


def get_custom_bar_dist(borders: torch.Tensor, criterion: nn.Module) -> nn.Module:
    # Tested that a bar_dist with borders 0.54 (-> softplus 1.0) yields the same
    # bar distribution as the passed one.
    borders_ = torch.nn.functional.softplus(borders) + 0.001
    borders_ = torch.cumsum(
        torch.cat([criterion.borders[0:1], criterion.bucket_widths]) * borders_,
        0,
    )
    return criterion.__class__(
        borders=borders_,
        handle_nans=criterion.handle_nans,
    )
