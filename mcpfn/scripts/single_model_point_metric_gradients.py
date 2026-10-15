"""Prediction-space subgradients of the frozen synthetic point-metric definitions."""

import math

import numpy as np
import torch


SPECIFICATION = {
    "domain": "synthetic only; diagnostic helper, not a training recipe or selection change",
    "imputation": "pooled query RMSE and equal-column common NRMSE with original NumPy target-dtype population std and threshold1e-3",
    "recsys": "equal-user mean RMSE and MAE after original cached two-way inverse scaling and prediction clamp to[1,5]",
    "zero_residual": "zero norm/absolute-value subgradient",
    "clamp_boundary": "PyTorch clamp subgradient: one at exact1 and5, zero outside",
    "undefined_common": "None if every queried column has truth std below1e-3; never a fabricated zero metric",
    "canonical_nrmse_gradients_not_claimed": True,
    "parameter_gradients_not_computed_here": True,
}


def validate(prediction, episode):
    if episode.metadata.get("gradient_data") != "synthetic" or episode.metadata.get("domain") not in ("imputation", "recsys"):
        raise ValueError("Only declared synthetic episodes may enter point-gradient diagnostics")
    if (prediction.ndim != 3 or prediction.shape[0] != 1 or min(prediction.shape) < 1
            or episode.target.shape != prediction.shape or episode.mask.shape != prediction.shape
            or episode.mask.dtype != torch.bool or not episode.mask.any()
            or prediction.device != episode.target.device or prediction.device != episode.mask.device
            or not prediction.is_floating_point() or not episode.target.is_floating_point()
            or not torch.isfinite(prediction[episode.mask]).all() or not torch.isfinite(episode.target[episode.mask]).all()):
        raise ValueError("Require one aligned finite queried prediction/target grid and a nonempty boolean query mask")
    if (episode.metadata["domain"] == "recsys") != (episode.ratings is not None):
        raise ValueError("Domain and cached rating normalization must agree")


def term(value, gradient, *, mask, queries, defined_units, metric):
    if (not math.isfinite(value) or value < 0 or not torch.isfinite(gradient).all()
            or gradient.dtype != torch.float64 or queries < 1 or defined_units < 1):
        raise ValueError("Invalid finite metric value, coverage or FP64 derivative")
    return {"metric": metric, "value": value, "gradient": gradient, "query_mask": mask.detach().clone(), "queries": queries,
            "defined_units": defined_units, "zero_residual_subgradient": 0.}


@torch.no_grad()
def imputation_terms(prediction, episode):
    mask = episode.mask
    error = torch.where(mask, prediction.double() - episode.target.double(), 0.)
    queries = int(mask.sum())
    rmse = float(torch.linalg.vector_norm(error) / math.sqrt(queries))
    gradient = error / (queries * rmse) if rmse else torch.zeros_like(error)
    result = {"rmse": term(rmse, gradient, mask=mask, queries=queries, defined_units=queries, metric="pooled_query_rmse")}
    truth = episode.target[0].detach().cpu().numpy()
    selected = mask[0].detach().cpu().numpy()
    scales = np.zeros(prediction.shape[-1], dtype=np.float64)
    for column in range(prediction.shape[-1]):
        if selected[:, column].any():
            scales[column] = float(np.std(truth[selected[:, column], column]))
    if not np.isfinite(scales).all():
        raise ValueError("Original target-dtype NumPy column variance must remain finite")
    valid = scales >= 1e-3
    if not valid.any():
        result["common_nrmse"] = None
        return result
    counts = mask.sum(dim=1).double()
    column_rmse = error.square().sum(dim=1).div(counts.clamp_min(1)).sqrt()
    scales_tensor = torch.as_tensor(scales, device=prediction.device, dtype=torch.float64).unsqueeze(0)
    valid_tensor = torch.as_tensor(valid, device=prediction.device).unsqueeze(0)
    units = int(valid.sum())
    value = float((column_rmse[valid_tensor] / scales_tensor[valid_tensor]).mean())
    coefficients = torch.zeros_like(column_rmse)
    nonzero = valid_tensor & (column_rmse > 0)
    coefficients[nonzero] = 1. / (units * counts[nonzero] * scales_tensor[nonzero] * column_rmse[nonzero])
    result["common_nrmse"] = term(value, error * coefficients[:, None, :], mask=mask, queries=queries,
                                 defined_units=units, metric="truth_defined_common_column_nrmse")
    return result


@torch.no_grad()
def rating_terms(prediction, episode):
    ratings = episode.ratings
    rows, columns = prediction.shape[1:]
    for name, size in (("means", columns), ("stds", columns), ("row_offsets", rows), ("row_scales", rows)):
        value = getattr(ratings, name)
        if value.shape != (size,) or value.device != prediction.device or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Preserve the original finite cached rating scaling vectors")
        if name in ("stds", "row_scales") and not (value > 0).all():
            raise ValueError("Rating scales must be positive")
    location = ratings.means[None, None, :] + ratings.row_offsets[None, :, None]
    scale = ratings.stds[None, None, :] * ratings.row_scales[None, :, None]
    if not torch.isfinite(location).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Cached rating affine transform must remain finite and positive")
    raw = prediction.double() * scale + location
    truth = episode.target.double() * scale + location
    error = torch.where(episode.mask, raw.clamp(1., 5.) - truth, 0.)
    counts = episode.mask.sum(dim=-1).double()
    supported = counts > 0
    units, queries = int(supported.sum()), int(episode.mask.sum())
    row_rmse = error.square().sum(dim=-1).div(counts.clamp_min(1)).sqrt()
    row_mae = error.abs().sum(dim=-1).div(counts.clamp_min(1))
    chain = torch.where(episode.mask & (raw >= 1.) & (raw <= 5.), scale.double(), 0.)
    rmse_coefficients = torch.zeros_like(row_rmse)
    nonzero = supported & (row_rmse > 0)
    rmse_coefficients[nonzero] = 1. / (units * counts[nonzero] * row_rmse[nonzero])
    mae_coefficients = torch.zeros_like(row_rmse)
    mae_coefficients[supported] = 1. / (units * counts[supported])
    return {"rmse": term(float(row_rmse[supported].mean()), error * chain * rmse_coefficients[:, :, None],
                         mask=episode.mask, queries=queries, defined_units=units, metric="equal_user_clamped_rmse"),
            "mae": term(float(row_mae[supported].mean()), error.sign() * chain * mae_coefficients[:, :, None],
                        mask=episode.mask, queries=queries, defined_units=units, metric="equal_user_clamped_mae")}


@torch.no_grad()
def point_metric_terms(prediction, episode):
    validate(prediction, episode)
    return (imputation_terms if episode.ratings is None else rating_terms)(prediction, episode)


@torch.no_grad()
def linearization(prediction, mask, record):
    if record is None:
        raise ValueError("Never replace an undefined common metric by a zero objective")
    if (prediction.shape != record["gradient"].shape or mask.shape != prediction.shape or mask.dtype != torch.bool
            or prediction.device != record["gradient"].device or mask.device != prediction.device
            or not torch.equal(mask, record["query_mask"])
            or int(mask.sum()) != record["queries"] or not torch.isfinite(prediction[mask]).all()
            or not torch.isfinite(record["gradient"]).all() or (record["gradient"][~mask] != 0).any()):
        raise ValueError("Linearization must retain the original query grid and metric derivative")
    coefficients = record["gradient"] * record["queries"]
    offset = record["value"] - float((prediction[mask].double() * record["gradient"][mask]).sum())
    return coefficients, offset
