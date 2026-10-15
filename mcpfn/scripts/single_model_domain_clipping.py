"""Training-only domain-gradient clipping before the original equal-weight mean."""

from contextlib import contextmanager
import math

import torch


SPECIFICATION = {"domain_order": ["imputation", "recsys"], "domain_weight": .5, "domain_unweighted_clip": .5,
                 "weighted_clip": .25, "final_global_clip": .5,
                 "policy": "Clip each isolated weighted domain gradient to0.25, then sum; retain the original final global0.5 guard and unchanged scalar losses"}
ADMISSION = {"minimum_structured_median_norm_ratio_exclusive": 2.,
             "rating_adverse_sgd_cases_must_decrease": True, "imputation_adverse_sgd_cases_must_not_increase": True,
             "scope": "Exploratory synthetic gradient evidence only; not an AdamW-performance guarantee"}


def admitted(audit):
    if (not audit.get("complete") or audit.get("pairs") != 64 or audit.get("gradient_vectors") != 128
            or not audit.get("all_original_coordinates_remeasured") or not audit.get("original_artifacts_unchanged")
            or audit.get("cuda_initialized") or not audit.get("runtime_reported_model_unchanged")):
        raise ValueError("Require the completed original read-only64-pair CPU audit")
    structured, all_cases = audit["groups"]["structured"], audit["groups"]["all"]
    if structured["defined_norm_ratios"] != 48 or all_cases["cases"] != 64:
        raise ValueError("Never omit undefined domain-gradient measurements")
    current, proposed = [all_cases["methods"][name] for name in ("joint_clip", "domain_clip_then_mean")]
    return (structured["median_imputation_to_rating_norm"] > ADMISSION["minimum_structured_median_norm_ratio_exclusive"]
            and proposed["rating_negative_descent_cases"] < current["rating_negative_descent_cases"]
            and proposed["imputation_negative_descent_cases"] <= current["imputation_negative_descent_cases"])


class DomainGradientPair:
    def __init__(self, backward):
        self.backward = backward
        self.pending = None

    def __call__(self, model, criterion, episode, teacher, weight):
        domain = episode.metadata.get("domain")
        expected = "imputation" if self.pending is None else "recsys"
        if (domain != expected or episode.metadata.get("gradient_data") != "synthetic"
                or weight != SPECIFICATION["domain_weight"]):
            raise ValueError("Require the original ordered, equally weighted synthetic domain pair")
        if self.pending is not None and self.pending[0] is not model:
            raise ValueError("Never combine gradients from different student models")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Each domain must start with isolated empty gradients")
        loss = self.backward(model, criterion, episode, teacher, weight)
        if any(parameter.grad is None for parameter in model.parameters()):
            raise ValueError("A training parameter has no domain gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), SPECIFICATION["weighted_clip"], error_if_nonfinite=True))
        coefficient = min(1., SPECIFICATION["weighted_clip"] / (norm + 1e-6))
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite unchanged scalar domain loss")
        episode.metadata["domain_gradient_clipping"] = {"domain_weight": weight, "weighted_clip": SPECIFICATION["weighted_clip"],
            "unweighted_clip": SPECIFICATION["domain_unweighted_clip"], "weighted_norm_before_clip": norm,
            "clip_coefficient": coefficient, "weighted_norm_after_clip": norm * coefficient,
            "gradient_isolated_before_backward": True}
        if domain == "imputation":
            self.pending = (model, [parameter.grad for parameter in model.parameters()])
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                for parameter, imputation_gradient in zip(model.parameters(), self.pending[1]):
                    parameter.grad.add_(imputation_gradient)
            self.pending = None
        return loss


@contextmanager
def runtime(backend):
    original = backend.backward_episode
    accumulator = DomainGradientPair(original)
    backend.backward_episode = accumulator
    try:
        yield accumulator
        if accumulator.pending is not None:
            raise ValueError("A domain pair ended before its rating contribution")
    finally:
        backend.backward_episode = original
        accumulator.pending = None
