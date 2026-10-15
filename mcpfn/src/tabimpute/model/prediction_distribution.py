"""Bind a forward pass's optional translation to the categorical density."""
from __future__ import annotations


class TranslatedBarDistribution:
    def __init__(self, base, location):
        self.base = base
        self.location = location.float()

    def __call__(self, logits, target):
        return self.base(logits, target - self.location)

    def mean(self, logits):
        return self.base.mean(logits) + self.location

    def median(self, logits):
        return self.base.median(logits) + self.location

    def sample(self, logits, **kwargs):
        return self.base.sample(logits=logits, **kwargs) + self.location

    def map_to_bucket_idx(self, target):
        return self.base.map_to_bucket_idx(target - self.location)

    def __getattr__(self, name):
        # Unsupported density operations must not silently ignore translation.
        if name in {"num_bars", "borders", "bucket_means"}:
            return getattr(self.base, name)
        raise AttributeError(name)


def prediction_distribution(model, base):
    """Call immediately after forward, before another forward overwrites context."""
    location = getattr(model, "last_prediction_location", None)
    return base if location is None else TranslatedBarDistribution(base, location)


def distribution_target(model, target):
    location = getattr(model, "last_prediction_location", None)
    return target if location is None else target - location
