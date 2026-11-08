"""Bind unchanged quarter-step reconstruction to the four-draw protection module."""

import inspect
import types

import torch

import single_model_fixed_quarter_trial as template
import single_model_four_draw_protection as protection


POLICY = (
    "Use the original exact quarter-step candidate reconstruction and independent saved-vector "
    "checks with the registered four-draw protection module. Preserve original projection, "
    "rounding, device, finite decisions, rollback and no-endpoint behavior. Only the protection "
    "module identity differs from the two-draw helper; no reconstruction arithmetic changes."
)
for name in ("check_preflight_evidence", "require_report", "reconstruct_candidate", "check_replayed_candidate"):
    function = getattr(template, name)
    primitive = inspect.unwrap(function)
    if primitive.__closure__ is not None:
        raise ValueError("Never retain a two-draw reconstruction callback closure")
    cloned = types.FunctionType(primitive.__code__, globals(), primitive.__name__, primitive.__defaults__)
    cloned.__kwdefaults__ = primitive.__kwdefaults__
    globals()[name] = torch.no_grad()(cloned) if primitive is not function else cloned
