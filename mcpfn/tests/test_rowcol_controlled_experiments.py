import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rowcol_fixed_validation import apply_support_mask, evaluate_fixed_validation, preserved_rng
from tabimpute.prior.training_set_generation import MISSINGNESS_PROFILES, ProfileMixedPattern
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def tiny_model(**kwargs):
    return TabImputeModelRowColumnSVTHybrid(
        embedding_size=32, mlp_hidden_size=48, num_attention_heads=4, num_layers=1,
        num_outputs=5000, num_cls=0, use_rope=False, use_activation_checkpointing=False,
        rabel_num_kernels=8, rabel_exponent_embedding_size=8, count_hidden_size=8,
        svt_init_rank=3, svt_max_rank=5, svt_steps=2, svt_mixture_steps=[0, 1, 2],
        svt_mixture_hidden_size=32, svt_mixture_cv_fraction=0.3, svt_mixture_cv_min_count=1,
        svt_solver_precision="float32", svt_mixture_cv_reencode=True, **kwargs,
    )


class ControlledExperimentsTest(unittest.TestCase):
    def test_profile_overrides_and_global_defaults(self):
        original = copy.deepcopy(MISSINGNESS_PROFILES)
        default = ProfileMixedPattern({})
        self.assertEqual(default.profile, original["missmix_v1"])
        configured = ProfileMixedPattern({"profile_p_missing_anchor_prob": 1.0,
                                          "profile_p_missing_anchor_value": 0.23})
        configured.weights = {"mcar": 1.0}
        with patch("tabimpute.prior.training_set_generation.MCARPattern._induce_missingness",
                   autospec=True, side_effect=lambda self, x: torch.full_like(x, self.config["p_missing"])):
            result = configured._induce_missingness(torch.ones(5, 4))
        torch.testing.assert_close(result, torch.full_like(result, 0.23))
        self.assertEqual(original, MISSINGNESS_PROFILES)
        with self.assertRaises(ValueError):
            ProfileMixedPattern({"profile_p_missing_low": 0.9, "profile_p_missing_width": 0.3})

    def test_support_does_not_reveal_values_or_depend_on_targets(self):
        source = torch.randn(2, 30, 20)
        source[:, ::3, ::4] = float("nan")
        target = source.nan_to_num()
        x, y, mask, _ = apply_support_mask(source, target, "sparse_query", seed=42)
        x2, _, mask2, _ = apply_support_mask(source, target + 100, "sparse_query", seed=42)
        torch.testing.assert_close(x, x2, equal_nan=True)
        self.assertTrue(torch.equal(mask, mask2))
        self.assertTrue(torch.isnan(x)[torch.isnan(source)].all())
        self.assertTrue(torch.isnan(x)[mask].all())
        self.assertTrue(mask.any())
        self.assertTrue(torch.isfinite(y).all())

    def test_rng_isolation(self):
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        expected = (random.random(), float(np.random.rand()), torch.rand(2))
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        with preserved_rng(99):
            random.random(), np.random.rand(), torch.rand(8)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(float(np.random.rand()), expected[1])
        torch.testing.assert_close(torch.rand(2), expected[2])

    def test_solver_fp32_reconstruction_inside_autocast(self):
        model = tiny_model().eval()
        src = torch.randn(1, 9, 7)
        src[:, ::2, ::2] = float("nan")
        initial = torch.randn_like(src)
        expected = model._svt_iterates(src, initial, None, (0, 1, 2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model._svt_iterates(src, initial, None, (0, 1, 2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cv_initialization_excludes_validation_values(self):
        model = tiny_model().train()
        src = torch.randn(2, 12, 9)
        validation = model._svt_mixture_cv_mask(src)
        changed = src.clone()
        changed[validation] += 100
        captured = []
        original = model._svt_iterates
        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result.detach().clone())
            return result
        with patch.object(model, "_svt_iterates", side_effect=record):
            model._svt_mixture_cv_rmse(src, torch.zeros_like(src), None)
            model._svt_mixture_cv_rmse(changed, torch.full_like(src, 1000), None)
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        self.assertTrue(model.training)

    def test_validation_restores_modes_and_saves_predictions(self):
        from tabimpute.model.bar_distribution import FullSupportBarDistribution
        import importlib.resources
        borders = torch.load(importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True)
        criterion = FullSupportBarDistribution(borders)
        model = tiny_model().train()
        src = torch.randn(1, 9, 7)
        target = src.clone()
        src[:, ::2, ::2] = float("nan")
        cases = [{"id": "case", "group": "group", "monitor_only": False,
                  "source": src, "target": target, "mask": src.isnan()}]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "predictions.pth"
            result = evaluate_fixed_validation(model, criterion, cases, "cpu", "float32", p)
            self.assertTrue(np.isfinite(result["macro_rmse"]))
            self.assertTrue(p.exists())
            saved = torch.load(p, weights_only=True)
            error = saved["predictions"]["case"]["mean"] - target[src.isnan()]
            self.assertAlmostEqual(result["macro_rmse"], error.double().square().mean().sqrt().item())
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
