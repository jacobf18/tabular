import importlib.resources
import unittest
import tempfile
from pathlib import Path

import torch

from test_rowcol_controlled_experiments import tiny_model
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.prediction_distribution import prediction_distribution
from rowcol_fixed_validation import evaluate_fixed_validation


class OutputRemediesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.src = torch.randn(2, 7, 6)
        self.src[:, ::2, ::2] = float("nan")
        self.mask = self.src.isnan()
        self.dist = FullSupportBarDistribution(torch.load(
            importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True))

    def test_magnitude_warm_start_preserves_logits_and_has_gradient(self):
        old = tiny_model(use_svt_feature_decoder=True).eval()
        new = tiny_model(use_svt_feature_decoder=True, use_svt_magnitude_channel=True).eval()
        loaded = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(loaded.missing_keys, ["svt_magnitude_proj.weight"])
        self.assertEqual(loaded.unexpected_keys, [])
        a = old(self.src, decode_mask=self.mask)
        b = new(self.src, decode_mask=self.mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.dist(b, torch.randn(b.shape[0])).mean().backward()
        self.assertGreater(new.svt_magnitude_proj.weight.grad.norm().item(), 0)

    def test_optional_branches_preserve_shared_initialization_rng(self):
        torch.manual_seed(9)
        old = tiny_model(use_svt_feature_decoder=True)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(9)
        new = tiny_model(use_svt_feature_decoder=True, use_svt_magnitude_channel=True,
                         use_svt_output_correction=True)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

    def test_validation_uses_translated_mean_and_saves_solver(self):
        model = tiny_model(use_svt_feature_decoder=True, use_svt_output_correction=True).eval()
        target = self.src.nan_to_num()
        target[self.mask] = 1000
        case = {"id": "one", "group": "one", "monitor_only": False,
                "source": self.src, "target": target, "mask": self.mask}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.pth"
            result = evaluate_fixed_validation(model, self.dist, [case], "cpu", "amp_bfloat16", path)
            saved = torch.load(path, weights_only=True)["predictions"]["one"]
            torch.testing.assert_close(saved["mean"], saved["svt_mean"], atol=1e-5, rtol=1e-5)
            self.assertEqual(result["range_slices"]["one"]["outside_fixed_mean_range"]["n"],
                             int(self.mask.sum()))

    def test_correction_mean_density_and_gradients(self):
        model = tiny_model(use_svt_feature_decoder=True, use_svt_output_correction=True).eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(self.src, decode_mask=self.mask)
        logits = logits.float()
        dist = prediction_distribution(model, self.dist)
        torch.testing.assert_close(dist.mean(logits), model.last_svt_points, atol=1e-5, rtol=1e-5)
        target = torch.randn(logits.shape[0])
        torch.testing.assert_close(dist(logits, target), self.dist(logits, target - dist.location))
        loss = dist(logits, target).mean() + (dist.mean(logits) - target).square().mean()
        loss.backward()
        gradient = model.svt_output_correction.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        with torch.no_grad():
            model.svt_output_correction.bias.fill_(1000)
            full = model(self.src).float()
            full_dist = prediction_distribution(model, self.dist)
            torch.testing.assert_close(full_dist.mean(full), model.last_svt_points + 1000)
            self.assertGreater(full_dist.mean(full).min().item(), 900)
            empty = model(self.src, decode_mask=torch.zeros_like(self.mask))
            self.assertEqual(empty.shape, (0, 5000))
            self.assertIsNone(model.last_prediction_location)


if __name__ == "__main__":
    unittest.main()
