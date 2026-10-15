import copy
import unittest

import torch

from test_rowcol_controlled_experiments import tiny_model
from tabimpute.model.cell_query_decoder import CellQueryDecoder, CellQueryLayer


class CellQueryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(39)
        self.src = torch.randn(2, 7, 6)
        self.src[:, ::2, ::2] = float("nan")
        self.indices = self.src.isnan().nonzero(as_tuple=True)
        self.rows = torch.randn(2, 7, 16)
        self.cols = torch.randn(2, 6, 16)
        self.features = torch.randn(len(self.indices[0]), 24)

    def module(self, chunk=5):
        module = CellQueryDecoder(16, 24, width=16, heads=4, layers=2, chunk_size=chunk)
        nn = torch.nn
        nn.init.normal_(module.output.weight, std=.02)
        return module

    def test_chunking_and_checkpoint_gradients(self):
        a = self.module(100).train()
        a.checkpoint_chunks = False
        b = copy.deepcopy(a)
        b.chunk_size = 5
        b.checkpoint_chunks = True
        x = a(self.features, self.rows, self.cols, self.src, self.indices)
        y = b(self.features, self.rows, self.cols, self.src, self.indices)
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        x.square().mean().backward()
        y.square().mean().backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(q.grad).all(), name)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
        self.assertGreater(b.value_proj[0].weight.grad.norm().item(), 0)
        self.assertGreater(b.layers[0].q.weight.grad.norm().item(), 0)

    def test_empty_support_and_empty_queries(self):
        m = self.module()
        src = torch.full_like(self.src, float("nan"))
        value = m(self.features, self.rows, self.cols, src, self.indices)
        self.assertTrue(torch.isfinite(value).all())
        empty = m(self.features[:0], self.rows, self.cols, src, tuple(i[:0] for i in self.indices))
        self.assertEqual(empty.shape, (0, 24))

    def test_axis_balanced_warm_start_and_empty_axis_gradients(self):
        legacy = self.module().eval()
        balanced = CellQueryDecoder(16, 24, width=16, heads=4, layers=2,
                                   chunk_size=5, axis_balanced=True).eval()
        missing, unexpected = balanced.load_state_dict(legacy.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all("axis_gate" in k or "axis_blend" in k for k in missing))
        expected = legacy(self.features, self.rows, self.cols, self.src, self.indices)
        actual = balanced(self.features, self.rows, self.cols, self.src, self.indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for n, p in balanced.named_parameters()
                               if "axis_blend" in n), 0.)
        for layer in balanced.layers:
            layer.axis_blend.data.fill_(.5)
        for source in (self.src, torch.full_like(self.src, float("nan"))):
            balanced.zero_grad(set_to_none=True)
            value = balanced(self.features, self.rows, self.cols, source, self.indices)
            value.square().mean().backward()
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in balanced.parameters()))

    def test_axis_fusion_does_not_weight_by_axis_degree(self):
        layer = CellQueryLayer(16, 4, axis_balanced=True).eval()
        layer.axis_blend.data.fill_(20.)
        query, row, col, shared = (torch.randn(2, n, 16) for n in (1, 1, 1, 2))
        a = torch.cat((row, col, shared), 1)
        b = torch.cat((row, col.expand(-1, 30, -1), shared), 1)
        x = layer(query[:, 0], a, torch.ones(2, 4, dtype=torch.bool), 1)
        y = layer(query[:, 0], b, torch.ones(2, 33, dtype=torch.bool), 1)
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_axis_permutation_equivariance(self):
        m = self.module().eval()
        pr, pc = torch.randperm(7), torch.randperm(6)
        b, r, c = self.indices
        expected = m(self.features, self.rows, self.cols, self.src, self.indices)
        actual = m(self.features, self.rows[:, pr], self.cols[:, pc], self.src[:, pr][:, :, pc],
                   (b, pr.argsort()[r], pc.argsort()[c]))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_observed_value_gradient_and_missing_mask(self):
        m = self.module().eval()
        src = self.src.clone().requires_grad_()
        m(self.features, self.rows, self.cols, src, self.indices).square().mean().backward()
        self.assertTrue(torch.isfinite(src.grad).all())
        self.assertEqual(src.grad[self.src.isnan()].abs().sum().item(), 0)
        self.assertGreater(src.grad[torch.isfinite(self.src)].abs().sum().item(), 0)

    def test_integrated_zero_init_and_grid_pair_consistency(self):
        torch.manual_seed(15)
        old = tiny_model(use_svt_feature_decoder=True).eval()
        torch.manual_seed(15)
        new = tiny_model(use_svt_feature_decoder=True, use_cell_query_decoder=True,
                         cell_query_width=16, cell_query_chunk_size=5).eval()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        mask = self.src.isnan()
        a = old(self.src, decode_mask=mask)
        b = new(self.src, decode_mask=mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.nn.init.normal_(new.cell_query_decoder.output.weight, std=.02)
        paired = new(self.src, decode_mask=mask)
        grid = new(self.src)
        torch.testing.assert_close(paired, grid[mask], atol=3e-5, rtol=3e-4)

    def test_legacy_value_bias_chunking_preserves_values_and_gradients(self):
        model = tiny_model(use_svt_feature_decoder=True, use_observed_attention_decoder=True,
                           observed_attention_dim=8, observed_attention_hidden_size=16).train()
        values = torch.randn(1, 65540, 4)
        reference = model.observed_attention_value_bias(values).squeeze(-1)
        actual = model._observed_attention_bias(values)
        torch.testing.assert_close(actual, reference)
        parameters = list(model.observed_attention_value_bias.parameters())
        expected = torch.autograd.grad(reference.square().mean(), parameters)
        found = torch.autograd.grad(actual.square().mean(), parameters)
        for a, b in zip(expected, found):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
