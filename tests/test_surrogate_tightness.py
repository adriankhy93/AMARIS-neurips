import unittest

import torch

from amaris_merge.analyze_surrogate_tightness import batch_metrics


class SurrogateTightnessTests(unittest.TestCase):
    def test_batch_metrics_matches_explicit_projector(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        q, _ = torch.linalg.qr(torch.randn(4, 2))
        delta = torch.randn(5, 4)
        delta_basis = delta @ q
        sigma_sq = torch.linalg.matrix_norm(delta, ord=2).square().item()

        exact, bound, projected, tokens = batch_metrics(x, q, delta_basis, sigma_sq)
        flat_x = x.reshape(-1, 4)
        projector = q @ q.T
        expected_exact = (
            torch.linalg.matrix_norm(flat_x @ projector @ delta.T).square().item()
        )
        expected_projected = (
            torch.linalg.matrix_norm(flat_x @ projector).square().item()
        )

        self.assertEqual(tokens, 6)
        self.assertAlmostEqual(exact, expected_exact, places=4)
        self.assertAlmostEqual(projected, expected_projected, places=4)
        self.assertAlmostEqual(bound, expected_projected * sigma_sq, places=4)
        self.assertLessEqual(exact, bound + 1e-3)
