import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.loss_functions import (
    BayesianConditionalLoss,
    LatentRepresentationLoss,
    SoftRankICLoss,
    SoftSharpeLoss,
    numpy_rank_ic,
    numpy_sharpe_ratio,
)


class TestCustomLossFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_soft_sharpe_loss(self):
        loss_fn = SoftSharpeLoss(annualization_factor=252.0)

        # Perfect long positions with positive returns
        positions = torch.ones(100, requires_grad=True)
        returns = torch.normal(0.01, 0.005, size=(100,))

        loss = loss_fn(positions, returns)
        # Loss is -Sharpe, so it should be negative (Sharpe > 0)
        self.assertLess(loss.item(), 0.0)

        # Check backward pass
        loss.backward()
        self.assertIsNotNone(positions.grad)
        self.assertEqual(positions.grad.shape, positions.shape)

        # Compare with numpy helper
        py_sharpe = -loss.item()
        np_sharpe = numpy_sharpe_ratio(
            positions.detach().numpy(), returns.numpy(), annualization_factor=252.0
        )
        self.assertAlmostEqual(py_sharpe, np_sharpe, places=3)

    def test_soft_rank_ic_loss(self):
        loss_fn = SoftRankICLoss(temperature=0.1)

        # Perfectly correlated vectors
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

        loss = loss_fn(x, y)
        # Soft rank IC should be close to -1.0 (since loss is -RankIC)
        self.assertLess(loss.item(), -0.95)

        loss.backward()
        self.assertIsNotNone(x.grad)

        # Inversely correlated vectors
        x_inv = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], requires_grad=True)
        loss_inv = loss_fn(x_inv, y)
        # Loss should be positive (RankIC is negative)
        self.assertGreater(loss_inv.item(), 0.95)

        # Check numpy rank IC
        np_ic = numpy_rank_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        self.assertAlmostEqual(np_ic, 1.0, places=4)

    def test_bayesian_conditional_loss(self):
        loss_fn = BayesianConditionalLoss(gamma=2.0, alpha=0.5, vol_scale=1.0)

        logits = torch.randn(50, requires_grad=True)
        targets = torch.randint(0, 2, size=(50,)).float()
        regime_vol = torch.ones(50) * 1.5

        loss = loss_fn(logits, targets, regime_volatility=regime_vol)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_latent_representation_loss(self):
        loss_fn = LatentRepresentationLoss(beta_kl=0.01, lambda_dir=0.5)

        x = torch.randn(16, 32)
        recon_x = torch.randn(16, 32, requires_grad=True)
        mu = torch.zeros(16, 8, requires_grad=True)
        logvar = torch.zeros(16, 8, requires_grad=True)
        pred_ret = torch.randn(16, requires_grad=True)
        target_ret = torch.randn(16)

        total_loss, metrics = loss_fn(recon_x, x, mu, logvar, pred_ret, target_ret)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("recon_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertIn("dir_loss", metrics)

        total_loss.backward()
        self.assertIsNotNone(recon_x.grad)
        self.assertIsNotNone(mu.grad)


if __name__ == "__main__":
    unittest.main()
