"""
Quantitative Finance Custom Loss Functions for PyTorch and NumPy.

Includes:
1. Differentiable Sharpe Ratio Loss (Soft Sharpe Loss)
   - Directly optimizes the annualized Sharpe ratio of continuous positions / portfolio returns.
   - Sharpe = E[R] / (std(R) + eps) * sqrt(annualization_factor)
   - Minimizes: -Sharpe

2. Differentiable Rank IC Loss (Soft Spearman Rank Correlation Loss)
   - Differentiable surrogate for Spearman rank correlation between predictions and forward returns.
   - Uses soft ranking via pairwise sigmoid comparisons:
     rank_i = 1 + sum_{j != i} sigmoid((x_i - x_j) / tau)
   - Computes Pearson correlation between soft ranks.
   - Minimizes: -RankIC

3. Bayesian Conditional Probability Loss (Regime-Switching Focal Loss)
   - Models conditional returns given market regime P(Up | Regime, Features).
   - Incorporates regime volatility-aware modulation and focal gamma penalty:
     L = -alpha_t * (1 - p_t)^gamma * log(p_t)
   - Weights samples dynamically by volatility / regime uncertainty.

4. Latent Representation Loss (Financial VAE Reconstruction + KL + Return Mutual Information)
   - For autoencoders learning compressed latent states of financial time series.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ==============================================================================
# 1. Differentiable Sharpe Ratio Loss (Soft Sharpe Loss)
# ==============================================================================


class SoftSharpeLoss(nn.Module):
    """
    Differentiable Sharpe Ratio Loss in PyTorch.
    Directly maximizes annualized Sharpe Ratio by minimizing its negative.

    Parameters
    ----------
    annualization_factor : float
        Number of periods per year (e.g., 252 for daily, 252*24 for hourly, 252*24*4 for 15m).
        Default is 252.0.
    eps : float
        Small constant for numerical stability in standard deviation. Default is 1e-8.
    risk_free_rate : float
        Period risk-free benchmark return (default 0.0).
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        eps: float = 1e-8,
        risk_free_rate: float = 0.0,
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.eps = eps
        self.risk_free_rate = risk_free_rate

    def forward(self, positions: torch.Tensor, forward_returns: torch.Tensor) -> torch.Tensor:
        """
        Calculates negative Sharpe Ratio of strategy portfolio returns.

        Parameters
        ----------
        positions : torch.Tensor
            Predicted positions or portfolio weights, shape (N,) or (N, 1), continuous in [-1.0, 1.0].
        forward_returns : torch.Tensor
            Realized future asset returns, shape (N,) or (N, 1).

        Returns
        -------
        torch.Tensor
            Scalar loss: -Sharpe Ratio.
        """
        pos = positions.view(-1)
        ret = forward_returns.view(-1)

        # Portfolio period returns
        p_ret = pos * ret - self.risk_free_rate

        mean_ret = torch.mean(p_ret)
        # Unbiased variance calculation
        var_ret = torch.var(p_ret, unbiased=True)
        std_ret = torch.sqrt(var_ret + self.eps)

        sharpe = (mean_ret / std_ret) * np.sqrt(self.annualization_factor)
        return -sharpe


def numpy_sharpe_ratio(
    positions: np.ndarray,
    forward_returns: np.ndarray,
    annualization_factor: float = 252.0,
    eps: float = 1e-8,
) -> float:
    """NumPy evaluation of Sharpe Ratio."""
    pos = np.asarray(positions).flatten()
    ret = np.asarray(forward_returns).flatten()
    p_ret = pos * ret
    mean_ret = np.mean(p_ret)
    var_ret = np.var(p_ret, ddof=1)
    std_ret = np.sqrt(var_ret + eps)
    return float((mean_ret / std_ret) * np.sqrt(annualization_factor))


# ==============================================================================
# 2. Differentiable Rank IC Loss (Soft Spearman Rank Correlation)
# ==============================================================================


class SoftRankICLoss(nn.Module):
    """
    Differentiable Information Coefficient (Rank IC / Soft Spearman) Loss.
    Uses continuous soft ranking via pairwise sigmoid comparisons:
        rank(x_i) = 1 + sum_{j != i} sigmoid((x_i - x_j) / temperature)

    Parameters
    ----------
    temperature : float
        Smoothing parameter tau. Smaller temperature approaches hard ranks,
        larger temperature provides smoother gradients. Default is 0.1.
    eps : float
        Numerical epsilon. Default is 1e-6.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes soft ranks for a 1D tensor of shape (N,).
        Returns soft ranks of shape (N,).
        """
        # x: (N, 1)
        x = x.view(-1, 1)
        # Pairwise diff: diff[i, j] = x[i] - x[j]
        diff = x - x.t()
        # Soft ranking: sigmoid(diff / temperature)
        # sum along dim=1 gives how many elements x[i] is greater than
        ranks = 1.0 + torch.sum(torch.sigmoid(diff / self.temperature), dim=1)
        return ranks

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions (cross-sectional alpha scores), shape (N,) or (N, 1).
        targets : torch.Tensor
            Realized cross-sectional returns, shape (N,) or (N, 1).

        Returns
        -------
        torch.Tensor
            Scalar loss: -Rank IC (-Spearman correlation between predictions and targets).
        """
        pred = predictions.view(-1)
        tgt = targets.view(-1)
        n = pred.size(0)

        if n < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_ranks = self._soft_rank(pred)
        tgt_ranks = self._soft_rank(tgt)

        # Pearson correlation between soft ranks
        pred_mean = torch.mean(pred_ranks)
        tgt_mean = torch.mean(tgt_ranks)

        pred_diff = pred_ranks - pred_mean
        tgt_diff = tgt_ranks - tgt_mean

        cov = torch.sum(pred_diff * tgt_diff)
        pred_norm = torch.sqrt(torch.sum(pred_diff**2) + self.eps)
        tgt_norm = torch.sqrt(torch.sum(tgt_diff**2) + self.eps)

        soft_spearman = cov / (pred_norm * tgt_norm + self.eps)
        return -soft_spearman


def numpy_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """NumPy evaluation of exact Spearman Rank Correlation (Rank IC)."""
    pred = np.asarray(predictions).flatten()
    tgt = np.asarray(targets).flatten()
    if len(pred) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(pred, tgt)
        return float(corr) if not np.isnan(corr) else 0.0
    except ImportError:
        # Fallback ranking using numpy without scipy dependency
        def get_ranks(arr):
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(arr))
            return ranks

        r_pred = get_ranks(pred)
        r_tgt = get_ranks(tgt)
        r_pred = (r_pred - np.mean(r_pred)) / (np.std(r_pred) + 1e-9)
        r_tgt = (r_tgt - np.mean(r_tgt)) / (np.std(r_tgt) + 1e-9)
        corr = np.mean(r_pred * r_tgt)
        return float(corr) if not np.isnan(corr) else 0.0


# ==============================================================================
# 3. Bayesian Conditional Probability Loss (Regime-Switching Focal Loss)
# ==============================================================================


class BayesianConditionalLoss(nn.Module):
    """
    Bayesian Conditional Probability Loss with Regime Modulation & Focal Reweighting.
    Models conditional return probability P(Up | Regime, Features).

    In high-volatility or non-stationary regimes, samples often contain higher noise.
    This loss dynamically reweights prediction errors based on:
    1. Regime volatility factor sigma_regime
    2. Modulating focal factor (1 - p_t)^gamma to focus on hard-to-classify turning points.

    Formula:
        FL(p_t) = - w_regime * alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focal focusing exponent. Default is 2.0.
    alpha : float
        Base class balance factor. Default is 0.5.
    vol_scale : float
        Scaling for regime volatility modulation. Default is 1.0.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.5,
        vol_scale: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.vol_scale = vol_scale
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        regime_volatility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw model output logits (before sigmoid), shape (N,) or (N, 1).
        targets : torch.Tensor
            Binary direction (1 for Up, 0 for Down), shape (N,) or (N, 1).
        regime_volatility : Optional[torch.Tensor]
            Normalized volatility of the detected regime (e.g., 0.5 to 2.5), shape (N,) or (N, 1).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Focal modulating factor
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        # Cross entropy
        bce = -torch.log(p_t)

        # Regime volatility modulation (Bayesian uncertainty weighting)
        if regime_volatility is not None:
            reg_vol = regime_volatility.view(-1).float()
            # In higher regime volatility, downscale noisy spikes or adapt weights
            regime_weight = torch.exp(-self.vol_scale * (reg_vol - 1.0).clamp(min=-1.0, max=2.0))
        else:
            regime_weight = 1.0

        loss = regime_weight * focal_weight * bce
        return torch.mean(loss)


# ==============================================================================
# 4. Latent Space Representation Autoencoder Loss
# ==============================================================================


class LatentRepresentationLoss(nn.Module):
    """
    Financial Time-Series Latent Space Loss for Variational Autoencoder (VAE) / Transformer.
    Combines:
    1. Reconstruction Loss (Smooth L1 Loss on features)
    2. KL Divergence (Latent Regularization against Prior N(0, I))
    3. Directional Alignment Loss (Ensuring latent embedding z predicts forward returns)
    """

    def __init__(self, beta_kl: float = 0.01, lambda_dir: float = 0.5):
        super().__init__()
        self.beta_kl = beta_kl
        self.lambda_dir = lambda_dir

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        pred_ret: torch.Tensor | None = None,
        target_ret: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed time series features.
        x : torch.Tensor
            Original input features.
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log-variance of latent distribution.
        pred_ret : Optional[torch.Tensor]
            Predicted return from latent space.
        target_ret : Optional[torch.Tensor]
            Target future return.
        """
        # 1. Feature reconstruction loss (Smooth L1 / Huber)
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction="mean")

        # 2. KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.beta_kl * kl_loss
        metrics = {
            "recon_loss": float(recon_loss.item()),
            "kl_loss": float(kl_loss.item()),
        }

        # 3. Optional directional loss
        if pred_ret is not None and target_ret is not None:
            dir_loss = F.mse_loss(pred_ret.view(-1), target_ret.view(-1))
            total_loss += self.lambda_dir * dir_loss
            metrics["dir_loss"] = float(dir_loss.item())

        return total_loss, metrics
