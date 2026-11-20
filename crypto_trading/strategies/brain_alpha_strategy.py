"""FASTEXPR-to-Crypto Time-Series Translator & Dynamic Alpha Ensemble Strategy.

Translates verified / discovered WorldQuant BRAIN FASTEXPR price-volume alphas
from `worldquant_brain/alpha_vault.db` directly into strictly causal Pandas
time-series signals on Crypto OHLCV bars, and blends them with Advanced Technical
Indicators & Conditional Probability hysteresis so BRAIN alphas actively power
crypto trading.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from worldquant_brain.alpha_vault import AlphaVault


def _ts_rank_series(s: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    return s.rolling(w, min_periods=2).rank(pct=True).fillna(0.5)


def _ts_decay_linear_series(s: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    weights = np.arange(1, w + 1, dtype=np.float64)
    weights /= weights.sum()
    return s.rolling(w, min_periods=1).apply(
        lambda x: float(np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()), raw=True
    )


def evaluate_fastexpr_on_ohlcv(expression: str, df: pd.DataFrame) -> pd.Series:
    """Safely evaluate a price-volume WorldQuant FASTEXPR formula causally on a single-asset OHLCV DataFrame."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    open_ = pd.to_numeric(df["Open"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan).ffill().fillna(1.0)
    typical = (high + low + close) / 3.0
    vwap = (typical * volume).rolling(24, min_periods=1).sum() / volume.rolling(24, min_periods=1).sum()
    returns = close.pct_change().fillna(0.0)

    # Skip pure equity-fundamental / option6 / sentiment expressions that lack OHLCV equivalents
    unsupported_tokens = (
        "operating_", "sales", "assets", "liabilities", "debt", "equity", "ebitda",
        "target_price", "eps_consensus", "opt6_", "scl12_",
    )
    expr_clean = (expression or "").strip()
    if not expr_clean or any(tok in expr_clean for tok in unsupported_tokens):
        # Fallback to canonical price-volume reversal-momentum blend
        expr_clean = "ts_decay_linear(rank(close - open) / (ts_std_dev(close, 20) + 0.0001), 5)"

    # Strip cross-sectional group wrappers: group_neutralize(X, subindustry) -> X
    for grp in ("subindustry", "industry", "sector", "market"):
        expr_clean = re.sub(rf"group_neutralize\s*\(\s*(.+?)\s*,\s*{grp}\s*\)", r"(\1)", expr_clean)
        expr_clean = re.sub(rf"group_rank\s*\(\s*(.+?)\s*,\s*{grp}\s*\)", r"rank(\1)", expr_clean)

    def rank(x: Any, rate: int = 2) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return _ts_rank_series(s, 48)

    def ts_rank(x: Any, d: int = 20, constant: float = 0.0) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return _ts_rank_series(s, d) + float(constant)

    def ts_delta(x: Any, d: int = 5) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.diff(max(1, int(d))).fillna(0.0)

    def ts_delay(x: Any, d: int = 1) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.shift(max(1, int(d))).bfill()

    def ts_mean(x: Any, d: int = 10) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.rolling(max(2, int(d)), min_periods=1).mean()

    def ts_std_dev(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.rolling(max(2, int(d)), min_periods=2).std(ddof=1).fillna(0.0)

    def ts_min(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.rolling(max(2, int(d)), min_periods=1).min()

    def ts_max(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.rolling(max(2, int(d)), min_periods=1).max()

    def ts_av_diff(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s - ts_mean(s, d)

    def ts_zscore(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        sd = ts_std_dev(s, d).replace(0, np.nan)
        return ((s - ts_mean(s, d)) / sd).fillna(0.0)

    def ts_corr(x: Any, y: Any, d: int = 20) -> pd.Series:
        sx = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        sy = y if isinstance(y, pd.Series) else pd.Series(y, index=df.index, dtype=float)
        return sx.rolling(max(3, int(d)), min_periods=3).corr(sy).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def ts_decay_linear(x: Any, d: int = 5, dense: bool = False) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return _ts_decay_linear_series(s.fillna(0.0), d)

    def ts_backfill(x: Any, lookback: int = 5, k: int = 1) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.ffill(limit=max(1, int(lookback))).fillna(0.0)

    def ts_regression(y: Any, x: Any, d: int = 20, lag: int = 0, rettype: int = 0) -> pd.Series:
        sy = y if isinstance(y, pd.Series) else pd.Series(y, index=df.index, dtype=float)
        sx = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        if lag > 0:
            sx = sx.shift(int(lag))
        cov = sy.rolling(max(3, int(d)), min_periods=3).cov(sx)
        var = sx.rolling(max(3, int(d)), min_periods=3).var().replace(0, np.nan)
        beta = (cov / var).fillna(0.0)
        alpha = ts_mean(sy, d) - beta * ts_mean(sx, d)
        resid = sy - (alpha + beta * sx)
        return resid.fillna(0.0) if rettype == 0 else beta

    def signed_power(x: Any, p: float = 2.0) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return np.sign(s) * (np.abs(s) ** float(p))

    def winsorize(x: Any, std: float = 4.0) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        m = s.rolling(48, min_periods=5).mean()
        sd = s.rolling(48, min_periods=5).std(ddof=1).fillna(1.0)
        return s.clip(lower=m - float(std) * sd, upper=m + float(std) * sd).fillna(0.0)

    def hump(x: Any, hump: float = 0.01) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.ewm(span=5, adjust=False).mean()

    def ts_ir(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        sd = ts_std_dev(s, d).replace(0, np.nan)
        return (ts_mean(s, d) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def ts_skewness(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        return s.rolling(max(3, int(d)), min_periods=3).skew().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def ts_argmax(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        w = max(2, int(d))
        return s.rolling(w, min_periods=1).apply(lambda arr: float(np.argmax(arr)), raw=True).fillna(0.0)

    def ts_argmin(x: Any, d: int = 20) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        w = max(2, int(d))
        return s.rolling(w, min_periods=1).apply(lambda arr: float(np.argmin(arr)), raw=True).fillna(0.0)

    def trade_when(cond: Any, alpha_val: Any, exit_val: Any = -1) -> pd.Series:
        c = cond if isinstance(cond, pd.Series) else pd.Series(cond, index=df.index)
        a = alpha_val if isinstance(alpha_val, pd.Series) else pd.Series(alpha_val, index=df.index, dtype=float)
        mask = c.astype(bool)
        if isinstance(exit_val, (int, float)) and exit_val <= 0:
            return a.where(mask).ffill().fillna(0.0)
        e = exit_val if isinstance(exit_val, pd.Series) else pd.Series(exit_val, index=df.index, dtype=float)
        return a.where(mask, e)

    def scale(x: Any, a: float = 1.0) -> pd.Series:
        s = x if isinstance(x, pd.Series) else pd.Series(x, index=df.index, dtype=float)
        denom = s.abs().rolling(48, min_periods=1).mean().replace(0, np.nan)
        return ((s / denom) * float(a)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    env = {
        "__builtins__": {},
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
        "returns": returns,
        "cap": close * volume,
        "abs": np.abs,
        "log": lambda x: np.log(np.maximum(x, 1e-9)),
        "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
        "sign": np.sign,
        "max": np.maximum,
        "min": np.minimum,
        "rank": rank,
        "ts_rank": ts_rank,
        "ts_delta": ts_delta,
        "ts_delay": ts_delay,
        "ts_mean": ts_mean,
        "ts_std_dev": ts_std_dev,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "ts_av_diff": ts_av_diff,
        "ts_zscore": ts_zscore,
        "ts_corr": ts_corr,
        "ts_correlation": ts_corr,
        "ts_decay_linear": ts_decay_linear,
        "ts_backfill": ts_backfill,
        "ts_regression": ts_regression,
        "ts_ir": ts_ir,
        "ts_skewness": ts_skewness,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
        "trade_when": trade_when,
        "scale": scale,
        "signed_power": signed_power,
        "winsorize": winsorize,
        "hump": hump,
    }
    try:
        from worldquant_brain.safe_expr_eval import safe_eval_expr
        raw = safe_eval_expr(expr_clean, env)
        ser = raw if isinstance(raw, pd.Series) else pd.Series(raw, index=df.index, dtype=float)
        ser = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Standardize causally into [-1, 1] via rolling z-score tanh
        rm = ser.rolling(48, min_periods=5).mean()
        rsd = ser.rolling(48, min_periods=5).std(ddof=1).replace(0, np.nan)
        z = ((ser - rm) / rsd).fillna(0.0)
        return pd.Series(np.tanh(z * 0.75), index=df.index)
    except Exception:
        fallback = (close - open_) / (high - low + 1e-4)
        return pd.Series(np.tanh(fallback), index=df.index)


class BrainAlphaCryptoStrategy(BaseStrategy):
    """將 WorldQuant BRAIN Alpha 庫中的高分因子轉譯為加密貨幣多因子共振 + 條件機率操盤策略。"""

    def __init__(
        self,
        vault_db_path: str | None = None,
        max_alphas: int = 8,
        entry_threshold: float = 0.22,
        exit_threshold: float = 0.06,
    ) -> None:
        self.vault = AlphaVault(db_path=vault_db_path)
        self.max_alphas = max(1, int(max_alphas))
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.cond_prob = ConditionalProbTechStrategy()
        self.active_alphas_summary: list[dict[str, Any]] = []

    def _select_active_alphas(self) -> list[dict[str, Any]]:
        rows = self.vault.list_alphas()
        if not rows:
            self.vault.seed_initial_verified_alphas()
            rows = self.vault.list_alphas()
        # Prioritize official BRAIN_COMPLETE / SUBMITTED / VERIFIED alphas sorted by Sharpe * Fitness
        scored = sorted(
            rows,
            key=lambda a: (
                1 if str(a.get("status", "")).upper() in {"BRAIN_COMPLETE", "SUBMITTED", "IS_CHECK_PASSED"} else 0,
                float(a.get("sharpe") or 0.0) * 0.6 + float(a.get("fitness") or 0.0) * 0.4,
            ),
            reverse=True,
        )
        return scored[: self.max_alphas]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.cond_prob.generate_signals(df)
        alphas = self._select_active_alphas()
        self.active_alphas_summary = [
            {"name": a.get("name"), "status": a.get("status"), "sharpe": a.get("sharpe"), "fitness": a.get("fitness")}
            for a in alphas
        ]

        if not alphas or len(out) < 10:
            out["brain_alpha_score"] = 0.0
            return out

        weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
        total_weight = 0.0
        for a in alphas:
            w = max(0.2, float(a.get("sharpe") or 1.0) * 0.6 + float(a.get("fitness") or 1.0) * 0.4)
            sig_ser = evaluate_fastexpr_on_ohlcv(str(a.get("expression") or ""), out)
            weighted_sum = weighted_sum + w * sig_ser
            total_weight += w

        alpha_score = (weighted_sum / max(total_weight, 1e-9)).fillna(0.0)
        out["brain_alpha_score"] = alpha_score

        # Combine BRAIN alpha ensemble score with Conditional Probability + Hysteresis
        cond_sig = out["signal"].to_numpy(dtype=float)
        score_arr = alpha_score.to_numpy(dtype=float)
        final_signals = np.zeros(len(out), dtype=float)
        pos = 0.0

        for i in range(len(out)):
            s = score_arr[i]
            cp = cond_sig[i]
            # Strong BRAIN alpha consensus OR agreement with Conditional Probability triggers entry
            want_long = (s >= self.entry_threshold and cp >= 0) or (s >= 0.10 and cp > 0)
            want_short = (s <= -self.entry_threshold and cp <= 0) or (s <= -0.10 and cp < 0)

            if pos == 0.0:
                if want_long and not want_short:
                    pos = 1.0
                elif want_short and not want_long:
                    pos = -1.0
            elif pos == 1.0:
                if want_short:
                    pos = -1.0
                elif s < self.exit_threshold and cp <= 0:
                    pos = 0.0
            elif pos == -1.0:
                if want_long:
                    pos = 1.0
                elif s > -self.exit_threshold and cp >= 0:
                    pos = 0.0
            final_signals[i] = pos

        out["signal"] = final_signals
        return out
