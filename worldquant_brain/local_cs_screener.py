"""Real local cross-sectional FASTEXPR pre-screener for WorldQuant BRAIN price-volume alphas.

Architecture:
- Price-volume expressions (close, open, high, low, volume, vwap, returns) are evaluated on a real
  multi-stock daily OHLCV panel (loaded from local cache or fetched once via yfinance into cache).
- Computes cross-sectional market-neutral daily PnL, annualized Sharpe, daily Turnover, and pairwise
  PnL correlation across accepted candidates so weak/redundant formulas are filtered locally before
  consuming WorldQuant BRAIN simulation quota.
- Non-PV expressions (fundamental / analyst / sentiment / options fields) bypass local PV screening
  and pass straight through to BRAIN.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PV_FIELDS = {"close", "open", "high", "low", "volume", "vwap", "returns"}
NON_PV_MARKERS = (
    "operating_", "sales", "assets", "liabilities", "debt", "equity", "ebitda",
    "cap", "target_price", "eps_consensus", "opt6_", "scl12_", "fnd", "anl", "news",
)

DEFAULT_US_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "AVGO", "COST", "ABBV",
    "MRK", "PEP", "KO", "ADBE", "CRM", "AMD", "NFLX", "TMO", "ACN", "CSCO",
    "MCD", "ABT", "DHR", "LIN", "TXN", "NEE", "PM", "WFC", "DIS", "VZ",
    "INTC", "QCOM", "IBM", "AMGN", "CAT", "GE", "GS", "LOW", "SPGI", "UNP",
)


@dataclass(frozen=True, slots=True)
class LocalScreenResult:
    expression: str
    is_pv_only: bool
    passed_local_gate: bool
    local_sharpe: float | None
    local_turnover: float | None
    local_fitness: float | None
    max_peer_corr: float | None
    reason: str


def is_price_volume_only(expression: str) -> bool:
    expr = (expression or "").strip().lower()
    if not expr:
        return False
    return not any(tok in expr for tok in NON_PV_MARKERS)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def _ts_rank(df: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    w = max(2, int(d))
    return df.rolling(w, min_periods=max(2, w // 2)).rank(pct=True)


def _ts_decay_linear(df: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    w = max(2, int(d))
    weights = np.arange(1, w + 1, dtype=np.float64)
    weights /= weights.sum()
    arr = df.to_numpy(dtype=np.float64)
    out = np.full_like(arr, np.nan)
    n_rows = arr.shape[0]
    if n_rows < w:
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    for i in range(w - 1, n_rows):
        window = arr[i - w + 1 : i + 1]
        valid = np.isfinite(window)
        w_mat = np.where(valid, weights[:, None], 0.0)
        w_sum = w_mat.sum(axis=0)
        num = np.where(valid, window * weights[:, None], 0.0).sum(axis=0)
        out[i] = np.where(w_sum > 0, num / np.where(w_sum > 0, w_sum, 1.0), np.nan)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def evaluate_fastexpr_panel(expression: str, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Safely evaluate a price-volume FASTEXPR on a 2D (date x symbol) OHLCV panel."""
    close = panel["close"]
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"].replace(0, np.nan).ffill().fillna(1.0)
    vwap = panel.get("vwap")
    if vwap is None:
        vwap = (high + low + close) / 3.0
    returns = panel.get("returns")
    if returns is None:
        returns = close.pct_change().fillna(0.0)

    expr = (expression or "").strip()
    for grp in ("subindustry", "industry", "sector", "market"):
        expr = re.sub(rf"group_neutralize\s*\(\s*(.+?)\s*,\s*{grp}\s*\)", r"neutralize(\1)", expr)
        expr = re.sub(rf"group_rank\s*\(\s*(.+?)\s*,\s*{grp}\s*\)", r"rank(\1)", expr)

    def _as_df(x: Any) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame(float(x), index=close.index, columns=close.columns)

    def neutralize(x: Any) -> pd.DataFrame:
        df = _as_df(x)
        return df.sub(df.mean(axis=1), axis=0)

    def rank(x: Any, rate: int = 2) -> pd.DataFrame:
        return _cs_rank(_as_df(x))

    def winsorize(x: Any, std: float = 4.0) -> pd.DataFrame:
        df = _as_df(x)
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        lo = mu - float(std) * sd
        hi = mu + float(std) * sd
        return df.clip(lower=lo, upper=hi, axis=0)

    def hump(x: Any, hump: float = 0.01) -> pd.DataFrame:
        return _as_df(x).ewm(span=4, adjust=False).mean()

    def ts_rank(x: Any, d: int = 20, constant: float = 0.0) -> pd.DataFrame:
        return _ts_rank(_as_df(x), d) + float(constant)

    def ts_delta(x: Any, d: int = 5) -> pd.DataFrame:
        return _as_df(x).diff(max(1, int(d)))

    def ts_delay(x: Any, d: int = 1) -> pd.DataFrame:
        return _as_df(x).shift(max(1, int(d)))

    def ts_mean(x: Any, d: int = 10) -> pd.DataFrame:
        w = max(2, int(d))
        return _as_df(x).rolling(w, min_periods=max(1, w // 2)).mean()

    def ts_std_dev(x: Any, d: int = 20) -> pd.DataFrame:
        w = max(2, int(d))
        return _as_df(x).rolling(w, min_periods=max(2, w // 2)).std(ddof=1)

    def ts_min(x: Any, d: int = 20) -> pd.DataFrame:
        w = max(2, int(d))
        return _as_df(x).rolling(w, min_periods=1).min()

    def ts_max(x: Any, d: int = 20) -> pd.DataFrame:
        w = max(2, int(d))
        return _as_df(x).rolling(w, min_periods=1).max()

    def ts_av_diff(x: Any, d: int = 20) -> pd.DataFrame:
        df = _as_df(x)
        return df - ts_mean(df, d)

    def ts_zscore(x: Any, d: int = 20) -> pd.DataFrame:
        df = _as_df(x)
        sd = ts_std_dev(df, d).replace(0, np.nan)
        return (df - ts_mean(df, d)) / sd

    def ts_ir(x: Any, d: int = 20) -> pd.DataFrame:
        df = _as_df(x)
        sd = ts_std_dev(df, d).replace(0, np.nan)
        return ts_mean(df, d) / sd

    def ts_skewness(x: Any, d: int = 20) -> pd.DataFrame:
        w = max(3, int(d))
        return _as_df(x).rolling(w, min_periods=3).skew()

    def ts_corr(x: Any, y: Any, d: int = 20) -> pd.DataFrame:
        w = max(3, int(d))
        return _as_df(x).rolling(w, min_periods=3).corr(_as_df(y))

    def ts_decay_linear(x: Any, d: int = 5, dense: bool = False) -> pd.DataFrame:
        return _ts_decay_linear(_as_df(x), d)

    def ts_backfill(x: Any, lookback: int = 5, k: int = 1) -> pd.DataFrame:
        return _as_df(x).ffill(limit=max(1, int(lookback)))

    def ts_regression(y: Any, x: Any, d: int = 20, lag: int = 0, rettype: int = 0) -> pd.DataFrame:
        sy = _as_df(y)
        sx = _as_df(x)
        if lag > 0:
            sx = sx.shift(int(lag))
        w = max(3, int(d))
        cov = sy.rolling(w, min_periods=3).cov(sx)
        var_x = sx.rolling(w, min_periods=3).var().replace(0, np.nan)
        beta = (cov / var_x).fillna(0.0)
        alpha = ts_mean(sy, w) - beta * ts_mean(sx, w)
        yhat = alpha + beta * sx
        rt = int(rettype)
        if rt == 1:
            return alpha
        if rt == 2:
            return beta
        if rt == 3:
            return yhat
        if rt == 6:
            var_y = sy.rolling(w, min_periods=3).var().replace(0, np.nan)
            return ((cov ** 2) / (var_x * var_y)).clip(0.0, 1.0).fillna(0.0)
        return sy - yhat

    def signed_power(x: Any, p: float = 2.0) -> pd.DataFrame:
        df = _as_df(x)
        return np.sign(df) * (df.abs() ** float(p))

    def trade_when(cond: Any, alpha_expr: Any, exit_cond: Any = -1) -> pd.DataFrame:
        c = _as_df(cond).fillna(0).astype(bool)
        a = _as_df(alpha_expr)
        if isinstance(exit_cond, (int, float)) and exit_cond <= 0:
            return a.where(c).ffill()
        ex = _as_df(exit_cond).fillna(0).astype(bool)
        sentinel = np.inf
        marked = a.where(c).mask(ex, sentinel).ffill()
        return marked.replace(sentinel, np.nan)

    env: dict[str, Any] = {
        "__builtins__": {},
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
        "returns": returns,
        "abs": lambda x: _as_df(x).abs(),
        "log": lambda x: np.log(_as_df(x).clip(lower=1e-9)),
        "sign": lambda x: np.sign(_as_df(x)),
        "neutralize": neutralize,
        "rank": rank,
        "winsorize": winsorize,
        "hump": hump,
        "ts_rank": ts_rank,
        "ts_delta": ts_delta,
        "ts_delay": ts_delay,
        "ts_mean": ts_mean,
        "ts_std_dev": ts_std_dev,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "ts_av_diff": ts_av_diff,
        "ts_zscore": ts_zscore,
        "ts_ir": ts_ir,
        "ts_skewness": ts_skewness,
        "ts_corr": ts_corr,
        "ts_decay_linear": ts_decay_linear,
        "ts_backfill": ts_backfill,
        "ts_regression": ts_regression,
        "signed_power": signed_power,
        "trade_when": trade_when,
    }
    from worldquant_brain.safe_expr_eval import safe_eval_expr
    res = safe_eval_expr(expr, env)
    return _as_df(res).replace([np.inf, -np.inf], np.nan)


class LocalCrossSectionalScreener:
    """Fast local multi-stock cross-sectional pre-screener for PV alphas."""

    def __init__(
        self,
        panel: Mapping[str, pd.DataFrame] | None = None,
        *,
        cache_path: str | Path | None = None,
        min_local_sharpe: float = 0.65,
        max_local_turnover: float = 0.85,
        max_peer_corr: float = 0.65,
        max_workers: int = 4,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else (Path(__file__).resolve().parent / "local_us_panel_cache.pkl")
        self.min_local_sharpe = float(min_local_sharpe)
        self.max_local_turnover = float(max_local_turnover)
        self.max_peer_corr = float(max_peer_corr)
        self.max_workers = max(1, int(max_workers))
        self._panel: dict[str, pd.DataFrame] | None = dict(panel) if panel is not None else None
        self._accepted_pnl_history: list[pd.Series] = []

    def has_panel(self) -> bool:
        return self._panel is not None or self.cache_path.exists()

    def load_or_fetch_panel(self, *, allow_download: bool = False, period: str = "3y") -> dict[str, pd.DataFrame] | None:
        if self._panel is not None:
            return self._panel
        if self.cache_path.exists():
            try:
                loaded = pd.read_pickle(self.cache_path)
                if isinstance(loaded, dict) and "close" in loaded and isinstance(loaded["close"], pd.DataFrame):
                    self._panel = loaded
                    return self._panel
            except Exception:
                pass
        if not allow_download:
            return None
        try:
            import yfinance as yf
            raw = yf.download(
                list(DEFAULT_US_TICKERS),
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            if raw is None or raw.empty:
                return None
            panel = {
                "close": raw["Close"].dropna(axis=1, how="all").ffill(),
                "open": raw["Open"].dropna(axis=1, how="all").ffill(),
                "high": raw["High"].dropna(axis=1, how="all").ffill(),
                "low": raw["Low"].dropna(axis=1, how="all").ffill(),
                "volume": raw["Volume"].dropna(axis=1, how="all").fillna(0.0),
            }
            panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
            panel["returns"] = panel["close"].pct_change().fillna(0.0)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(panel, self.cache_path)
            self._panel = panel
            return self._panel
        except Exception:
            return None

    def _evaluate_one(self, expression: str, panel: Mapping[str, pd.DataFrame]) -> tuple[LocalScreenResult, pd.Series | None]:
        if not is_price_volume_only(expression):
            return (
                LocalScreenResult(expression, False, True, None, None, None, None, "NON_PV_DIRECT_PASS"),
                None,
            )
        try:
            alpha = evaluate_fastexpr_panel(expression, panel)
            # Market-neutral cross-sectional weights, L1-normalized to 1.0, strictly delayed by 1 day (delay=1)
            centered = alpha.sub(alpha.mean(axis=1), axis=0)
            denom = centered.abs().sum(axis=1).replace(0, np.nan)
            weights = centered.div(denom, axis=0).clip(lower=-0.10, upper=0.10)
            w_prev = weights.shift(1)

            fwd_ret = panel["returns"]
            daily_pnl = (w_prev * fwd_ret).sum(axis=1, min_count=1).dropna()
            if len(daily_pnl) < 40:
                return (
                    LocalScreenResult(expression, True, False, 0.0, 1.0, 0.0, None, "INSUFFICIENT_LOCAL_BARS"),
                    None,
                )

            mu = float(daily_pnl.mean())
            sd = float(daily_pnl.std(ddof=1))
            sharpe = (mu / sd) * np.sqrt(252.0) if sd > 1e-12 else 0.0
            turnover = float(weights.diff().abs().sum(axis=1).mean())
            ann_ret = mu * 252.0
            fitness = sharpe * np.sqrt(abs(ann_ret) / max(turnover, 0.125))

            max_corr = 0.0
            for prev_pnl in self._accepted_pnl_history:
                aligned = pd.concat([daily_pnl, prev_pnl], axis=1).dropna()
                if len(aligned) >= 30:
                    c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if np.isfinite(c):
                        max_corr = max(max_corr, abs(c))

            # Local Sub-Universe Sharpe check (top 33% most liquid stocks by dollar volume)
            sub_ok = True
            sub_reason = ""
            try:
                dol_vol = (panel["volume"] * panel["close"]).shift(1)
                is_liquid = dol_vol.rank(axis=1, pct=True) >= 0.67
                sub_alpha = alpha.where(is_liquid)
                sub_centered = sub_alpha.sub(sub_alpha.mean(axis=1), axis=0)
                sub_denom = sub_centered.abs().sum(axis=1).replace(0, np.nan)
                sub_w = sub_centered.div(sub_denom, axis=0).clip(lower=-0.10, upper=0.10).shift(1)
                sub_pnl = (sub_w * fwd_ret).sum(axis=1, min_count=1).dropna()
                if len(sub_pnl) < 40 or float(sub_pnl.std(ddof=1)) <= 1e-12:
                    sub_ok = False
                    sub_reason = "SUB_UNIVERSE_SHARPE_NAN"
                else:
                    sub_sharpe = (float(sub_pnl.mean()) / float(sub_pnl.std(ddof=1))) * np.sqrt(252.0)
                    cutoff = 0.75 * np.sqrt(1.0 / 3.0) * sharpe
                    if not np.isfinite(sub_sharpe) or (sharpe > 0 and sub_sharpe < cutoff):
                        sub_ok = False
                        sub_reason = f"LOW_SUB_UNIVERSE_SHARPE:{sub_sharpe:.2f}<{cutoff:.2f}"
            except Exception:
                sub_ok = True

            passed = (
                sharpe >= self.min_local_sharpe
                and turnover <= self.max_local_turnover
                and max_corr < self.max_peer_corr
                and sub_ok
            )
            reason = "LOCAL_PV_PASSED"
            if sharpe < self.min_local_sharpe:
                reason = f"LOW_LOCAL_SHARPE:{sharpe:.2f}"
            elif turnover > self.max_local_turnover:
                reason = f"HIGH_LOCAL_TURNOVER:{turnover:.2f}"
            elif max_corr >= self.max_peer_corr:
                reason = f"HIGH_LOCAL_PEER_CORR:{max_corr:.2f}"
            elif not sub_ok:
                reason = sub_reason

            return (
                LocalScreenResult(
                    expression=expression,
                    is_pv_only=True,
                    passed_local_gate=passed,
                    local_sharpe=round(float(sharpe), 3),
                    local_turnover=round(float(turnover), 3),
                    local_fitness=round(float(fitness), 3),
                    max_peer_corr=round(float(max_corr), 3),
                    reason=reason,
                ),
                daily_pnl,
            )
        except Exception as exc:
            return (
                LocalScreenResult(expression, True, False, None, None, None, None, f"LOCAL_EVAL_ERROR:{type(exc).__name__}"),
                None,
            )

    def screen_batch(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 5,
        allow_download: bool = False,
    ) -> list[dict[str, Any]]:
        """Screen a batch of candidates in parallel; non-PV pass through, PV are ranked by local Sharpe/fitness."""
        panel = self.load_or_fetch_panel(allow_download=allow_download)
        if panel is None:
            # No local panel cached yet -> pass through without blocking mining
            return [dict(c) for c in candidates[:top_k]]

        exprs = [str(c.get("expression") or "") for c in candidates]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            evaluated = list(pool.map(lambda e: self._evaluate_one(e, panel), exprs))

        non_pv: list[dict[str, Any]] = []
        pv_passed: list[tuple[float, dict[str, Any], pd.Series | None]] = []

        for cand, (res, pnl) in zip(candidates, evaluated):
            enriched = dict(cand)
            enriched["local_screen"] = {
                "is_pv_only": res.is_pv_only,
                "passed": res.passed_local_gate,
                "local_sharpe": res.local_sharpe,
                "local_turnover": res.local_turnover,
                "local_fitness": res.local_fitness,
                "max_peer_corr": res.max_peer_corr,
                "reason": res.reason,
            }
            if not res.is_pv_only:
                non_pv.append(enriched)
            elif res.passed_local_gate:
                score = float(res.local_sharpe or 0.0) + 0.35 * float(res.local_fitness or 0.0)
                pv_passed.append((score, enriched, pnl))

        pv_passed.sort(key=lambda x: x[0], reverse=True)
        selected: list[dict[str, Any]] = []
        batch_pnls: list[pd.Series] = []
        for _sc, item, pnl in pv_passed:
            if len(selected) >= top_k:
                break
            if pnl is not None:
                too_corr = False
                for bp in batch_pnls:
                    aligned = pd.concat([pnl, bp], axis=1).dropna()
                    if len(aligned) >= 30:
                        c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                        if np.isfinite(c) and abs(c) >= self.max_peer_corr:
                            too_corr = True
                            break
                if too_corr:
                    continue
                batch_pnls.append(pnl)
                self._accepted_pnl_history.append(pnl)
                if len(self._accepted_pnl_history) > 100:
                    self._accepted_pnl_history.pop(0)
            selected.append(item)

        for item in non_pv:
            if len(selected) >= top_k:
                break
            selected.append(item)

        return selected
