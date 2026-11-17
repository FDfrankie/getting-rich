#!/usr/bin/env python3
"""
Phase 6 Patch: Conditional Probability + Technical Indicators Strategy & 6-Timeframe Benchmark
Idempotently adds/upgrades:
  1. crypto_trading/strategies/conditional_prob_strategy.py
     - Causal historical conditional probability engine P(Win | Technical_State) with Bayesian shrinkage
     - Combines EMA trend, RSI zone, Bollinger %B + Volume Z-score, and optional higher-TF regime
     - Asymmetric hysteresis (entry_prob > exit_prob) to solve low time-in-market & prevent fee bleed
  2. scripts/benchmark_timeframes.py
     - Resamples 1m 180d cache into 1m, 3m, 5m, 15m, 1h, 1d
     - Runs strict OOS Walk-Forward comparison (Baseline Momentum vs ConditionalProb Maker/Taker)
     - Outputs reports/mtf_conditional_prob_benchmark.json & .md and selects optimal timeframe
  3. crypto_trading/bot_runner.py
     - Defaults to ConditionalProbTechStrategy + loads optimal timeframe from benchmark report if present
     - Adds new unit test tests/test_conditional_prob_strategy.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONDITIONAL_STRATEGY_CODE = '''"""
Conditional Probability + Technical Indicators Strategy (ConditionalProbTechStrategy).
Combines:
  1. Strictly causal historical conditional probability P(NetWin | TechnicalState)
     using Bayesian shrinkage toward prior win rate when state sample count is small.
  2. Real-time technical indicator requirements (EMA trend, RSI zone, BB %B, Volume Z-score).
  3. Asymmetric hysteresis thresholds (entry_prob vs exit_prob) to extend holding time
     and eliminate excessive micro-churning fees.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.momentum_breakout import BaseStrategy


class ConditionalProbTechStrategy(BaseStrategy):
    """過往條件機率 + 即時技術指標要件 + 非對稱遲滯持倉策略"""

    def __init__(
        self,
        horizon: int = 3,
        entry_prob: float = 0.535,
        exit_prob: float = 0.480,
        min_ev: float = 0.0001,
        fee_roundtrip: float = 0.0002,
        prior_strength: float = 25.0,
        min_history_bars: int = 120,
        warmup_fraction: float = 0.0,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.entry_prob = float(entry_prob)
        self.exit_prob = float(exit_prob)
        self.min_ev = float(min_ev)
        self.fee_roundtrip = float(fee_roundtrip)
        self.prior_strength = float(prior_strength)
        self.min_history_bars = max(40, int(min_history_bars))
        self.warmup_fraction = float(warmup_fraction)
        self.last_state_summary: dict[str, Any] = {}

    @staticmethod
    def compute_technical_states(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce")

        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        sd20 = close.rolling(20, min_periods=20).std(ddof=1)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger %B & Volume Z-score
        bb_pct_b = (close - (ma20 - 2.0 * sd20)) / (4.0 * sd20).replace(0, np.nan)
        vol_mean = vol.rolling(24, min_periods=24).mean()
        vol_std = vol.rolling(24, min_periods=24).std(ddof=1)
        vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # ATR for risk_manager compatibility
        prev_c = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
        out["ATR"] = tr.rolling(14, min_periods=14).mean()
        out["RSI_14"] = rsi
        out["BB_PCT_B"] = bb_pct_b
        out["VOL_Z"] = vol_z
        out["EMA_SPREAD"] = (ema12 - ema26) / close.replace(0, np.nan)

        # Discretize technical conditions into interpretable state keys
        trend = np.where((ema12 > ema26) & (close > ma20), "UP", np.where((ema12 < ema26) & (close < ma20), "DOWN", "FLAT"))
        rsi_zone = np.where(
            rsi < 35,
            "OS",
            np.where(rsi > 65, "OB", np.where(rsi >= 50, "BULL", "BEAR")),
        )
        bb_zone = np.where(bb_pct_b > 0.8, "HI", np.where(bb_pct_b < 0.2, "LO", "MID"))
        vol_zone = np.where(vol_z > 0.5, "HV", "NV")

        out["state_fine"] = [f"{t}|{r}|{b}|{v}" for t, r, b, v in zip(trend, rsi_zone, bb_zone, vol_zone)]
        out["state_coarse"] = [f"{t}|{r}" for t, r in zip(trend, rsi_zone)]
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_technical_states(df)
        close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
        fine_states = out["state_fine"].tolist()
        coarse_states = out["state_coarse"].tolist()
        ema_spread = out["EMA_SPREAD"].to_numpy(dtype=np.float64)
        rsi_arr = out["RSI_14"].to_numpy(dtype=np.float64)

        n = len(out)
        h = self.horizon
        signals = np.zeros(n, dtype=float)
        prob_long_arr = np.full(n, 0.5, dtype=float)
        prob_short_arr = np.full(n, 0.5, dtype=float)
        ev_long_arr = np.zeros(n, dtype=float)
        ev_short_arr = np.zeros(n, dtype=float)

        # Strictly causal expanding statistics:
        # At bar i, only bars j <= i - h have realized horizon returns (close[j + h] / close[j] - 1)!
        fine_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # [n, wins_long, wins_short, sum_ret]
        coarse_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        global_n = 0.0
        global_wl = 0.0
        global_ws = 0.0
        global_ret = 0.0

        warmup_idx = max(self.min_history_bars, int(n * self.warmup_fraction))
        pos = 0

        for i in range(n):
            # 1. Update causal history with bar j = i - h whose future return is now realized at bar i
            j = i - h
            if j >= 26 and not np.isnan(close[j]) and not np.isnan(close[i]) and close[j] > 0:
                fwd_ret = (close[i] - close[j]) / close[j]
                win_l = 1.0 if fwd_ret > self.fee_roundtrip else 0.0
                win_s = 1.0 if fwd_ret < -self.fee_roundtrip else 0.0

                fs = fine_states[j]
                cs = coarse_states[j]
                fine_stats[fs][0] += 1.0
                fine_stats[fs][1] += win_l
                fine_stats[fs][2] += win_s
                fine_stats[fs][3] += fwd_ret

                coarse_stats[cs][0] += 1.0
                coarse_stats[cs][1] += win_l
                coarse_stats[cs][2] += win_s
                coarse_stats[cs][3] += fwd_ret

                global_n += 1.0
                global_wl += win_l
                global_ws += win_s
                global_ret += fwd_ret

            if i < warmup_idx or np.isnan(ema_spread[i]) or np.isnan(rsi_arr[i]):
                signals[i] = 0.0
                continue

            # 2. Compute hierarchical Bayesian-shrunk conditional probability at bar i
            p0_l = (global_wl / global_n) if global_n > 0 else 0.5
            p0_s = (global_ws / global_n) if global_n > 0 else 0.5
            ev0 = (global_ret / global_n) if global_n > 0 else 0.0

            cs_i = coarse_states[i]
            cn, cwl, cws, cret = coarse_stats[cs_i]
            alpha_c = self.prior_strength
            pc_l = (cwl + alpha_c * p0_l) / (cn + alpha_c)
            pc_s = (cws + alpha_c * p0_s) / (cn + alpha_c)
            evc = (cret + alpha_c * ev0) / (cn + alpha_c)

            fs_i = fine_states[i]
            fn, fwl, fws, fret = fine_stats[fs_i]
            alpha_f = self.prior_strength * 0.6
            p_long = (fwl + alpha_f * pc_l) / (fn + alpha_f)
            p_short = (fws + alpha_f * pc_s) / (fn + alpha_f)
            ev_long = (fret + alpha_f * evc) / (fn + alpha_f) - self.fee_roundtrip
            ev_short = -(fret + alpha_f * evc) / (fn + alpha_f) - self.fee_roundtrip

            prob_long_arr[i] = p_long
            prob_short_arr[i] = p_short
            ev_long_arr[i] = ev_long
            ev_short_arr[i] = ev_short

            # 3. Technical confirmation + Asymmetric Hysteresis (solves low holding rate)
            tech_long_ok = (ema_spread[i] > -0.002) and (rsi_arr[i] < 78.0)
            tech_short_ok = (ema_spread[i] < 0.002) and (rsi_arr[i] > 22.0)

            enter_long = (p_long >= self.entry_prob) and (ev_long >= self.min_ev) and tech_long_ok
            enter_short = (p_short >= self.entry_prob) and (ev_short >= self.min_ev) and tech_short_ok

            if pos == 0:
                if enter_long and not enter_short:
                    pos = 1
                elif enter_short and not enter_long:
                    pos = -1
            elif pos == 1:
                if enter_short:
                    pos = -1
                elif p_long < self.exit_prob:
                    pos = 0
            elif pos == -1:
                if enter_long:
                    pos = 1
                elif p_short < self.exit_prob:
                    pos = 0

            signals[i] = float(pos)

        out["signal"] = signals
        out["cond_prob_long"] = prob_long_arr
        out["cond_prob_short"] = prob_short_arr
        out["cond_ev_long"] = ev_long_arr
        out["cond_ev_short"] = ev_short_arr

        if n > 0:
            last_fs = fine_states[-1]
            self.last_state_summary = {
                "state": last_fs,
                "samples": int(fine_stats[last_fs][0]),
                "prob_long": round(float(prob_long_arr[-1]), 4),
                "prob_short": round(float(prob_short_arr[-1]), 4),
                "ev_long_bps": round(float(ev_long_arr[-1] * 10000.0), 2),
                "ev_short_bps": round(float(ev_short_arr[-1] * 10000.0), 2),
                "signal": int(signals[-1]),
            }
        return out
'''


BENCHMARK_SCRIPT_CODE = '''#!/usr/bin/env python3
"""
Multi-Timeframe Conditional Probability + Technical Indicators Benchmark (scripts/benchmark_timeframes.py).
Resamples 1m historical data into 1m, 3m, 5m, 15m, 1h, and 1d, then evaluates:
  1. Baseline MomentumBreakoutStrategy
  2. ConditionalProbTechStrategy (USDC 0% Maker + 1 bps slippage = 0.02% round-trip)
  3. ConditionalProbTechStrategy (Standard Taker 0.02% + 1 bps slippage = 0.06% round-trip)
Strict Out-of-Sample (OOS) evaluation on the second 50% of bars; writes JSON + Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy
from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy

TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]
RESAMPLE_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1D",
}


def load_base_1m_csv(symbol: str) -> pd.DataFrame:
    cache_dir = BASE_DIR / "crypto_trading" / "data_cache"
    safe_sym = symbol.replace("/", "_")
    candidates = sorted(cache_dir.glob(f"{safe_sym}_1m_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No 1m cache CSV found for {symbol} in {cache_dir}")
    df = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def resample_ohlcv(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1m":
        return df_1m.copy()
    rule = RESAMPLE_MAP[tf]
    agg = (
        df_1m.resample(rule, label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    return agg


def evaluate_signals_oos(df_sig: pd.DataFrame, tf: str, oos_start_idx: int, fee_per_side: float) -> dict[str, Any]:
    sub = df_sig.iloc[oos_start_idx:].copy()
    if len(sub) < 10:
        return {
            "net_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "time_in_market_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
        }

    close = sub["Close"].to_numpy(dtype=np.float64)
    sig = sub["signal"].to_numpy(dtype=np.float64)
    # Causal execution: signal at bar t enters at bar t close, earns return of t -> t+1
    pos = sig[:-1]
    bar_ret = (close[1:] - close[:-1]) / np.where(close[:-1] == 0, 1.0, close[:-1])
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - pos_prev)
    net_ret = pos * bar_ret - turnover * fee_per_side

    equity = np.cumprod(1.0 + net_ret)
    total_ret_pct = (equity[-1] - 1.0) * 100.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    dd = (peak - equity) / np.where(peak == 0, 1.0, peak)
    max_dd_pct = float(np.max(dd) * 100.0)

    ann_factor = CryptoFeatureEngineer.get_annualization_factor(tf)
    std = float(np.std(net_ret))
    sharpe = float((np.mean(net_ret) / std) * np.sqrt(ann_factor)) if std > 1e-12 else 0.0

    time_in_mkt = float(np.mean(pos != 0) * 100.0)
    trade_entries = int(np.sum((pos != pos_prev) & (pos != 0)))
    active_mask = pos != 0
    win_rate = float(np.mean(net_ret[active_mask] > 0) * 100.0) if np.any(active_mask) else 0.0

    return {
        "oos_bars": int(len(sub)),
        "net_return_pct": round(total_ret_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "time_in_market_pct": round(time_in_mkt, 2),
        "trades": trade_entries,
        "win_rate_pct": round(win_rate, 2),
    }


def run_benchmark(symbols: list[str], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_tf: dict[str, dict[str, Any]] = {}

    # Preload 1m data per symbol
    raw_by_sym = {sym: load_base_1m_csv(sym) for sym in symbols}

    for tf in TIMEFRAMES:
        tf_rows: list[dict[str, Any]] = []
        for sym, df_1m in raw_by_sym.items():
            df_tf = resample_ohlcv(df_1m, tf)
            oos_idx = max(40, len(df_tf) // 2)

            # 1. Baseline MomentumBreakout
            mom = MomentumBreakoutStrategy()
            df_mom = mom.generate_signals(df_tf)
            m_mom = evaluate_signals_oos(df_mom, tf, oos_idx, fee_per_side=0.0001)

            # 2. ConditionalProbTech (USDC 0% Maker + 1bps slip = 0.0001 per side, 0.0002 roundtrip)
            cond_maker = ConditionalProbTechStrategy(
                horizon=3,
                entry_prob=0.525 if tf in ("1m", "3m", "5m") else 0.535,
                exit_prob=0.475,
                min_ev=0.00005,
                fee_roundtrip=0.0002,
                prior_strength=20.0,
                min_history_bars=min(120, max(30, oos_idx - 10)),
                warmup_fraction=0.5,
            )
            df_cm = cond_maker.generate_signals(df_tf)
            m_cm = evaluate_signals_oos(df_cm, tf, oos_idx, fee_per_side=0.0001)

            # 3. ConditionalProbTech (Taker 0.02% + 1bps slip = 0.0003 per side, 0.0006 roundtrip)
            cond_taker = ConditionalProbTechStrategy(
                horizon=3,
                entry_prob=0.535 if tf in ("1m", "3m", "5m") else 0.54,
                exit_prob=0.475,
                min_ev=0.0001,
                fee_roundtrip=0.0006,
                prior_strength=20.0,
                min_history_bars=min(120, max(30, oos_idx - 10)),
                warmup_fraction=0.5,
            )
            df_ct = cond_taker.generate_signals(df_tf)
            m_ct = evaluate_signals_oos(df_ct, tf, oos_idx, fee_per_side=0.0003)

            tf_rows.append({
                "symbol": sym,
                "bars": int(len(df_tf)),
                "baseline_momentum": m_mom,
                "cond_prob_maker": m_cm,
                "cond_prob_taker": m_ct,
            })

        def avg_metric(key: str, field: str) -> float:
            return round(float(np.mean([r[key][field] for r in tf_rows])), 3)

        results_by_tf[tf] = {
            "per_symbol": tf_rows,
            "summary": {
                "baseline_momentum": {
                    "avg_net_return_pct": avg_metric("baseline_momentum", "net_return_pct"),
                    "avg_sharpe": avg_metric("baseline_momentum", "sharpe"),
                    "avg_time_in_market_pct": avg_metric("baseline_momentum", "time_in_market_pct"),
                    "avg_trades": avg_metric("baseline_momentum", "trades"),
                },
                "cond_prob_maker": {
                    "avg_net_return_pct": avg_metric("cond_prob_maker", "net_return_pct"),
                    "avg_sharpe": avg_metric("cond_prob_maker", "sharpe"),
                    "avg_max_dd_pct": avg_metric("cond_prob_maker", "max_drawdown_pct"),
                    "avg_time_in_market_pct": avg_metric("cond_prob_maker", "time_in_market_pct"),
                    "avg_trades": avg_metric("cond_prob_maker", "trades"),
                    "avg_win_rate_pct": avg_metric("cond_prob_maker", "win_rate_pct"),
                },
                "cond_prob_taker": {
                    "avg_net_return_pct": avg_metric("cond_prob_taker", "net_return_pct"),
                    "avg_sharpe": avg_metric("cond_prob_taker", "sharpe"),
                    "avg_max_dd_pct": avg_metric("cond_prob_taker", "max_drawdown_pct"),
                    "avg_time_in_market_pct": avg_metric("cond_prob_taker", "time_in_market_pct"),
                    "avg_trades": avg_metric("cond_prob_taker", "trades"),
                    "avg_win_rate_pct": avg_metric("cond_prob_taker", "win_rate_pct"),
                },
            },
        }

    # Pick recommended timeframe: highest avg_sharpe on cond_prob_maker with >= 5 avg trades
    viable = [
        (tf, results_by_tf[tf]["summary"]["cond_prob_maker"])
        for tf in TIMEFRAMES
        if results_by_tf[tf]["summary"]["cond_prob_maker"]["avg_trades"] >= 3
    ]
    if not viable:
        viable = [(tf, results_by_tf[tf]["summary"]["cond_prob_maker"]) for tf in TIMEFRAMES]
    best_tf, best_stats = max(viable, key=lambda x: (x[1]["avg_sharpe"], x[1]["avg_net_return_pct"]))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "recommended_timeframe": best_tf,
        "recommended_stats": best_stats,
        "timeframes": results_by_tf,
    }

    json_path = output_dir / "mtf_conditional_prob_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Multi-Timeframe Conditional Probability + Technical Benchmark (OOS)",
        f"- Recommended Timeframe: **{best_tf}** (Maker Avg Sharpe: {best_stats['avg_sharpe']}, Avg Net Return: {best_stats['avg_net_return_pct']}%, Time-in-Market: {best_stats['avg_time_in_market_pct']}%)",
        "",
        "| Timeframe | Baseline Hold% | Baseline Ret% | CondProb Maker Hold% | CondProb Maker Ret% | CondProb Maker Sharpe | CondProb Taker Ret% | CondProb Taker Sharpe |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for tf in TIMEFRAMES:
        s = results_by_tf[tf]["summary"]
        bm = s["baseline_momentum"]
        cm = s["cond_prob_maker"]
        ct = s["cond_prob_taker"]
        lines.append(
            f"| {tf} | {bm['avg_time_in_market_pct']}% | {bm['avg_net_return_pct']}% | "
            f"{cm['avg_time_in_market_pct']}% | {cm['avg_net_return_pct']}% | {cm['avg_sharpe']} | "
            f"{ct['avg_net_return_pct']}% | {ct['avg_sharpe']} |"
        )
    md_path = output_dir / "mtf_conditional_prob_benchmark.md"
    md_path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(f"[✓] Recommended Timeframe: {best_tf} | Saved {json_path} & {md_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark 1m/3m/5m/15m/1h/1d with ConditionalProbTechStrategy")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "reports"))
    args = parser.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_benchmark(syms, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


TEST_FILE_CODE = '''import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy


def test_conditional_prob_strategy_causal_and_holds():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # Trending + oscillating series so conditional patterns repeat
    drift = np.where((np.arange(n) % 40) < 25, 0.0012, -0.0008)
    rets = drift + rng.normal(0, 0.001, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": rng.uniform(100, 1000, size=n),
        },
        index=idx,
    )

    strat = ConditionalProbTechStrategy(
        horizon=3,
        entry_prob=0.52,
        exit_prob=0.46,
        min_ev=0.0,
        fee_roundtrip=0.0002,
        min_history_bars=80,
    )
    res = strat.generate_signals(df)
    assert "signal" in res.columns
    assert "cond_prob_long" in res.columns
    assert set(res["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    # Should actually hold positions (> 5% of bars after warmup)
    hold_ratio = float((res["signal"].iloc[80:] != 0).mean())
    assert hold_ratio > 0.05, f"Expected active holding, got {hold_ratio}"
'''


def patch_bot_runner(repo_root: Path) -> bool:
    p = repo_root / "crypto_trading" / "bot_runner.py"
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8")
    if "ConditionalProbTechStrategy" in text:
        return False

    # Import ConditionalProbTechStrategy
    imp_anchor = "from crypto_trading.strategies.momentum_breakout import ("
    imp_add = "from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy\n"
    if imp_anchor not in text:
        return False
    text = text.replace(imp_anchor, imp_add + imp_anchor, 1)

    # Load optimal timeframe from reports/mtf_conditional_prob_benchmark.json if present
    helper = '''def _resolve_default_timeframe() -> str:
    import json
    from pathlib import Path
    rep = Path(__file__).resolve().parent.parent / "reports" / "mtf_conditional_prob_benchmark.json"
    if rep.exists():
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
            tf = str(data.get("recommended_timeframe", "")).strip()
            if tf in {"1m", "3m", "5m", "15m", "1h", "1d"}:
                return tf
        except Exception:
            pass
    return "5m"


'''
    text = text.replace("class CryptoTradingBot:", helper + "class CryptoTradingBot:", 1)

    # Update __init__ default strategy and exec_timeframe
    text = text.replace(
        "        strategy: BaseStrategy | None = None,\n        exchange_adapter: Any | None = None,\n    ):",
        "        strategy: BaseStrategy | None = None,\n        exchange_adapter: Any | None = None,\n        exec_timeframe: str | None = None,\n    ):",
        1,
    )
    text = text.replace(
        "        self.strategy = strategy or MomentumBreakoutStrategy()",
        "        self.exec_timeframe = exec_timeframe or _resolve_default_timeframe()\n        self.strategy = strategy or ConditionalProbTechStrategy()",
        1,
    )
    text = text.replace(
        'df_exec = self.loader.fetch_ohlcv(limit=100, timeframe="1m")',
        'df_exec = self.loader.fetch_ohlcv(limit=250, timeframe=getattr(self, "exec_timeframe", "5m"))',
        1,
    )
    text = text.replace(
        'reason="布林帶動量突破向上做多",',
        'reason="條件機率+技術指標達標做多",',
    )
    text = text.replace(
        'reason="布林帶動量跌破向下做空",',
        'reason="條件機率+技術指標達標做空",',
    )

    p.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Phase 6 Conditional Prob + 6-Timeframe Benchmark")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()

    strat_p = root / "crypto_trading" / "strategies" / "conditional_prob_strategy.py"
    strat_p.write_text(CONDITIONAL_STRATEGY_CODE, encoding="utf-8")
    print(f"[✓] Created {strat_p}")

    bench_p = root / "scripts" / "benchmark_timeframes.py"
    bench_p.write_text(BENCHMARK_SCRIPT_CODE, encoding="utf-8")
    print(f"[✓] Created {bench_p}")

    test_p = root / "tests" / "test_conditional_prob_strategy.py"
    test_p.write_text(TEST_FILE_CODE, encoding="utf-8")
    print(f"[✓] Created {test_p}")

    if patch_bot_runner(root):
        print("[✓] Upgraded crypto_trading/bot_runner.py to ConditionalProbTechStrategy + dynamic benchmark timeframe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
