#!/usr/bin/env python3
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
