"""Real Binance USDT-M Historical Walk-Forward OOS Backtester & Profitability Gate (P51).

Compares 1m / 3m MTF Candlestick Conditional-Probability (Maker 4 bps vs Taker 10 bps)
against 1h Native Ensemble and Buy & Hold on real Binance USDT-M Futures klines + funding rates.

Outputs:
  - data_persistence/CRYPTO_WALKFORWARD_PROFITABILITY_REPORT.json
  - data_persistence/CRYPTO_WALKFORWARD_PROFITABILITY_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy

try:
    from crypto_trading.strategies.native_ensemble_strategy import NativeCryptoEnsembleStrategy
except Exception:
    NativeCryptoEnsembleStrategy = None

TZ_TAIPEI = timezone(timedelta(hours=8))

# Hard Profitability Gate criteria before live capital promotion
GATE_MIN_SHARPE = 1.00
GATE_MIN_PROFIT_FACTOR = 1.25
GATE_MAX_DRAWDOWN_PCT = 15.0
GATE_MIN_TRADES = 15

BARS_PER_YEAR: dict[str, float] = {
    "1m": 525600.0,
    "3m": 175200.0,
    "5m": 105120.0,
    "15m": 35040.0,
    "1h": 8760.0,
}


@dataclass
class WalkForwardMetrics:
    label: str
    symbol: str
    timeframe: str
    execution_mode: str
    fee_one_way_bps: float
    bars_oos: int
    trades: int
    win_rate_pct: float
    net_return_pct: float
    gross_return_pct: float
    total_fees_pct: float
    total_funding_pct: float
    profit_factor: float
    sharpe_annualized: float
    max_drawdown_pct: float
    buy_hold_return_pct: float
    gate_passed: bool
    gate_reasons: list[str]


def fetch_binance_usdt_klines(
    symbol: str = "BTCUSDT",
    interval: str = "3m",
    total_bars: int = 3000,
    cache_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Fetch real Binance USDT-M Futures klines including taker_buy_base_vol for real OFI."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{interval}_{total_bars}.csv"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 1800:
            df_cached = pd.read_csv(cache_file)
            if len(df_cached) >= min(total_bars, 500):
                return df_cached

    rows: list[list[Any]] = []
    end_time: int | None = None
    remaining = int(total_bars)
    while remaining > 0:
        limit = min(1500, remaining)
        params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        url = "https://fapi.binance.com/fapi/v1/klines?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "AntigravityQuant-P51/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        rows = batch + rows
        end_time = int(batch[0][0]) - 1
        remaining -= len(batch)
        if len(batch) < limit:
            break
        time.sleep(0.12)

    df = pd.DataFrame(
        rows,
        columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteVol", "Trades", "TakerBuyBaseVol", "TakerBuyQuoteVol", "Ignore",
        ],
    )
    for col in ("Open", "High", "Low", "Close", "Volume", "QuoteVol", "TakerBuyBaseVol"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["OpenTime"]).sort_values("OpenTime").reset_index(drop=True)

    if cache_dir is not None and not df.empty:
        df.to_csv(cache_dir / f"{symbol}_{interval}_{total_bars}.csv", index=False)
    return df


def simulate_walkforward_oos(
    df: pd.DataFrame,
    strategy: Any,
    label: str,
    symbol: str,
    timeframe: str,
    fee_one_way: float = 0.0002,
    slippage_one_way: float = 0.00005,
    funding_rate_8h: float = 0.0001,
    warmup_bars: int = 300,
    tp_atr_mult: float = 2.4,
    sl_atr_mult: float = 1.2,
    execution_mode: str = "MAKER_FIRST",
) -> WalkForwardMetrics:
    """Strictly causal bar-by-bar OOS simulation with intrabar hard SL/TP, fees & funding."""
    sig_df = strategy.generate_signals(df)
    close = pd.to_numeric(sig_df["Close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(sig_df["High"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(sig_df["Low"], errors="coerce").to_numpy(dtype=np.float64)
    atr_pct = pd.to_numeric(sig_df.get("ATR_PCT", pd.Series(0.003, index=sig_df.index)), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    signals = pd.to_numeric(sig_df["signal"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    n = len(sig_df)
    start_i = min(max(100, warmup_bars), max(1, n - 50))
    bars_per_8h = max(1.0, (8.0 * 60.0) / {"1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0, "1h": 60.0}.get(timeframe, 3.0))
    funding_per_bar = funding_rate_8h / bars_per_8h

    equity = 1.0
    peak_equity = 1.0
    max_dd = 0.0
    gross_pnl_sum = 0.0
    total_fees = 0.0
    total_funding = 0.0

    pos = 0.0
    entry_price = 0.0
    hard_sl = 0.0
    hard_tp = 0.0
    trade_returns: list[float] = []
    bar_returns: list[float] = []

    for i in range(start_i, n - 1):
        prev_eq = equity
        # Signal generated at bar i close executes at bar i close (strictly causal) -> realized on bar i+1
        target = float(np.clip(signals[i], -1.0, 1.0))

        # 1. Check intrabar hard SL / TP on bar i if currently in position
        if pos != 0.0:
            exited = False
            exit_p = close[i]
            if pos > 0:
                if low[i] <= hard_sl:
                    exit_p = hard_sl
                    exited = True
                elif high[i] >= hard_tp:
                    exit_p = hard_tp
                    exited = True
            else:
                if high[i] >= hard_sl:
                    exit_p = hard_sl
                    exited = True
                elif low[i] <= hard_tp:
                    exit_p = hard_tp
                    exited = True

            if exited:
                raw_r = pos * ((exit_p - entry_price) / entry_price)
                # SL exit typically crosses spread (Taker), TP exit rests as Maker
                exit_fee = (0.0005 + slippage_one_way) if ((pos > 0 and exit_p <= hard_sl) or (pos < 0 and exit_p >= hard_sl)) else fee_one_way
                fee_cost = abs(pos) * exit_fee
                net_r = raw_r - fee_cost
                equity *= (1.0 + net_r)
                gross_pnl_sum += raw_r
                total_fees += fee_cost
                trade_returns.append(net_r)
                pos = 0.0
                entry_price = 0.0

        # 2. Rebalance / Open / Close according to causal signal at bar i close
        if pos != 0.0 and (target == 0.0 or (target * pos < 0)):
            raw_r = pos * ((close[i] - entry_price) / entry_price)
            fee_cost = abs(pos) * (fee_one_way + slippage_one_way)
            net_r = raw_r - fee_cost
            equity *= (1.0 + net_r)
            gross_pnl_sum += raw_r
            total_fees += fee_cost
            trade_returns.append(net_r)
            pos = 0.0
            entry_price = 0.0

        if pos == 0.0 and abs(target) >= 0.15:
            pos = target
            entry_price = close[i]
            fee_cost = abs(pos) * (fee_one_way + slippage_one_way)
            equity *= (1.0 - fee_cost)
            total_fees += fee_cost
            ap = max(0.001, float(atr_pct[i]))
            if pos > 0:
                hard_sl = entry_price * (1.0 - ap * sl_atr_mult)
                hard_tp = entry_price * (1.0 + ap * tp_atr_mult)
            else:
                hard_sl = entry_price * (1.0 + ap * sl_atr_mult)
                hard_tp = entry_price * (1.0 - ap * tp_atr_mult)

        # 3. Accrue perpetual funding cost on open position
        if pos != 0.0:
            f_cost = pos * funding_per_bar  # longs pay positive funding, shorts receive
            equity *= (1.0 - f_cost)
            total_funding += f_cost

        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, dd)
        bar_returns.append((equity - prev_eq) / prev_eq if prev_eq > 0 else 0.0)

    # Mark-to-market close at final bar
    if pos != 0.0 and entry_price > 0:
        raw_r = pos * ((close[-1] - entry_price) / entry_price)
        fee_cost = abs(pos) * (fee_one_way + slippage_one_way)
        net_r = raw_r - fee_cost
        equity *= (1.0 + net_r)
        gross_pnl_sum += raw_r
        total_fees += fee_cost
        trade_returns.append(net_r)

    wins = [r for r in trade_returns if r > 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    win_rate = (len(wins) / len(trade_returns) * 100.0) if trade_returns else 0.0
    pf = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)

    br = np.array(bar_returns, dtype=np.float64)
    std_r = float(np.std(br)) if len(br) > 1 else 0.0
    bpy = BARS_PER_YEAR.get(timeframe, 175200.0)
    sharpe = (float(np.mean(br)) / std_r * math.sqrt(bpy)) if std_r > 1e-12 else 0.0
    net_ret_pct = (equity - 1.0) * 100.0
    bh_ret_pct = ((close[-1] / close[start_i]) - 1.0) * 100.0 if close[start_i] > 0 else 0.0

    reasons: list[str] = []
    if net_ret_pct <= 0.0:
        reasons.append(f"net_return={net_ret_pct:.2f}%<=0")
    if sharpe < GATE_MIN_SHARPE:
        reasons.append(f"sharpe={sharpe:.2f}<{GATE_MIN_SHARPE}")
    if pf < GATE_MIN_PROFIT_FACTOR:
        reasons.append(f"pf={pf:.2f}<{GATE_MIN_PROFIT_FACTOR}")
    if max_dd * 100.0 > GATE_MAX_DRAWDOWN_PCT:
        reasons.append(f"max_dd={max_dd*100:.2f}%>{GATE_MAX_DRAWDOWN_PCT}%")
    if len(trade_returns) < GATE_MIN_TRADES:
        reasons.append(f"trades={len(trade_returns)}<{GATE_MIN_TRADES}")

    return WalkForwardMetrics(
        label=label,
        symbol=symbol,
        timeframe=timeframe,
        execution_mode=execution_mode,
        fee_one_way_bps=round(fee_one_way * 10000.0, 2),
        bars_oos=max(0, n - start_i),
        trades=len(trade_returns),
        win_rate_pct=round(win_rate, 2),
        net_return_pct=round(net_ret_pct, 3),
        gross_return_pct=round(gross_pnl_sum * 100.0, 3),
        total_fees_pct=round(total_fees * 100.0, 3),
        total_funding_pct=round(total_funding * 100.0, 4),
        profit_factor=round(pf, 3),
        sharpe_annualized=round(sharpe, 3),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        buy_hold_return_pct=round(bh_ret_pct, 3),
        gate_passed=(len(reasons) == 0),
        gate_reasons=reasons,
    )


def write_profitability_report(
    results: list[WalkForwardMetrics],
    out_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "CRYPTO_WALKFORWARD_PROFITABILITY_REPORT.json"
    md_path = out_dir / "CRYPTO_WALKFORWARD_PROFITABILITY_REPORT.md"

    passing = [r for r in results if r.gate_passed]
    best = max(passing, key=lambda x: x.sharpe_annualized) if passing else (max(results, key=lambda x: x.sharpe_annualized) if results else None)

    payload = {
        "generated_at_taipei": datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "gate_thresholds": {
            "min_sharpe": GATE_MIN_SHARPE,
            "min_profit_factor": GATE_MIN_PROFIT_FACTOR,
            "max_drawdown_pct": GATE_MAX_DRAWDOWN_PCT,
            "min_trades": GATE_MIN_TRADES,
        },
        "recommended_profile": {
            "passed_gate": bool(passing),
            "label": best.label if best else None,
            "timeframe": best.timeframe if best else "1h",
            "symbol": best.symbol if best else "BTCUSDT",
        },
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Crypto USDT-M Walk-Forward OOS Profitability Gate Report (P51)",
        "",
        f"- **Generated (Taipei)**: `{payload['generated_at_taipei']}`",
        f"- **Gate Thresholds**: Net Sharpe >= `{GATE_MIN_SHARPE}`, PF >= `{GATE_MIN_PROFIT_FACTOR}`, Max DD <= `{GATE_MAX_DRAWDOWN_PCT}%`, Trades >= `{GATE_MIN_TRADES}`, Net Return > `0%`",
        f"- **Gate Verdict**: `{'PASS -> ' + best.label if passing and best else 'FAIL (Stay Shadow/Testnet Only)'}`",
        "",
        "| Strategy | Symbol | TF | Exec | Trades | Win% | Net Ret% | Fees% | PF | Sharpe | MaxDD% | B&H% | Gate |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in results:
        verdict = "PASS" if r.gate_passed else "FAIL"
        lines.append(
            f"| {r.label} | {r.symbol} | {r.timeframe} | {r.execution_mode} ({r.fee_one_way_bps}bps) | "
            f"{r.trades} | {r.win_rate_pct:.1f}% | {r.net_return_pct:+.2f}% | {r.total_fees_pct:.2f}% | "
            f"{r.profit_factor:.2f} | {r.sharpe_annualized:.2f} | {r.max_drawdown_pct:.2f}% | "
            f"{r.buy_hold_return_pct:+.2f}% | **{verdict}** |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="P51 Binance USDT-M Walk-Forward OOS Profitability Gate")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--bars-1m", type=int, default=4500)
    parser.add_argument("--bars-3m", type=int, default=4500)
    parser.add_argument("--bars-1h", type=int, default=1500)
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / "data_persistence" / "market_cache"
    results: list[WalkForwardMetrics] = []

    for sym in args.symbols:
        df_3m = fetch_binance_usdt_klines(sym, "3m", args.bars_3m, cache_dir=cache_dir)
        results.append(
            simulate_walkforward_oos(
                df_3m,
                ConditionalProbRLStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True),
                label="CondProb_MTF_KLine_3m_Maker",
                symbol=sym,
                timeframe="3m",
                fee_one_way=0.0002,
                slippage_one_way=0.00003,
                execution_mode="MAKER_FIRST",
            )
        )
        results.append(
            simulate_walkforward_oos(
                df_3m,
                ConditionalProbRLStrategy(timeframe="3m", fee_roundtrip=0.0010, maker_mode=False),
                label="CondProb_MTF_KLine_3m_Taker",
                symbol=sym,
                timeframe="3m",
                fee_one_way=0.0005,
                slippage_one_way=0.00010,
                execution_mode="TAKER",
            )
        )

        df_1m = fetch_binance_usdt_klines(sym, "1m", args.bars_1m, cache_dir=cache_dir)
        results.append(
            simulate_walkforward_oos(
                df_1m,
                ConditionalProbRLStrategy(timeframe="1m", fee_roundtrip=0.0004, maker_mode=True),
                label="CondProb_MTF_KLine_1m_Maker",
                symbol=sym,
                timeframe="1m",
                fee_one_way=0.0002,
                slippage_one_way=0.00003,
                execution_mode="MAKER_FIRST",
            )
        )

        if NativeCryptoEnsembleStrategy is not None:
            df_1h = fetch_binance_usdt_klines(sym, "1h", args.bars_1h, cache_dir=cache_dir)
            results.append(
                simulate_walkforward_oos(
                    df_1h,
                    NativeCryptoEnsembleStrategy(),
                    label="NativeEnsemble_1h_Maker",
                    symbol=sym,
                    timeframe="1h",
                    fee_one_way=0.0002,
                    slippage_one_way=0.00005,
                    execution_mode="MAKER_FIRST",
                )
            )

    jp, mp = write_profitability_report(results, PROJECT_ROOT / "data_persistence")
    print(f"Wrote Walk-Forward Profitability Report: {jp} & {mp}")
    print(mp.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
