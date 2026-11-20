#!/usr/bin/env python3
"""Standalone Zero-UI Headless Multi-Symbol Crypto Trading Runner.

Runs `MultiSymbolBotManager` in the background without starting `web_ui`,
and writes an atomic PnL snapshot to `data_persistence/crypto_pnl_status.json`
on every cycle so the user can check profit/loss anytime without a browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass

from crypto_trading.multi_symbol_runner import MultiSymbolBotManager, parse_configured_symbols
from crypto_trading.settings import get_settings


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_pnl_snapshot(
    manager: MultiSymbolBotManager,
    *,
    initial_total_equity: float,
    cycle: int,
    trading_mode: str,
) -> dict[str, Any]:
    rows = manager.status_summary()
    total_equity = 0.0
    per_symbol: list[dict[str, Any]] = []
    for row in rows:
        sym = str(row.get("symbol") or "")
        bal = float(row.get("balance_usdt") or 0.0)
        pos = float(row.get("position") or 0.0)
        entry = float(row.get("entry_price") or 0.0)
        eq = float(row.get("equity_usdt") or (bal + (pos * entry if pos > 0 else 0.0)))
        total_equity += eq
        per_symbol.append(
            {
                "symbol": sym,
                "position_side": str(row.get("position_side") or "FLAT"),
                "position": round(pos, 6),
                "entry_price": round(entry, 4),
                "balance_usdt": round(bal, 2),
                "equity_usdt": round(eq, 2),
            }
        )
    pnl_usdt = total_equity - initial_total_equity
    pnl_pct = (pnl_usdt / initial_total_equity * 100.0) if initial_total_equity > 0 else 0.0
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "trading_mode": trading_mode,
        "initial_equity_usdt": round(initial_total_equity, 2),
        "total_equity_usdt": round(total_equity, 2),
        "pnl_usdt": round(pnl_usdt, 2),
        "pnl_pct": round(pnl_pct, 4),
        "symbols": per_symbol,
    }


def run_headless_crypto(
    *,
    symbols: list[str] | None = None,
    interval_sec: float = 15.0,
    iterations: int | None = None,
    status_path: str | Path | None = None,
    manager: MultiSymbolBotManager | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    mode_str = str(getattr(settings, "trading_mode", "PAPER"))
    syms = symbols or parse_configured_symbols()
    mgr = manager or MultiSymbolBotManager(symbols=syms)
    out_path = Path(status_path) if status_path else (BASE_DIR / "data_persistence" / "crypto_pnl_status.json")

    initial_rows = mgr.status_summary()
    initial_eq = sum(
        float(r.get("equity_usdt") or r.get("balance_usdt") or 0.0)
        for r in initial_rows
    )
    print(
        f"[CRYPTO_START] mode={mode_str} symbols={','.join(syms)} "
        f"initial_equity={initial_eq:.2f} USDT status_file={out_path}"
    )

    cycle = 0
    last_snap: dict[str, Any] = {}
    while True:
        cycle += 1
        try:
            mgr.step_all()
        except Exception as exc:
            print(f"[CRYPTO_CYCLE_ERROR] cycle={cycle} error={exc}")
        last_snap = build_pnl_snapshot(
            mgr,
            initial_total_equity=initial_eq,
            cycle=cycle,
            trading_mode=mode_str,
        )
        _atomic_write_json(out_path, last_snap)
        sign = "+" if last_snap["pnl_usdt"] >= 0 else ""
        print(
            f"[CRYPTO_PNL] cycle={cycle} mode={mode_str} "
            f"equity={last_snap['total_equity_usdt']:.2f} USDT "
            f"pnl={sign}{last_snap['pnl_usdt']:.2f} USDT ({sign}{last_snap['pnl_pct']:.2f}%)"
        )
        if iterations is not None and cycle >= iterations:
            break
        time.sleep(max(1.0, float(interval_sec)))

    return last_snap


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Zero-UI Crypto Trading Runner")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("CRYPTO_BOT_INTERVAL_SEC", "15")))
    parser.add_argument("--iterations", type=int, default=0, help="0 = run 24/7 continuously")
    parser.add_argument("--status-file", type=str, default="")
    args = parser.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    iters = args.iterations if args.iterations > 0 else None
    run_headless_crypto(
        symbols=syms,
        interval_sec=args.interval_sec,
        iterations=iters,
        status_path=args.status_file or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
