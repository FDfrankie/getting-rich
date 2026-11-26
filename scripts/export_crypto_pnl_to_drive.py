#!/usr/bin/env python3
"""Zero-Lock Cloud PnL Observability Exporter.

Reads local SQLite (read-only), live/testnet exchange account state (read-only,
10s timeout), model manifests, and running process state, then atomically writes:
  - data_persistence/crypto_pnl_status.json
  - data_persistence/CRYPTO_PNL_DASHBOARD.md
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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

from crypto_trading.settings import get_settings
from data_persistence.state_paths import resolve_sqlite_path

TAIPEI_TZ = timezone(timedelta(hours=8))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _connect_sqlite_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True, timeout=5.0)
    except Exception:
        return sqlite3.connect(str(db_path), timeout=5.0)


def read_sqlite_pnl_summary(db_path: Path | None = None) -> dict[str, Any]:
    resolved = db_path or Path(
        resolve_sqlite_path(
            "STATE_DB_PATH",
            "system_state.db",
            BASE_DIR / "data_persistence" / "system_state.db",
        )
    )
    if not Path(resolved).exists():
        return {
            "db_path": str(resolved),
            "db_exists": False,
            "total_closes": 0,
            "win_count": 0,
            "win_rate_pct": 0.0,
            "realized_pnl_usdt": 0.0,
            "total_fees_usdt": 0.0,
            "filtered_test_rows_count": 0,
            "recent_closes": [],
            "bot_state": {},
        }

    conn = _connect_sqlite_ro(Path(resolved))
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        filtered_test_rows_count = 0
        recent_closes: list[dict[str, Any]] = []
        total_closes = 0
        win_count = 0
        realized_pnl_usdt = 0.0

        if "trade_logs" in tables:
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(trade_logs)").fetchall()
            }
            has_exch_id = "exchange_order_id" in cols
            has_price = "price" in cols
            has_pnl = "pnl" in cols

            rows = conn.execute(
                "SELECT * FROM trade_logs ORDER BY id DESC LIMIT 2000"
            ).fetchall()
            real_closes: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                price = float(d.get("price") or 0.0) if has_price else 0.0
                exch_id = str(d.get("exchange_order_id") or "") if has_exch_id else ""
                if price in (45000.0, 50000.0, 55000.0) and exch_id.startswith("PAPER_"):
                    filtered_test_rows_count += 1
                    continue

                action = str(d.get("action") or "").upper()
                pnl_val = float(d.get("pnl") or 0.0) if has_pnl else 0.0
                is_close = ("CLOSE" in action) or (pnl_val != 0.0)
                if is_close:
                    real_closes.append(
                        {
                            "id": d.get("id"),
                            "timestamp": d.get("timestamp") or d.get("created_at") or "",
                            "symbol": d.get("symbol") or "",
                            "action": d.get("action") or "",
                            "price": round(price, 4),
                            "amount": round(float(d.get("amount") or 0.0), 6),
                            "pnl_usdt": round(pnl_val, 4),
                            "exchange_order_id": exch_id,
                        }
                    )

            total_closes = len(real_closes)
            win_count = sum(1 for r in real_closes if r["pnl_usdt"] > 0)
            realized_pnl_usdt = sum(r["pnl_usdt"] for r in real_closes)
            recent_closes = real_closes[:20]

        total_fees_usdt = 0.0
        if "orders" in tables:
            try:
                fee_row = conn.execute(
                    "SELECT COALESCE(SUM(fee), 0.0) FROM orders WHERE status = 'FILLED' AND (exchange_order_id NOT LIKE 'PAPER_%' OR price NOT IN (45000.0, 50000.0, 55000.0))"
                ).fetchone()
                total_fees_usdt = float(fee_row[0] or 0.0) if fee_row else 0.0
            except Exception:
                total_fees_usdt = 0.0

        bot_state: dict[str, Any] = {}
        if "bot_state" in tables:
            bs_rows = conn.execute("SELECT key, value FROM bot_state").fetchall()
            bot_state = {str(r["key"]): r["value"] for r in bs_rows}

        win_rate_pct = (win_count / total_closes * 100.0) if total_closes > 0 else 0.0
        return {
            "db_path": str(resolved),
            "db_exists": True,
            "total_closes": total_closes,
            "win_count": win_count,
            "win_rate_pct": round(win_rate_pct, 2),
            "realized_pnl_usdt": round(realized_pnl_usdt, 4),
            "total_fees_usdt": round(total_fees_usdt, 4),
            "filtered_test_rows_count": filtered_test_rows_count,
            "recent_closes": recent_closes,
            "bot_state": bot_state,
        }
    finally:
        conn.close()


def fetch_exchange_account_ro(trading_mode: str) -> dict[str, Any]:
    raw_val = getattr(trading_mode, "value", trading_mode)
    mode = str(raw_val or "LOCAL_PAPER").strip().upper()
    if mode.startswith("TRADINGMODE."):
        mode = mode.split(".", 1)[1]
    out: dict[str, Any] = {
        "trading_mode": mode,
        "connected": False,
        "total_equity_usdt": None,
        "free_balance_usdt": None,
        "unrealized_pnl_usdt": None,
        "open_positions": [],
        "exchange_error": None,
    }
    if mode not in ("BINANCE_TESTNET", "BINANCE_LIVE"):
        return out

    api_key = (os.getenv("BINANCE_TESTNET_API_KEY") or "").strip() if mode == "BINANCE_TESTNET" else (os.getenv("BINANCE_API_KEY") or "").strip()
    api_secret = (os.getenv("BINANCE_TESTNET_API_SECRET") or "").strip() if mode == "BINANCE_TESTNET" else (os.getenv("BINANCE_API_SECRET") or "").strip()
    if not api_key or not api_secret:
        out["exchange_error"] = f"Missing API key/secret for {mode}"
        return out

    try:
        import ccxt  # type: ignore

        exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "timeout": 10000,
                "enableRateLimit": True,
            }
        )
        if mode == "BINANCE_TESTNET":
            # P52: CCXT deprecated binanceusdm.set_sandbox_mode(True); prefer enable_demo_trading(True)
            if hasattr(exchange, "enable_demo_trading"):
                exchange.enable_demo_trading(True)
            else:
                exchange.set_sandbox_mode(True)

        bal = exchange.fetch_balance()
        usdt_info = bal.get("USDT") or {}
        usdc_info = bal.get("USDC") or {}
        total_eq = float(usdt_info.get("total") or 0.0)
        free_bal = float(usdt_info.get("free") or 0.0)
        legacy_usdc = float(usdc_info.get("total") or 0.0)

        positions_raw = exchange.fetch_positions()
        open_positions: list[dict[str, Any]] = []
        total_upnl = 0.0
        for p in positions_raw or []:
            contracts = float(p.get("contracts") or 0.0)
            if abs(contracts) <= 0.0:
                continue
            upnl = float(p.get("unrealizedPnl") or 0.0)
            total_upnl += upnl
            open_positions.append(
                {
                    "symbol": str(p.get("symbol") or ""),
                    "side": str(p.get("side") or "").upper(),
                    "size": round(contracts, 6),
                    "entry_price": round(float(p.get("entryPrice") or 0.0), 4),
                    "mark_price": round(float(p.get("markPrice") or 0.0), 4),
                    "unrealized_pnl": round(upnl, 4),
                    "leverage": p.get("leverage"),
                }
            )

        out.update(
            {
                "connected": True,
                "settlement_currency": "USDT",
                "total_equity_usdt": round(total_eq, 4),
                "free_balance_usdt": round(free_bal, 4),
                "legacy_usdc_balance": round(legacy_usdc, 4),
                "unrealized_pnl_usdt": round(total_upnl, 4),
                "open_positions": open_positions,
            }
        )
    except Exception as exc:
        out["exchange_error"] = f"{type(exc).__name__}: {exc}"
    return out


def read_model_statuses() -> dict[str, Any]:
    models_dir = BASE_DIR / "data_persistence" / "models"
    result: dict[str, Any] = {}
    for key, fname in (
        ("crypto_lgbm", "crypto_lgbm_manifest.json"),
        ("crypto_ppo", "crypto_ppo_agent_manifest.json"),
    ):
        p = models_dir / fname
        if not p.exists():
            result[key] = {"exists": False, "deployment_stage": "MISSING"}
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result[key] = {
                "exists": True,
                "deployment_stage": data.get("deployment_stage", "UNKNOWN"),
                "trained_at": data.get("trained_at", ""),
                "timeframe": data.get("timeframe", ""),
                "oos_metrics": data.get("oos_metrics") or data.get("metrics") or {},
            }
        except Exception as exc:
            result[key] = {"exists": True, "deployment_stage": "ERROR", "error": str(exc)}
    return result


def detect_crypto_bot_processes() -> tuple[bool, list[int]]:
    if os.name == "nt":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*headless_crypto_bot*' } | Select-Object -ExpandProperty ProcessId",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            pids = [int(line.strip()) for line in res.stdout.splitlines() if line.strip().isdigit()]
            return (len(pids) > 0, pids)
        except Exception:
            return (False, [])
    return (False, [])


def render_dashboard_md(payload: dict[str, Any]) -> str:
    ex = payload.get("exchange") or {}
    sq = payload.get("sqlite") or {}
    models = payload.get("models") or {}
    lines = [
        "# Crypto PnL Cloud Dashboard",
        "",
        f"- **Updated (Taipei UTC+8)**: `{payload.get('updated_at_taipei', '')}`",
        f"- **Trading Mode**: `{payload.get('trading_mode', 'LOCAL_PAPER')}`",
        f"- **Crypto Bot Running**: `{payload.get('crypto_bot_running', False)}` (PIDs: `{payload.get('crypto_bot_pids', [])}`)",
        "",
        "## Account & PnL Summary",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Equity (USDT) | `{ex.get('total_equity_usdt')}` |",
        f"| Free Balance (USDT) | `{ex.get('free_balance_usdt')}` |",
        f"| Unrealized PnL (USDT) | `{ex.get('unrealized_pnl_usdt')}` |",
        f"| Settlement Currency | `USDT` (VIP0 Maker 0.02% / Taker 0.05%) |",
        f"| Realized Net PnL (SQLite, USDT) | `{sq.get('realized_pnl_usdt', 0.0)}` |",
        f"| Cumulative Recorded Fees (USDT) | `{sq.get('total_fees_usdt', 0.0)}` |",
        f"| Total Real Closes / Win Rate | `{sq.get('total_closes', 0)}` / `{sq.get('win_rate_pct', 0.0)}%` |",
        f"| Filtered Unit-Test Rows | `{sq.get('filtered_test_rows_count', 0)}` |",
        f"| Exchange Error | `{ex.get('exchange_error') or 'None'}` |",
        "",
        "## Open Positions",
    ]
    positions = ex.get("open_positions") or []
    if positions:
        lines.extend(
            [
                "| Symbol | Side | Size | Entry | Mark | Unrealized PnL | Leverage |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for p in positions:
            lines.append(
                f"| {p['symbol']} | {p['side']} | {p['size']} | {p['entry_price']} | {p['mark_price']} | {p['unrealized_pnl']} | {p.get('leverage', '')} |"
            )
    else:
        lines.append("_No open positions._")

    lines.extend(["", "## Recent 10 Real Closes"])
    closes = (sq.get("recent_closes") or [])[:10]
    if closes:
        lines.extend(
            [
                "| Time | Symbol | Action | Price | Amount | PnL (USDT) |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for c in closes:
            lines.append(
                f"| {c['timestamp']} | {c['symbol']} | {c['action']} | {c['price']} | {c['amount']} | {c['pnl_usdt']} |"
            )
    else:
        lines.append("_No real close records._")

    lines.extend(["", "## Model Status", "| Model | Stage | Timeframe | Trained At |", "|---|---|---|---|"])
    for m_name, m_info in models.items():
        lines.append(
            f"| {m_name} | `{m_info.get('deployment_stage', 'UNKNOWN')}` | `{m_info.get('timeframe', '')}` | `{m_info.get('trained_at', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def export_crypto_pnl_to_drive(
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    db_path: Path | None = None,
    exchange_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    _raw_tm = getattr(settings, "trading_mode", "LOCAL_PAPER")
    trading_mode = str(getattr(_raw_tm, "value", _raw_tm) or "LOCAL_PAPER").strip().upper()
    if trading_mode.startswith("TRADINGMODE."):
        trading_mode = trading_mode.split(".", 1)[1]
    sq_summary = read_sqlite_pnl_summary(db_path=db_path)
    effective_exchange_mode = trading_mode
    if (
        effective_exchange_mode == "LOCAL_PAPER"
        and str((sq_summary.get("bot_state") or {}).get("execution_mode") or "").upper() == "BINANCE_TESTNET"
        and (os.getenv("BINANCE_TESTNET_API_KEY") or "").strip()
        and (os.getenv("BINANCE_TESTNET_API_SECRET") or "").strip()
    ):
        effective_exchange_mode = "BINANCE_TESTNET"
    ex_summary = exchange_override if exchange_override is not None else fetch_exchange_account_ro(effective_exchange_mode)
    models = read_model_statuses()
    bot_running, bot_pids = detect_crypto_bot_processes()

    now_utc = datetime.now(timezone.utc)
    now_tpe = now_utc.astimezone(TAIPEI_TZ)
    payload: dict[str, Any] = {
        "updated_at_utc": now_utc.isoformat(),
        "updated_at_taipei": now_tpe.strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "trading_mode": trading_mode,
        "crypto_bot_running": bot_running,
        "crypto_bot_pids": bot_pids,
        "sqlite": sq_summary,
        "exchange": ex_summary,
        "models": models,
    }

    out_json = Path(json_path) if json_path else (BASE_DIR / "data_persistence" / "crypto_pnl_status.json")
    out_md = Path(md_path) if md_path else (BASE_DIR / "data_persistence" / "CRYPTO_PNL_DASHBOARD.md")
    _atomic_write_json(out_json, payload)
    _atomic_write_text(out_md, render_dashboard_md(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-Lock Cloud PnL Observability Exporter")
    parser.add_argument("--once", action="store_true", help="Run a single export and exit")
    parser.add_argument("--json-path", type=str, default="")
    parser.add_argument("--md-path", type=str, default="")
    args = parser.parse_args()

    payload = export_crypto_pnl_to_drive(
        json_path=Path(args.json_path) if args.json_path else None,
        md_path=Path(args.md_path) if args.md_path else None,
    )
    ex = payload.get("exchange") or {}
    sq = payload.get("sqlite") or {}
    print(
        f"[CRYPTO_PNL_EXPORT] mode={payload['trading_mode']} "
        f"equity={ex.get('total_equity_usdt')} free={ex.get('free_balance_usdt')} "
        f"upnl={ex.get('unrealized_pnl_usdt')} realized={sq.get('realized_pnl_usdt')} "
        f"closes={sq.get('total_closes')} win_rate={sq.get('win_rate_pct')}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
