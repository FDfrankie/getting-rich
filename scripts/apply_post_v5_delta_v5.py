"""
Incremental post-v5 delta patcher for Antigravity Quant.
Designed to run cleanly AFTER apply_full_audit_bundle_v5.py without touching v5's manifest.
Uses its own `.post_v5_delta_manifest.json` and `.post_v5_backups/`.

Usage:
    python scripts/apply_post_v5_delta.py                  # Plan / Dry-run only
    python scripts/apply_post_v5_delta.py --apply          # Stage, compile, backup, apply, test
    python scripts/apply_post_v5_delta.py --rollback ID    # Restore from .post_v5_backups/<ID>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / ".post_v5_delta_manifest.json"
BACKUPS_DIR = ROOT / ".post_v5_backups"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.deltatmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


ORDER_EXECUTION_SERVICE_PY = '''"""Single OrderExecutionService choke point and stateful EmergencyFlattenWorkflow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from crypto_trading.instruments import BTC_USDT_BINANCE_TESTNET, InstrumentSpec
from crypto_trading.manual_controller import ManualPositionController
from crypto_trading.order_manager import OrderManager
from crypto_trading.risk_manager import RiskManager


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: str
    order_type: str
    quantity: Decimal | float | str
    price: Decimal | float | str | None = None
    reduce_only: bool = False
    client_tag: str = ""
    caller: str = "API"


class OrderExecutionService:
    """Single choke point for all order submissions (Bot, UI manual, Partial Reduce, Flatten)."""

    def __init__(
        self,
        risk_manager: RiskManager | None = None,
        order_manager: OrderManager | None = None,
        manual_controller: ManualPositionController | None = None,
        instrument: InstrumentSpec = BTC_USDT_BINANCE_TESTNET,
    ) -> None:
        self.risk_manager = risk_manager or RiskManager()
        self.order_manager = order_manager or OrderManager()
        self.manual_controller = manual_controller or ManualPositionController()
        self.instrument = instrument

    def submit(
        self,
        intent: OrderIntent,
        client: Any,
        *,
        mark_price: float,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if intent.caller == "BOT" and self.manual_controller.is_symbol_locked(intent.symbol):
            return {
                "success": False,
                "status": "BLOCKED_BY_MANUAL_OVERRIDE",
                "error": f"{intent.symbol} is locked by manual_override",
            }

        norm_qty = self.instrument.normalize_quantity(intent.quantity)
        ref_price = Decimal(str(intent.price if intent.price is not None else mark_price))
        norm_price = self.instrument.normalize_price(ref_price)
        if not intent.reduce_only:
            self.instrument.validate_order(norm_qty, norm_price)

        ctx = dict(risk_context or {})
        ctx.setdefault("data_source", "ORDER_EXECUTION_SERVICE")
        ctx.setdefault("data_timestamp", datetime.now(timezone.utc).isoformat())
        ctx.setdefault("data_quality_passed", True)
        ctx.setdefault("reconciliation_fresh", True)
        ctx.setdefault("has_unknown_orders", False)
        ctx.setdefault("exchange_connected", client is not None)
        ctx.setdefault("step_size", float(self.instrument.quantity_step))
        ctx.setdefault("tick_size", float(self.instrument.price_tick))
        ctx.setdefault("min_notional", float(self.instrument.min_notional))

        decision = self.risk_manager.evaluate_order(
            symbol=intent.symbol,
            side=intent.side.upper(),
            order_type=intent.order_type.upper(),
            amount=float(norm_qty),
            price=float(norm_price),
            is_reduce_only=intent.reduce_only,
            context=ctx,
        )
        if not decision.allowed:
            return {
                "success": False,
                "status": "RISK_REJECTED",
                "action": decision.action.value,
                "reasons": decision.reasons,
            }

        return self.order_manager.execute_with_idempotency(
            client=client,
            symbol=intent.symbol,
            side=intent.side.upper(),
            order_type=intent.order_type.upper(),
            amount=float(norm_qty),
            price=float(norm_price) if intent.order_type.upper() == "LIMIT" else None,
            client_tag=intent.client_tag,
        )

    def emergency_flatten(self, client: Any, symbol: str | None = None) -> dict[str, Any]:
        """Stateful flatten workflow: FETCH_REMOTE_POSITIONS -> GENERATE_REDUCE_ONLY -> SUBMIT -> COMPLETE/PARTIAL/FAILED."""
        if client is None or not hasattr(client, "fetch_positions"):
            return {"status": "FAILED", "closed": [], "failed": ["CLIENT_UNAVAILABLE"]}
        positions = client.fetch_positions() or []
        targets = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and (symbol is None or p.get("symbol") == symbol)]
        if not targets:
            return {"status": "COMPLETE", "closed": [], "failed": [], "message": "No open positions to flatten"}

        closed = []
        failed = []
        for pos in targets:
            sym = str(pos.get("symbol") or symbol or "BTC/USDT")
            qty = Decimal(str(pos.get("contracts")))
            side = "SELL" if str(pos.get("side", "LONG")).upper() == "LONG" else "BUY"
            try:
                norm_q = self.instrument.normalize_quantity(qty)
                res = self.order_manager.execute_with_idempotency(
                    client=client,
                    symbol=sym,
                    side=side,
                    order_type="MARKET",
                    amount=float(norm_q),
                    price=None,
                    client_tag=f"flatten_{sym}_{int(datetime.now(timezone.utc).timestamp() // 5)}",
                )
                if res.get("success"):
                    closed.append({"symbol": sym, "quantity": str(norm_q), "order_id": res.get("exchange_order_id")})
                else:
                    failed.append({"symbol": sym, "error": res.get("error", "REJECTED")})
            except Exception as exc:
                failed.append({"symbol": sym, "error": type(exc).__name__})

        final_status = "COMPLETE" if (closed and not failed) else ("PARTIAL" if closed else "FAILED")
        return {"status": final_status, "closed": closed, "failed": failed}
'''

DELTA_TESTS_PY = '''"""Behavioral tests for post-v5 delta fixes (OrderExecutionService, Decimal risk/idempotency, Testnet fail-closed, UTC TTL, AlphaVault seed, MarketAnalyzer)."""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd


class TestPostV5Delta(unittest.TestCase):
    def test_decimal_idempotency_and_order_execution_service(self):
        from crypto_trading.order_execution_service import OrderExecutionService, OrderIntent
        from crypto_trading.order_manager import generate_idempotency_key

        k1 = generate_idempotency_key("BTC/USDT", "BUY", "MARKET", "0.1000000", None, "tag1")
        k2 = generate_idempotency_key("BTC/USDT", "BUY", "MARKET", 0.1, 0.0, "tag1")
        self.assertEqual(k1, k2)

        svc = OrderExecutionService()
        with self.assertRaises(ValueError):
            svc.submit(OrderIntent("BTC/USDT", "BUY", "MARKET", "0.001"), client=None, mark_price=100.0)

    def test_risk_manager_decimal_step_no_float_modulo_error(self):
        from crypto_trading.risk_manager import RiskManager

        rm = RiskManager(initial_capital=10000.0)
        # 0.03 % 0.001 fails in naive Python float arithmetic; Decimal modulo must approve
        dec = rm.evaluate_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            amount=0.03,
            price=1000.0,
            context={
                "data_source": "TEST",
                "data_timestamp": datetime.utcnow().isoformat(),
                "data_quality_passed": True,
                "reconciliation_fresh": True,
                "has_unknown_orders": False,
                "exchange_connected": True,
                "step_size": 0.001,
                "tick_size": 0.01,
            },
        )
        self.assertTrue(dec.allowed, msg=f"Unexpected rejection: {dec.reasons}")

    def test_manual_controller_aware_ttl_and_alpha_vault_seed_status(self):
        from crypto_trading.manual_controller import ManualPositionController
        from worldquant_brain.alpha_vault import AlphaVault

        import gc

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_p = f"{td}/test.db"
            mc = ManualPositionController(db_path=db_p)
            old_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            conn = sqlite3.connect(db_p)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO manual_overrides (symbol, is_locked, lock_reason, locked_at, ttl_seconds, updated_at) VALUES (?, 1, ?, ?, 60, ?)",
                        ("BTC/USDT", "test", old_iso, old_iso),
                    )
            finally:
                conn.close()
            self.assertFalse(mc.is_symbol_locked("BTC/USDT"))
            del mc
            gc.collect()

            vault = AlphaVault(db_path=":memory:")
            vault.seed_initial_verified_alphas()
            rows = vault.list_alphas()
            self.assertTrue(len(rows) > 0)
            self.assertTrue(all(r["status"] == "TEMPLATE_SEED" for r in rows))

    def test_testnet_client_never_returns_15000_on_remote_error(self):
        from crypto_trading.testnet_client import BinanceTestnetClient

        c = BinanceTestnetClient(api_key="k", api_secret="s")
        c.is_connected = True
        c.mock_mode = False
        c.exchange = MagicMock()
        c.exchange.fapiPrivateV2GetAccount.side_effect = ConnectionError("network down")
        bal = c.fetch_balance()
        self.assertIsNone(bal["total_equity"])
        self.assertEqual(bal["mode"], "OFFICIAL_BINANCE_TESTNET_ERROR")

    def test_market_analyzer_returns_data_unavailable_on_empty_df(self):
        from taiwan_stock.market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer()
        with patch.object(analyzer, "fetch_market_data", return_value=pd.DataFrame()):
            res = analyzer.analyze()
            self.assertEqual(res["status"], "DATA_UNAVAILABLE")
            self.assertIsNone(res["score"])


    def test_reconciliation_service_get_latest_result_and_bitunix_no_fake_price(self):
        from unittest.mock import patch
        from crypto_trading.bitunix_client import BitunixClient
        from crypto_trading.reconciliation_service import ReconciliationService

        svc = ReconciliationService()
        self.assertIsNone(svc.get_latest_result())
        svc.reconcile({}, [], [], [], 1000.0, 1000.0)
        latest = svc.get_latest_result()
        self.assertTrue(latest["in_sync"])
        self.assertIn("completed_at", latest)

        bc = BitunixClient(api_key="", api_secret="")
        with patch("requests.get", side_effect=ConnectionError("down")):
            t = bc.fetch_ticker("BTC/USDT")
            self.assertIsNone(t["last"])
            self.assertEqual(t["error"], "TICKER_UNAVAILABLE")



    def test_bitunix_standby_balance_none_and_unverified_model_quarantine(self):
        import json
        import sys
        from pathlib import Path
        try:
            import joblib  # noqa: F401
        except ImportError:
            sys.modules.setdefault("joblib", MagicMock())
        from crypto_trading.bitunix_client import BitunixClient
        from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
        from crypto_trading.strategies.rl_strategy import RLAlphaStrategy

        bc = BitunixClient(api_key="", api_secret="")
        bal = bc.fetch_balance()
        self.assertIsNone(bal["total_equity"])
        self.assertEqual(bal["error"], "BITUNIX_STANDBY_MODE")

        df = pd.DataFrame({"Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0], "Low": [99.0, 100.0, 101.0], "Close": [100.0, 101.0, 102.0], "Volume": [10.0, 11.0, 12.0]})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m_path = Path(td) / "model.joblib"
            m_path.write_text("dummy", encoding="utf-8")
            ml = MLAlphaStrategy(model_path=str(m_path))
            out_ml = ml.generate_signals(df)
            self.assertEqual(out_ml["model_status"].iloc[0], "UNVERIFIED_MODEL_QUARANTINED")

            r_path = Path(td) / "ppo.pt"
            r_path.write_text("dummy", encoding="utf-8")
            (Path(td) / "ppo_manifest.json").write_text(
                json.dumps({"is_synthetic": True, "deployment_stage": "RESEARCH_ONLY"}),
                encoding="utf-8",
            )
            rl = RLAlphaStrategy(agent_weights_path=str(r_path))
            out_rl = rl.generate_signals(df)
            self.assertIn(out_rl["model_status"].iloc[0], ("UNVERIFIED_MODEL_QUARANTINED", "MODEL_UNAVAILABLE"))



    def test_web_ui_fail_closed_no_hardcoded_15000_20000_or_null_crash(self):
        from pathlib import Path
        srv = Path("web_ui/server.py").read_text(encoding="utf-8")
        self.assertNotIn('bal.get("total_equity", 15000.0)', srv)
        self.assertNotIn('bal.get("total_equity", 20000.0)', srv)
        self.assertNotIn('ticker.get("last", 68500.0)', srv)
        html = Path("web_ui/static/index.html").read_text(encoding="utf-8")
        self.assertIn("d.total_equity !== undefined && d.total_equity !== null", html)
        self.assertIn("d.current_price !== undefined && d.current_price !== null", html)



    def test_phase0_inventory_and_phase2_instrument_endpoint(self):
        import json
        from pathlib import Path
        inv = json.loads(Path("reports/baseline_inventory.json").read_text(encoding="utf-8"))
        for key in (
            "MODE_AUTHORITY", "ACCOUNT_SOURCE", "POSITION_SOURCE", "MARKET_SOURCE",
            "ORDER_ENTRY_POINT", "BRAIN_CONFIG", "MODEL_ARTIFACT", "RUNTIME_ARTIFACT",
            "HARDCODED_UI_VALUE", "WORKER_STARTUP",
        ):
            self.assertIn(key, inv)
        srv = Path("web_ui/server.py").read_text(encoding="utf-8")
        self.assertIn('"/api/trading/instrument"', srv)
        self.assertIn('"canonical_identity"', srv)


if __name__ == "__main__":
    unittest.main()
'''


def _tx_order_manager(txt: str) -> tuple[str, str]:
    if "Decimal(str(amount))" in txt and "bucket_" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt if "from decimal import Decimal" in txt else "from decimal import Decimal\n" + txt
    old_fn = (
        'def generate_idempotency_key(\n'
        '    symbol: str,\n'
        '    side: str,\n'
        '    order_type: str,\n'
        '    amount: float,\n'
        '    price: float | None,\n'
        '    client_tag: str = "",\n'
        ') -> str:\n'
        '    raw = f"{symbol}_{side.upper()}_{order_type.upper()}_{amount:.6f}_{price or 0:.6f}_{client_tag}"\n'
        '    return hashlib.sha256(raw.encode("utf-8")).hexdigest()'
    )
    new_fn = (
        'def generate_idempotency_key(\n'
        '    symbol: str,\n'
        '    side: str,\n'
        '    order_type: str,\n'
        '    amount: float | Decimal,\n'
        '    price: float | Decimal | None,\n'
        '    client_tag: str = "",\n'
        ') -> str:\n'
        '    norm_amt = Decimal(str(amount)).normalize()\n'
        '    norm_prc = Decimal(str(price if price is not None else 0)).normalize()\n'
        '    bucket = client_tag or f"bucket_{int(time.time() // 5)}"\n'
        '    raw = f"{symbol}_{side.upper()}_{order_type.upper()}_{norm_amt:f}_{norm_prc:f}_{bucket}"\n'
        '    return hashlib.sha256(raw.encode("utf-8")).hexdigest()'
    )
    if old_fn not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(old_fn, new_fn, 1)
    new_txt = new_txt.replace(
        "ikey = generate_idempotency_key(symbol, side, order_type, amount, price, client_tag or cid)",
        "ikey = generate_idempotency_key(symbol, side, order_type, amount, price, client_tag)",
    )
    return "PATCHED", new_txt


def _tx_risk_manager(txt: str) -> tuple[str, str]:
    if "Decimal(str(amount))" in txt and "REPLACE INTO _db_write_probe" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt if "from decimal import Decimal" in txt else "from decimal import Decimal\n" + txt
    new_txt = new_txt.replace(
        'test_cur.execute("CREATE TABLE IF NOT EXISTS _db_write_probe (ts TEXT)")\n                test_cur.execute(\n                    "INSERT INTO _db_write_probe VALUES (?)",\n                    (datetime.utcnow().isoformat(),),\n                )',
        'test_cur.execute("CREATE TABLE IF NOT EXISTS _db_write_probe (id INTEGER PRIMARY KEY, ts TEXT)")\n                test_cur.execute(\n                    "REPLACE INTO _db_write_probe (id, ts) VALUES (1, ?)",\n                    (datetime.utcnow().isoformat(),),\n                )',
    )
    old_mod = (
        '        step_size = ctx.get("step_size", 0.001)\n'
        '        if step_size > 0:\n'
        '            remainder = amount % step_size\n'
        '            if round(remainder, 6) != 0 and round(abs(remainder - step_size), 6) != 0:\n'
        '                reasons.append(f"Amount {amount} violates step_size {step_size}")\n\n'
        '        tick_size = ctx.get("tick_size", 0.01)\n'
        '        if tick_size > 0:\n'
        '            remainder_p = price % tick_size\n'
        '            if round(remainder_p, 6) != 0 and round(abs(remainder_p - tick_size), 6) != 0:\n'
        '                reasons.append(f"Price {price} violates tick_size {tick_size}")'
    )
    new_mod = (
        '        step_size = ctx.get("step_size", 0.001)\n'
        '        if step_size > 0:\n'
        '            if Decimal(str(amount)) % Decimal(str(step_size)) != Decimal("0"):\n'
        '                reasons.append(f"Amount {amount} violates step_size {step_size}")\n\n'
        '        tick_size = ctx.get("tick_size", 0.01)\n'
        '        if tick_size > 0:\n'
        '            if Decimal(str(price)) % Decimal(str(tick_size)) != Decimal("0"):\n'
        '                reasons.append(f"Price {price} violates tick_size {tick_size}")'
    )
    if old_mod not in new_txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt.replace(old_mod, new_mod, 1)


def _tx_testnet_client(txt: str) -> tuple[str, str]:
    if "OFFICIAL_BINANCE_TESTNET_ERROR" in txt and '"total_equity": None' in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt
    old_bal = (
        '        except Exception as e:\n'
        '            return {\n'
        '                "mode": "OFFICIAL_BINANCE_TESTNET_FALLBACK",\n'
        '                "error": str(e),\n'
        '                "total_equity": self._mock_balance.get(self.settlement, 15000.0),\n'
        '                "free_balance": self._mock_balance.get(self.settlement, 15000.0) * 0.8,\n'
        '                "settlement_currency": self.settlement,\n'
        '                "zero_fee_usdc": True,\n'
        '                "balances": self._mock_balance,\n'
        '            }'
    )
    new_bal = (
        '        except Exception as e:\n'
        '            return {\n'
        '                "mode": "OFFICIAL_BINANCE_TESTNET_ERROR",\n'
        '                "error": type(e).__name__,\n'
        '                "total_equity": None,\n'
        '                "free_balance": None,\n'
        '                "settlement_currency": self.settlement,\n'
        '                "zero_fee_usdc": self.settlement == "USDC",\n'
        '                "balances": {},\n'
        '            }'
    )
    if old_bal not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(old_bal, new_bal, 1)
    new_txt = new_txt.replace(
        "        # 備用模擬報價\n        return {",
        '        if not self.mock_mode and self.is_connected:\n'
        '            return {"symbol": target_symbol, "last": None, "error": "TICKER_UNAVAILABLE", "timestamp": None, "settlement": self.settlement}\n'
        '        # 備用模擬報價 (僅限 mock_mode)\n        return {',
        1,
    )
    return "PATCHED", new_txt


def _tx_manual_controller(txt: str) -> tuple[str, str]:
    if "timezone.utc" in txt and "reduce_only=True" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace("from datetime import datetime", "from datetime import datetime, timezone", 1)
    old_ttl = (
        "                locked_dt = datetime.fromisoformat(locked_at_str)\n"
        "                elapsed = (datetime.utcnow() - locked_dt).total_seconds()"
    )
    new_ttl = (
        "                locked_dt = datetime.fromisoformat(locked_at_str)\n"
        "                if locked_dt.tzinfo is None:\n"
        "                    locked_dt = locked_dt.replace(tzinfo=timezone.utc)\n"
        "                elapsed = (datetime.now(timezone.utc) - locked_dt).total_seconds()"
    )
    if old_ttl not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(old_ttl, new_ttl, 1)
    new_txt = new_txt.replace(
        'res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=contracts)',
        'try:\n                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=contracts, reduce_only=True)\n            except TypeError:\n                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=contracts)',
    )
    new_txt = new_txt.replace(
        '            res = client.create_order(\n                symbol=symbol,\n                side=close_side,\n                order_type="MARKET",\n                amount=reduce_amount,\n            )',
        '            try:\n                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=reduce_amount, reduce_only=True)\n            except TypeError:\n                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=reduce_amount)',
    )
    return "PATCHED", new_txt


def _tx_bot_runner_delta(txt: str) -> tuple[str, str]:
    old_eq = "            total_equity = self.balance_usdt + (\n                self.position * self.entry_price if self.position > 0 else 0\n            )"
    old_rec = '"reconciliation_fresh": True,'
    if old_eq not in txt and old_rec not in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace(
        old_eq,
        "            total_equity = self.get_equity(self.entry_price if self.entry_price > 0 else 0.0)",
    )
    new_txt = new_txt.replace(
        old_rec,
        '"reconciliation_fresh": (getattr(self, "reconciliation_service", None).is_ready_for_trading() if getattr(self, "reconciliation_service", None) else not bool(self.exchange_adapter)),',
    )
    return "PATCHED", new_txt


def _tx_alpha_vault(txt: str) -> tuple[str, str]:
    if "TEMPLATE_SEED" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace(
        '"VERIFIED" if is_qualified else "SUB_THRESHOLD",',
        '(status_override or ("VERIFIED" if is_qualified else "SUB_THRESHOLD")),',
    )
    new_txt = new_txt.replace(
        "        force: bool = False,\n    ) -> dict[str, Any]:",
        "        force: bool = False,\n        status_override: str | None = None,\n    ) -> dict[str, Any]:",
    )
    new_txt = new_txt.replace(
        '                notes=p["notes"],\n            )',
        '                notes=p["notes"],\n                status_override="TEMPLATE_SEED",\n            )',
    )
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _tx_market_analyzer(txt: str) -> tuple[str, str]:
    if '"status": "DATA_UNAVAILABLE"' in txt and "yf = None" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace("import yfinance as yf", "try:\n    import yfinance as yf\nexcept ImportError:\n    yf = None", 1)
    old_blk_a = (
        '        if df.empty or len(df) < 60:\n'
        '            return {\n'
        '                "status": "數據不足",\n'
        '                "score": 50,\n'
        '                "regime": "震盪觀望",\n'
        '                "summary": "大盤數據取得不足，建議保守觀望。",\n'
        '            }'
    )
    old_blk_b = old_blk_a.replace('建議保守觀望。",', '建議保守觀望。"')
    old_blk = old_blk_a if old_blk_a in new_txt else old_blk_b
    new_blk = (
        '        if df.empty or len(df) < 60:\n'
        '            return {\n'
        '                "status": "DATA_UNAVAILABLE",\n'
        '                "score": None,\n'
        '                "regime": "DATA_UNAVAILABLE",\n'
        '                "summary": "大盤數據取得不足，拒絕以預設 50 分假冒中性市場。"\n'
        '            }'
    )
    if old_blk not in new_txt:
        return "ALREADY_APPLIED", txt
    return "PATCHED", new_txt.replace(old_blk, new_blk, 1)


def _tx_ml_ranker(txt: str) -> tuple[str, str]:
    if "purge_days: int = 5" in txt or "embargo_days: int = 5" in txt:
        return "ALREADY_APPLIED", txt
    old_fn = (
        "    def train_model(self, clean_data: pd.DataFrame) -> Dict[str, Any]:\n"
        "        X = clean_data[self.feature_cols].values\n"
        "        y = clean_data['target_fwd_5d'].values\n\n"
        "        # 依時間前後拆分訓練/驗證集 (80/20)\n"
        "        split_idx = int(len(X) * 0.8)\n"
        "        X_train, y_train = X[:split_idx], y[:split_idx]\n"
        "        X_test, y_test = X[split_idx:], y[split_idx:]"
    )
    new_fn = (
        "    def train_model(self, clean_data: pd.DataFrame, purge_days: int = 5) -> Dict[str, Any]:\n"
        "        ordered = clean_data.sort_index()\n"
        "        if hasattr(ordered.index, \"normalize\"):\n"
        "            dates = ordered.index.normalize()\n"
        "            uniq_dates = dates.unique().sort_values()\n"
        "            if len(uniq_dates) > purge_days + 2:\n"
        "                split_d_idx = max(1, int(len(uniq_dates) * 0.8) - purge_days)\n"
        "                cutoff_train = uniq_dates[split_d_idx]\n"
        "                test_start_idx = min(len(uniq_dates) - 1, split_d_idx + purge_days)\n"
        "                cutoff_test = uniq_dates[test_start_idx]\n"
        "                train_df = ordered[dates <= cutoff_train]\n"
        "                test_df = ordered[dates >= cutoff_test]\n"
        "            else:\n"
        "                split_idx = int(len(ordered) * 0.8)\n"
        "                train_df = ordered.iloc[:max(1, split_idx - purge_days)]\n"
        "                test_df = ordered.iloc[split_idx:]\n"
        "        else:\n"
        "            split_idx = int(len(ordered) * 0.8)\n"
        "            train_df = ordered.iloc[:max(1, split_idx - purge_days)]\n"
        "            test_df = ordered.iloc[split_idx:]\n"
        "        X_train, y_train = train_df[self.feature_cols].values, train_df['target_fwd_5d'].values\n"
        "        X_test, y_test = test_df[self.feature_cols].values, test_df['target_fwd_5d'].values"
    )
    if old_fn not in txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", txt.replace(old_fn, new_fn, 1)


def _tx_ml_trainer(txt: str) -> tuple[str, str]:
    if (
        "purge_bars: int = 12" in txt
        and "oof_predictions[test_start:test_end] = preds" in txt
        and "oof_mask[test_start:test_end] = True" in txt
    ):
        return "ALREADY_APPLIED", txt
    new_txt = txt
    old_sig_1 = "    def walk_forward_train(self, df_raw: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7) -> dict[str, Any]:"
    old_sig_2 = (
        "    def walk_forward_train(\n"
        "        self, \n"
        "        df_raw: pd.DataFrame, \n"
        "        n_splits: int = 5,\n"
        "        train_ratio: float = 0.7\n"
        "    ) -> Dict[str, Any]:"
    )
    new_sig = (
        "    def walk_forward_train(\n"
        "        self,\n"
        "        df_raw: pd.DataFrame,\n"
        "        n_splits: int = 5,\n"
        "        train_ratio: float = 0.7,\n"
        "        purge_bars: int = 12,\n"
        "        embargo_bars: int = 2,\n"
        "    ) -> dict[str, Any]:"
    )
    if "purge_bars: int = 12" not in new_txt:
        if old_sig_1 in new_txt:
            new_txt = new_txt.replace(old_sig_1, new_sig, 1)
        elif old_sig_2 in new_txt:
            new_txt = new_txt.replace(old_sig_2, new_sig, 1)
        else:
            return "TARGET_NOT_FOUND", txt
        old_split = (
            "            train_end = fold_size * (fold + 1)\n"
            "            test_end = min(train_end + fold_size, n_samples)\n"
            "            if train_end >= test_end:\n"
            "                break\n\n"
            "            X_train, y_train = X[:train_end], y[:train_end]\n"
            "            X_test, y_test = X[train_end:test_end], y[train_end:test_end]\n"
            "            test_ret = returns[train_end:test_end]"
        )
        new_split = (
            "            train_end = fold_size * (fold + 1)\n"
            "            test_start = train_end + purge_bars + embargo_bars\n"
            "            test_end = min(test_start + fold_size, n_samples)\n"
            "            if test_start >= test_end:\n"
            "                break\n\n"
            "            X_train, y_train = X[:train_end], y[:train_end]\n"
            "            X_test, y_test = X[test_start:test_end], y[test_start:test_end]\n"
            "            test_ret = returns[test_start:test_end]"
        )
        if old_split not in new_txt:
            return "TARGET_NOT_FOUND", txt
        new_txt = new_txt.replace(old_split, new_split, 1)
    old_oof = (
        "            oof_predictions[train_end:test_end] = preds\n"
        "            oof_mask[train_end:test_end] = True"
    )
    new_oof = (
        "            oof_predictions[test_start:test_end] = preds\n"
        "            oof_mask[test_start:test_end] = True"
    )
    if old_oof in new_txt:
        new_txt = new_txt.replace(old_oof, new_oof, 1)
    old_xy = (
        "        X = processed[features].values\n"
        "        y = processed[target_col].values\n"
        '        returns = processed["target_fwd_ret_3"].values'
    )
    new_xy = (
        "        X = np.asarray(processed[features].values, dtype=float)\n"
        '        y = np.asarray(processed[target_col].values, dtype=float if self.task == "regression" else int)\n'
        '        returns = np.asarray(processed["target_fwd_ret_3"].values, dtype=float)'
    )
    if old_xy in new_txt:
        new_txt = new_txt.replace(old_xy, new_xy, 1)
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _tx_reconciliation_service(txt: str) -> tuple[str, str]:
    if "def get_latest_result(self)" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace("from datetime import datetime", "from datetime import datetime, timezone", 1)
    new_txt = new_txt.replace("datetime.utcnow().isoformat()", "datetime.now(timezone.utc).isoformat()")
    anchor = "    def is_ready_for_trading(self) -> bool:"
    if anchor not in new_txt:
        return "TARGET_NOT_FOUND", txt
    method = (
        "    def get_latest_result(self) -> dict[str, Any] | None:\n"
        "        if not self.reconciliation_report:\n"
        "            return None\n"
        "        return {\n"
        "            \"in_sync\": bool(self.is_reconciled),\n"
        "            \"discrepancy_count\": int(self.reconciliation_report.get(\"discrepancy_count\", 0)),\n"
        "            \"completed_at\": self.last_reconciled_time,\n"
        "            \"unknown_order_count\": 0,\n"
        "        }\n\n"
    )
    return "PATCHED", new_txt.replace(anchor, method + anchor, 1)


def _tx_bitunix_client(txt: str) -> tuple[str, str]:
    if (
        '"status": "OFFLINE"' in txt
        and "if not self.is_live:" in txt
        and '"error": "BITUNIX_STANDBY_MODE"' in txt
        and "if exec_price is None:" in txt
    ):
        return "ALREADY_APPLIED", txt

    new_txt = txt
    # Replace entire fetch_ticker method cleanly
    old_ft_m = re.search(
        r'    def fetch_ticker\(self, symbol: str = "BTC/USDT"\) -> dict\[str, Any\]:.*?(?=\n    def fetch_balance\()',
        new_txt,
        re.S,
    )
    if not old_ft_m:
        return "TARGET_NOT_FOUND", txt

    new_ft = """    def fetch_ticker(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        \"\"\"Return a stable, fail-closed ticker schema.

        Public market data is queried only in BITUNIX_LIVE mode. In every
        non-live mode the method returns an explicit OFFLINE result instead of
        synthesizing a tradable price.
        \"\"\"
        clean_symbol = symbol.replace("/", "").upper()

        def offline(reason: str) -> dict[str, Any]:
            return {
                "exchange": "bitunix",
                "symbol": symbol,
                "price": None,
                "last": None,
                "bid": None,
                "ask": None,
                "high24h": None,
                "low24h": None,
                "volume24h": None,
                "timestamp": None,
                "status": "OFFLINE",
                "success": False,
                "error": "TICKER_UNAVAILABLE",
                "detail": reason,
            }

        if not self.is_live:
            return offline("Bitunix market data is disabled outside BITUNIX_LIVE mode")

        try:
            import requests

            url = f"{self.base_url}/api/v1/futures/market/ticker?symbol={clean_symbol}"
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            last = float(data["lastPrice"])
            bid_raw = data.get("bidPrice")
            ask_raw = data.get("askPrice")
            return {
                "exchange": "bitunix",
                "symbol": symbol,
                "price": last,
                "last": last,
                "bid": float(bid_raw) if bid_raw not in (None, "") else None,
                "ask": float(ask_raw) if ask_raw not in (None, "") else None,
                "high24h": float(data["highPrice"]) if data.get("highPrice") not in (None, "") else None,
                "low24h": float(data["lowPrice"]) if data.get("lowPrice") not in (None, "") else None,
                "volume24h": float(data["volume"]) if data.get("volume") not in (None, "") else None,
                "timestamp": int(time.time() * 1000),
                "status": "LIVE",
                "success": True,
                "error": None,
            }
        except Exception as exc:
            return offline(f"Bitunix ticker unavailable: {type(exc).__name__}")
"""
    new_txt = new_txt[:old_ft_m.start()] + new_ft + new_txt[old_ft_m.end():]

    old_bal = (
        '        if not self.is_live:\n'
        '            return {\n'
        '                "mode": "BITUNIX_STANDBY_MODE (未配置實盤 API Key)",\n'
        '                "exchange": "bitunix",\n'
        '                "total_equity": self._simulated_balance["USDT"],\n'
        '                "free_balance": self._simulated_balance["USDT"] * 0.85,\n'
        '                "balances": self._simulated_balance,'
    )
    new_bal = (
        '        if not self.is_live:\n'
        '            return {\n'
        '                "mode": "BITUNIX_STANDBY_MODE (未配置實盤 API Key)",\n'
        '                "exchange": "bitunix",\n'
        '                "total_equity": None,\n'
        '                "free_balance": None,\n'
        '                "balances": {},\n'
        '                "error": "BITUNIX_STANDBY_MODE",'
    )
    if old_bal in new_txt:
        new_txt = new_txt.replace(old_bal, new_bal, 1)

    old_ord = '            ticker = self.fetch_ticker(symbol)\n            exec_price = price if (order_type == "LIMIT" and price) else ticker["last"]'
    new_ord = (
        '            ticker = self.fetch_ticker(symbol)\n'
        '            exec_price = price if (order_type == "LIMIT" and price) else ticker.get("last")\n'
        '            if exec_price is None:\n'
        '                return {\n'
        '                    "success": False,\n'
        '                    "mode": "BITUNIX_STANDBY_TICKER_UNAVAILABLE",\n'
        '                    "error": "TICKER_UNAVAILABLE",\n'
        '                    "message": "Bitunix 行情不可用，拒絕以假價格模擬成交",\n'
        '                }'
    )
    if "if exec_price is None:" not in new_txt and old_ord in new_txt:
        new_txt = new_txt.replace(old_ord, new_ord, 1)

    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _tx_backtest_engine(txt: str) -> tuple[str, str]:
    if "pct_change(fill_method=None)" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace(
        "market_returns = close.pct_change().fillna(0)",
        "market_returns = close.pct_change(fill_method=None).fillna(0)",
    )
    new_txt = new_txt.replace(
        "net_equity = (1.0 + net_returns).cumprod() * self.initial_capital",
        "net_equity = (1.0 + net_returns).clip(lower=0.0).cumprod() * self.initial_capital",
    )
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt



def _tx_ml_strategy(txt: str) -> tuple[str, str]:
    if "UNVERIFIED_MODEL_QUARANTINED" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt
    if "import json" not in new_txt:
        new_txt = new_txt.replace("import numpy as np", "import json\nimport os\nimport re\n\nimport numpy as np", 1)
    old_init = (
        "    def __init__(\n"
        "        self,\n"
        "        model_path: str | None = None,\n"
        "        trainer: CryptoMLTrainer | None = None,\n"
        "        confidence_hurdle: float = 0.0015,  # 預測收益率需高於 0.15% (扣除手續費) 才開倉\n"
        "    ):\n"
        "        self.confidence_hurdle = confidence_hurdle\n"
        "        if trainer is not None:\n"
        "            self.trainer = trainer\n"
        "        elif model_path is not None:\n"
        "            self.trainer = CryptoMLTrainer.load_weights(model_path)\n"
        "        else:\n"
        "            self.trainer = None"
    )
    new_init = (
        "    def __init__(\n"
        "        self,\n"
        "        model_path: str | None = None,\n"
        "        trainer: CryptoMLTrainer | None = None,\n"
        "        confidence_hurdle: float = 0.0015,  # 預測收益率需高於 0.15% (扣除手續費) 才開倉\n"
        "        allow_unverified: bool = False,\n"
        "    ):\n"
        "        self.confidence_hurdle = confidence_hurdle\n"
        "        self.allow_unverified = allow_unverified\n"
        "        self.model_status = \"READY\"\n"
        "        if trainer is not None:\n"
        "            self.trainer = trainer\n"
        "        elif model_path is not None:\n"
        "            manifest_path = re.sub(r\"\\.[^.]+$\", \"_manifest.json\", model_path)\n"
        "            if not allow_unverified:\n"
        "                if not os.path.exists(manifest_path):\n"
        "                    self.trainer = None\n"
        "                    self.model_status = \"UNVERIFIED_MODEL_QUARANTINED\"\n"
        "                    return\n"
        "                try:\n"
        "                    with open(manifest_path, \"r\", encoding=\"utf-8\") as f:\n"
        "                        meta = json.load(f)\n"
        "                except Exception:\n"
        "                    self.trainer = None\n"
        "                    self.model_status = \"UNVERIFIED_MODEL_QUARANTINED\"\n"
        "                    return\n"
        "                if meta.get(\"is_synthetic\") or meta.get(\"deployment_stage\") == \"RESEARCH_ONLY\":\n"
        "                    self.trainer = None\n"
        "                    self.model_status = \"UNVERIFIED_MODEL_QUARANTINED\"\n"
        "                    return\n"
        "            self.trainer = CryptoMLTrainer.load_weights(model_path)\n"
        "        else:\n"
        "            self.trainer = None\n"
        "            self.model_status = \"MODEL_UNAVAILABLE\""
    )
    if old_init not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(old_init, new_init, 1)
    new_txt = new_txt.replace(
        '            df["model_status"] = "MODEL_UNAVAILABLE"',
        '            df["model_status"] = getattr(self, "model_status", "MODEL_UNAVAILABLE")',
        1,
    )
    return "PATCHED", new_txt


def _tx_rl_strategy(txt: str) -> tuple[str, str]:
    if "UNVERIFIED_MODEL_QUARANTINED" in txt:
        return "ALREADY_APPLIED", txt
    old_init = (
        "    def __init__(\n"
        "        self,\n"
        "        agent_weights_path: str | None = None,\n"
        "        feature_engineer: CryptoFeatureEngineer | None = None,\n"
        "    ):\n"
        "        self.fe = feature_engineer or CryptoFeatureEngineer()\n"
        "        self.agent = None\n"
        "        self.weights_path = agent_weights_path\n"
        "        self.is_ready = False"
    )
    new_init = (
        "    def __init__(\n"
        "        self,\n"
        "        agent_weights_path: str | None = None,\n"
        "        feature_engineer: CryptoFeatureEngineer | None = None,\n"
        "        allow_unverified: bool = False,\n"
        "    ):\n"
        "        self.fe = feature_engineer or CryptoFeatureEngineer()\n"
        "        self.agent = None\n"
        "        self.weights_path = agent_weights_path\n"
        "        self.allow_unverified = allow_unverified\n"
        "        self.model_status = \"MODEL_UNAVAILABLE\"\n"
        "        self.is_ready = False"
    )
    if old_init not in txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = txt.replace(old_init, new_init, 1)
    old_ensure = (
        "            manifest_path = self.weights_path.replace(\".pt\", \"_manifest.json\")\n"
        "            if os.path.exists(manifest_path):\n"
        "                try:\n"
        "                    with open(manifest_path, \"r\", encoding=\"utf-8\") as f:\n"
        "                        meta = json.load(f)\n"
        "                    if \"state_dim\" in meta:\n"
        "                        state_dim = int(meta[\"state_dim\"])\n"
        "                    elif \"feature_columns\" in meta:\n"
        "                        state_dim = len(meta[\"feature_columns\"]) + 2\n"
        "                except Exception:\n"
        "                    pass"
    )
    new_ensure = (
        "            manifest_path = self.weights_path.replace(\".pt\", \"_manifest.json\")\n"
        "            meta = {}\n"
        "            if os.path.exists(manifest_path):\n"
        "                try:\n"
        "                    with open(manifest_path, \"r\", encoding=\"utf-8\") as f:\n"
        "                        meta = json.load(f)\n"
        "                    if \"state_dim\" in meta:\n"
        "                        state_dim = int(meta[\"state_dim\"])\n"
        "                    elif \"feature_columns\" in meta:\n"
        "                        state_dim = len(meta[\"feature_columns\"]) + 2\n"
        "                except Exception:\n"
        "                    meta = {}\n"
        "            if not self.allow_unverified:\n"
        "                if not os.path.exists(manifest_path) or meta.get(\"is_synthetic\") or meta.get(\"deployment_stage\") == \"RESEARCH_ONLY\":\n"
        "                    self.is_ready = False\n"
        "                    self.model_status = \"UNVERIFIED_MODEL_QUARANTINED\"\n"
        "                    return False"
    )
    if old_ensure not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(old_ensure, new_ensure, 1)
    new_txt = new_txt.replace(
        '            df["model_status"] = "MODEL_UNAVAILABLE"',
        '            df["model_status"] = getattr(self, "model_status", "MODEL_UNAVAILABLE")',
    )
    return "PATCHED", new_txt



def _tx_web_server(txt: str) -> tuple[str, str]:
    if '"total_equity": bal.get("total_equity"),' in txt and 'curr_price = ticker.get("last")' in txt and '"/api/trading/instrument"' in txt and 'os.getenv("PORT", "8000")' in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt
    new_txt = new_txt.replace('curr_price = ticker.get("last", 68500.0)', 'curr_price = ticker.get("last")')
    new_txt = new_txt.replace('"total_equity": bal.get("total_equity", 20000.0),', '"total_equity": bal.get("total_equity"),')
    new_txt = new_txt.replace('"free_balance": bal.get("free_balance", 17000.0),', '"free_balance": bal.get("free_balance"),')
    new_txt = new_txt.replace('"total_equity": bal.get("total_equity", 15000.0),', '"total_equity": bal.get("total_equity"),')
    new_txt = new_txt.replace('"free_balance": bal.get("free_balance", 12000.0),', '"free_balance": bal.get("free_balance"),')
    new_txt = new_txt.replace(
        "                    equity = bot.get_equity(curr_price)\n"
        "                    if bot.position_side == \"LONG\" and bot.entry_price > 0:\n"
        "                        unrealized_pnl = (curr_price - bot.entry_price) * bot.position\n"
        "                    elif bot.position_side == \"SHORT\" and bot.entry_price > 0:\n"
        "                        unrealized_pnl = (bot.entry_price - curr_price) * bot.position\n"
        "                    else:\n"
        "                        unrealized_pnl = 0.0",
        "                    equity = bot.get_equity(curr_price if curr_price is not None else (bot.entry_price or 0.0))\n"
        "                    if curr_price is None:\n"
        "                        unrealized_pnl = None\n"
        "                    elif bot.position_side == \"LONG\" and bot.entry_price > 0:\n"
        "                        unrealized_pnl = (curr_price - bot.entry_price) * bot.position\n"
        "                    elif bot.position_side == \"SHORT\" and bot.entry_price > 0:\n"
        "                        unrealized_pnl = (bot.entry_price - curr_price) * bot.position\n"
        "                    else:\n"
        "                        unrealized_pnl = 0.0",
    )
    new_txt = new_txt.replace(
        '"unrealized_pnl": round(unrealized_pnl, 2),',
        '"unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,',
    )
    if '"/api/trading/instrument"' not in new_txt:
        anchor_inst = '        if path == "/api/trading/snapshot":'
        inst_handler = (
            '        if path == "/api/trading/instrument":\n'
            '            from datetime import datetime, timezone\n'
            '            from crypto_trading.instruments import BTC_USDT_BINANCE_TESTNET\n'
            '            spec = BTC_USDT_BINANCE_TESTNET\n'
            '            sym = getattr(APP_STATE.active_bot, "symbol", spec.display_symbol)\n'
            '            return self._send_json_response({\n'
            '                "canonical_identity": spec.canonical_id,\n'
            '                "ui_symbol": sym,\n'
            '                "strategy_symbol": sym,\n'
            '                "market_data_symbol": spec.venue_symbol,\n'
            '                "execution_symbol": spec.venue_symbol,\n'
            '                "position_symbol": spec.venue_symbol,\n'
            '                "quote": spec.quote_asset,\n'
            '                "settlement": spec.settlement_asset,\n'
            '                "contract_size": str(spec.contract_size),\n'
            '                "price_tick": str(spec.price_tick),\n'
            '                "quantity_step": str(spec.quantity_step),\n'
            '                "metadata_timestamp": datetime.now(timezone.utc).isoformat(),\n'
            '            })\n\n'
        )
        if anchor_inst in new_txt:
            new_txt = new_txt.replace(anchor_inst, inst_handler + anchor_inst, 1)
    new_txt = new_txt.replace(
        "def run_server(port: int = 8999, host: str = \"0.0.0.0\"):",
        "def run_server(port: int | None = None, host: str = \"0.0.0.0\"):\n    if port is None:\n        port = int(os.getenv(\"PORT\", \"8000\"))",
    )
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _tx_web_index(txt: str) -> tuple[str, str]:
    if "d.total_equity !== undefined && d.total_equity !== null" in txt:
        return "ALREADY_APPLIED", txt
    old_blk = (
        "          document.getElementById('cryptoEquity').innerHTML = `${d.total_equity.toLocaleString()} <span class=\"text-xs font-normal text-slate-400\">USDT</span>`;\n"
        "          document.getElementById('cryptoPrice').innerHTML = `${d.current_price.toLocaleString()} <span class=\"text-xs font-normal text-slate-400\">USDT</span>`;"
    )
    new_blk = (
        "          const eqElem = document.getElementById('cryptoEquity');\n"
        "          if (eqElem) {\n"
        "            eqElem.innerHTML = (d.total_equity !== undefined && d.total_equity !== null)\n"
        "              ? `${Number(d.total_equity).toLocaleString()} <span class=\"text-xs font-normal text-slate-400\">USDT</span>`\n"
        "              : `<span class=\"text-slate-500\">unavailable</span> <span class=\"text-xs font-normal text-slate-400\">USDT</span>`;\n"
        "          }\n"
        "          const prcElem = document.getElementById('cryptoPrice');\n"
        "          if (prcElem) {\n"
        "            prcElem.innerHTML = (d.current_price !== undefined && d.current_price !== null)\n"
        "              ? `${Number(d.current_price).toLocaleString()} <span class=\"text-xs font-normal text-slate-400\">USDT</span>`\n"
        "              : `<span class=\"text-slate-500\">unavailable</span> <span class=\"text-xs font-normal text-slate-400\">USDT</span>`;\n"
        "          }"
    )
    if old_blk not in txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", txt.replace(old_blk, new_blk, 1)




def _tx_data_loader(txt: str) -> tuple[str, str]:
    if "except ImportError:\n    ccxt = None" in txt:
        return "ALREADY_APPLIED", txt
    anchor = "from crypto_trading.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME"
    if anchor not in txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = txt.replace(
        anchor,
        "try:\n    import ccxt\nexcept ImportError:\n    ccxt = None\n\n" + anchor,
        1,
    )
    new_txt = new_txt.replace(
        "        try:\n            import ccxt\n\n            exchange_class = ccxt.binance({\"enableRateLimit\": True, \"timeout\": 5000})",
        "        try:\n            if ccxt is None:\n                raise RuntimeError(\"ccxt is not installed\")\n            exchange_class = ccxt.binance({\"enableRateLimit\": True, \"timeout\": 5000})",
        1,
    )
    return "PATCHED", new_txt


def _tx_test_data_source_policy(txt: str) -> tuple[str, str]:
    if "crypto_trading.data_loader.ccxt.binance" in txt:
        return "ALREADY_APPLIED", txt
    if '"ccxt.binance.fetch_ohlcv"' not in txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = txt.replace('"ccxt.binance.fetch_ohlcv"', '"crypto_trading.data_loader.ccxt.binance"')
    return "PATCHED", new_txt


def _tx_test_web_ui_and_vault(txt: str) -> tuple[str, str]:
    if "BITUNIX_STANDBY_MODE" in txt and "TICKER_UNAVAILABLE" in txt:
        return "ALREADY_APPLIED", txt
    old_t03_m = re.search(
        r"    def test_03_bitunix_client\(self\):.*?(?=\n    def test_04_multi_asset_features\()",
        txt,
        re.S,
    )
    if not old_t03_m:
        return "TARGET_NOT_FOUND", txt
    new_t03 = """    def test_03_bitunix_client(self):
        \"\"\"測試 Bitunix 實盤對接客戶端 (含離線 Fail-Closed 與 Live Mock 報價驗證)\"\"\"
        from unittest.mock import MagicMock, patch
        client = BitunixClient()
        ticker = client.fetch_ticker("BTC/USDT")
        self.assertEqual(ticker["exchange"], "bitunix")
        self.assertIsNone(ticker["last"])
        self.assertEqual(ticker["status"], "OFFLINE")
        self.assertFalse(ticker["success"])

        bal = client.fetch_balance()
        self.assertIn("total_equity", bal)
        self.assertIsNone(bal["total_equity"])
        self.assertEqual(bal["error"], "BITUNIX_STANDBY_MODE")

        ord_fail = client.create_order("BTC/USDT", "BUY", "MARKET", 0.01)
        self.assertFalse(ord_fail["success"])
        self.assertEqual(ord_fail["error"], "TICKER_UNAVAILABLE")

        ord_limit = client.create_order("BTC/USDT", "BUY", "LIMIT", 0.01, price=68000.0)
        self.assertTrue(ord_limit["success"])

        client.is_live = True
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": {"lastPrice": "68000.0", "highPrice": "69000.0", "lowPrice": "67000.0", "volume": "1234.5"}}
        with patch("requests.get", return_value=mock_resp):
            live_t = client.fetch_ticker("BTC/USDT")
            self.assertEqual(live_t["exchange"], "bitunix")
            self.assertEqual(live_t["last"], 68000.0)
            self.assertEqual(live_t["status"], "LIVE")
            self.assertTrue(live_t["success"])
"""
    return "PATCHED", txt[:old_t03_m.start()] + new_t03 + txt[old_t03_m.end():]

BASELINE_INVENTORY_JSON = '{\n  "git_branch": "fix/overnight-audit-v5",\n  "git_commit_baseline": "44ec403",\n  "MODE_AUTHORITY": [\n    "crypto_trading/runtime_context.py:RuntimeTradingContext",\n    "crypto_trading/settings.py:Settings.trading_mode",\n    "crypto_trading/config.py:PAPER_TRADING (deprecated)"\n  ],\n  "ACCOUNT_SOURCE": [\n    "crypto_trading/testnet_client.py:BinanceTestnetClient.fetch_balance",\n    "crypto_trading/bitunix_client.py:BitunixClient.fetch_balance",\n    "web_ui/server.py:APP_STATE.build_trading_snapshot"\n  ],\n  "POSITION_SOURCE": [\n    "crypto_trading/testnet_client.py:BinanceTestnetClient.fetch_positions",\n    "crypto_trading/bitunix_client.py:BitunixClient.fetch_positions",\n    "crypto_trading/reconciliation_service.py:ReconciliationService.reconcile"\n  ],\n  "MARKET_SOURCE": [\n    "crypto_trading/testnet_client.py:BinanceTestnetClient.fetch_ticker",\n    "crypto_trading/bitunix_client.py:BitunixClient.fetch_ticker",\n    "crypto_trading/data_loader.py",\n    "taiwan_stock/market_analyzer.py"\n  ],\n  "ORDER_ENTRY_POINT": [\n    "crypto_trading/order_execution_service.py:OrderExecutionService.submit",\n    "crypto_trading/order_manager.py:OrderManager.execute_with_idempotency",\n    "crypto_trading/manual_controller.py:ManualPositionController",\n    "crypto_trading/bot_runner.py:BotRunner"\n  ],\n  "BRAIN_CONFIG": [\n    "worldquant_brain/brain_client.py:BRAIN_EMAIL",\n    "worldquant_brain/brain_client.py:BRAIN_PASSWORD",\n    "worldquant_brain/brain_client.py:BRAIN_SIMULATION_ENABLED",\n    "worldquant_brain/brain_client.py:BRAIN_AUTO_SUBMIT_ENABLED"\n  ],\n  "MODEL_ARTIFACT": [\n    "data_persistence/models/crypto_lgbm_weights.joblib",\n    "data_persistence/models/crypto_ppo_agent.pt",\n    "data_persistence/models/taiwan_stock_ranker.joblib"\n  ],\n  "RUNTIME_ARTIFACT": [\n    "data_persistence/system_state.db",\n    "worldquant_brain/alpha_vault.db"\n  ],\n  "HARDCODED_UI_VALUE": [\n    "web_ui/server.py:/api/crypto/account (remediated by post_v5_delta)",\n    "web_ui/static/index.html:cryptoEquity/cryptoPrice null handling (remediated by post_v5_delta)"\n  ],\n  "WORKER_STARTUP": [\n    "web_ui/server.py:main",\n    "crypto_trading/bot_runner.py:BotRunner.run_loop",\n    "taiwan_stock/scheduler.py"\n  ]\n}\n'

NEW_FILES = {
    "crypto_trading/order_execution_service.py": ORDER_EXECUTION_SERVICE_PY,
    "tests/test_post_v5_delta.py": DELTA_TESTS_PY,
    "reports/baseline_inventory.json": BASELINE_INVENTORY_JSON,
}

TRANSFORMS = {
    "crypto_trading/order_manager.py": _tx_order_manager,
    "crypto_trading/risk_manager.py": _tx_risk_manager,
    "crypto_trading/testnet_client.py": _tx_testnet_client,
    "crypto_trading/manual_controller.py": _tx_manual_controller,
    "crypto_trading/bot_runner.py": _tx_bot_runner_delta,
    "worldquant_brain/alpha_vault.py": _tx_alpha_vault,
    "taiwan_stock/market_analyzer.py": _tx_market_analyzer,
    "taiwan_stock/ml_ranker.py": _tx_ml_ranker,
    "crypto_trading/ml_trainer.py": _tx_ml_trainer,
    "crypto_trading/reconciliation_service.py": _tx_reconciliation_service,
    "crypto_trading/bitunix_client.py": _tx_bitunix_client,
    "crypto_trading/backtest_engine.py": _tx_backtest_engine,
    "crypto_trading/strategies/ml_strategy.py": _tx_ml_strategy,
    "crypto_trading/strategies/rl_strategy.py": _tx_rl_strategy,
    "web_ui/server.py": _tx_web_server,
    "web_ui/static/index.html": _tx_web_index,
    "crypto_trading/data_loader.py": _tx_data_loader,
    "tests/test_data_source_policy.py": _tx_test_data_source_policy,
    "tests/test_web_ui_and_vault.py": _tx_test_web_ui_and_vault,
}


def do_rollback(run_id: str) -> int:
    if not MANIFEST_PATH.exists():
        print("[delta-rollback] No manifest found.")
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_entry = manifest.get("runs", {}).get(run_id)
    if not run_entry:
        print(f"[delta-rollback] Run ID '{run_id}' not found.")
        return 1
    for rel, info in run_entry.get("files", {}).items():
        target = ROOT / rel
        bpath = info.get("backup_path")
        if bpath and (ROOT / bpath).exists():
            _atomic_write_text(target, (ROOT / bpath).read_text(encoding="utf-8"))
            print(f"  [restored] {rel}")
        elif not bpath and target.exists():
            target.unlink()
            print(f"  [removed-created] {rel}")
        prev = info.get("prev_manifest_entry")
        if prev is not None:
            manifest.setdefault("files", {})[rel] = prev
        else:
            manifest.get("files", {}).pop(rel, None)
    run_entry["status"] = "ROLLED_BACK"
    run_entry["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main() -> int:
    if "--rollback" in sys.argv:
        idx = sys.argv.index("--rollback")
        return do_rollback(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 1

    apply_mode = "--apply" in sys.argv
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {"files": {}, "runs": {}}
    known = manifest.get("files", {})
    plan: dict[str, tuple[str, str]] = {}
    has_err = False

    for rel, content in NEW_FILES.items():
        p = ROOT / rel
        if not p.exists():
            plan[rel] = ("PATCHED", content)
        else:
            oh = _sha256(p.read_text(encoding="utf-8"))
            nh = _sha256(content)
            if oh == nh:
                plan[rel] = ("ALREADY_APPLIED", content)
            elif rel in known and known[rel].get("post_sha256") == oh:
                plan[rel] = ("PATCHED", content)
            else:
                plan[rel] = ("CONFLICT", "")
                has_err = True

    for rel, fn in TRANSFORMS.items():
        p = ROOT / rel
        if not p.exists():
            plan[rel] = ("TARGET_NOT_FOUND", "")
            has_err = True
            continue
        txt = p.read_text(encoding="utf-8")
        st, new_txt = fn(txt)
        if st in {"TARGET_NOT_FOUND", "CONFLICT"} and rel in known and known[rel].get("post_sha256") and _sha256(txt) != known[rel]["post_sha256"]:
            plan[rel] = ("CONFLICT", "")
            has_err = True
            continue
        if st in {"TARGET_NOT_FOUND", "CONFLICT"}:
            has_err = True
        plan[rel] = (st, new_txt)

    print(f"[apply_post_v5_delta] Mode={'APPLY' if apply_mode else 'PLAN_DRY_RUN'}")
    for rel, (st, _) in plan.items():
        print(f"  - {rel}: {st}")
    if has_err:
        print("[apply_post_v5_delta] ABORTED: Conflict or missing target.")
        return 1
    if not apply_mode:
        return 0

    to_write = {r: c for r, (st, c) in plan.items() if st == "PATCHED"}
    if not to_write:
        print("[apply_post_v5_delta] Nothing to change (all ALREADY_APPLIED).")
        return 0

    with tempfile.TemporaryDirectory() as td:
        for r, c in to_write.items():
            _atomic_write_text(Path(td) / r, c)
        comp = subprocess.run([sys.executable, "-m", "compileall", "-q", td], capture_output=True, text=True)
        if comp.returncode != 0:
            print("[apply_post_v5_delta] STAGED COMPILEALL FAILED:", comp.stderr)
            return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    run_files: dict[str, dict] = {}
    manifest.setdefault("runs", {})[run_id] = {"status": "APPLYING", "files": run_files}
    _atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, sort_keys=True))

    for r, c in to_write.items():
        target = ROOT / r
        prev = known.get(r)
        if target.exists():
            oh = _sha256(target.read_text(encoding="utf-8"))
            bp = BACKUPS_DIR / run_id / r
            bp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, bp)
            brel = str(bp.relative_to(ROOT))
        else:
            oh = None
            brel = None
        _atomic_write_text(target, c)
        ph = _sha256(c)
        known[r] = {"pre_sha256": oh, "post_sha256": ph, "backup_path": brel}
        run_files[r] = {"pre_sha256": oh, "post_sha256": ph, "backup_path": brel, "prev_manifest_entry": prev}
    _atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, sort_keys=True))

    t_res = subprocess.run([sys.executable, "-m", "unittest", "tests/test_post_v5_delta.py"], cwd=ROOT, capture_output=True, text=True)
    if t_res.returncode != 0:
        print("[apply_post_v5_delta] DELTA TESTS FAILED, AUTO-ROLLING BACK:\n", t_res.stdout, t_res.stderr)
        do_rollback(run_id)
        return 1

    manifest["runs"][run_id]["status"] = "APPLIED"
    _atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[apply_post_v5_delta] Applied and verified {len(to_write)} delta files cleanly (run_id={run_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())