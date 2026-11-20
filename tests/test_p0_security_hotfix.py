"""Comprehensive unit tests for P0 Security & Bugfix Hotfix v1."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crypto_trading.bitunix_client import BitunixClient
from crypto_trading.order_execution_service import OrderExecutionService, OrderIntent
from crypto_trading.order_manager import OrderManager
from crypto_trading.risk_manager import DecisionAction, RiskManager
from crypto_trading.settings import Settings, TradingMode, reset_settings
from taiwan_stock.universe_registry import _safe_float
from worldquant_brain.safe_expr_eval import safe_eval_expr


class TestP0SecurityHotfix(unittest.TestCase):
    def test_safe_eval_expr_blocks_rce_and_allows_valid_math(self) -> None:
        env = {"x": 10.0, "y": 2.0, "add": lambda a, b: a + b}
        self.assertEqual(safe_eval_expr("add(x, -y) * 2", env), 16.0)
        for payload in (
            "__import__('os').system('echo pwn')",
            "add.__globals__['os']",
            "().__class__.__mro__[1].__subclasses__()",
            "(lambda z: z)(1)",
            "[a for a in (1, 2)]",
        ):
            with self.assertRaises(ValueError):
                safe_eval_expr(payload, env)

    def test_no_raw_eval_remains_in_strategy_or_screener(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for rel in (
            "crypto_trading/strategies/brain_alpha_strategy.py",
            "worldquant_brain/local_cs_screener.py",
        ):
            src = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("= eval(", src)
            self.assertIn("safe_eval_expr", src)

    def test_settings_has_live_trading_unlocked_and_risk_manager_no_attribute_error(self) -> None:
        paper_settings = Settings(env={"TRADING_MODE": "LOCAL_PAPER"})
        self.assertFalse(paper_settings.live_trading_unlocked)
        reset_settings(paper_settings)
        try:
            rm = RiskManager(initial_capital=1000.0)
            dec = rm.evaluate_order(
                symbol="BTC/USDT",
                side="BUY",
                order_type="MARKET",
                amount=0.01,
                price=50000.0,
                context={"trading_mode": TradingMode.BITUNIX_LIVE, "data_source": "LIVE"},
            )
            self.assertFalse(dec.allowed)
            self.assertTrue(any("BITUNIX_LIVE is locked" in r for r in dec.reasons))
        finally:
            reset_settings(None)

    def test_order_execution_service_passes_reduce_only_and_client_order_id(self) -> None:
        captured = {}

        class SpyClient:
            def fetch_positions(self):
                return [{"symbol": "BTC/USDT", "side": "LONG", "contracts": 0.01}]

            def create_order(self, symbol, side, order_type, amount, price=None, reduce_only=False, client_order_id=None):
                captured.update(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    reduce_only=reduce_only,
                    client_order_id=client_order_id,
                )
                return {"success": True, "exchange_order_id": "ex_123", "order": {"status": "closed"}}

        with tempfile.TemporaryDirectory() as tmp:
            om = OrderManager(db_path=str(Path(tmp) / "om.db"))
            rm = RiskManager(initial_capital=10000.0)
            svc = OrderExecutionService(risk_manager=rm, order_manager=om)
            res = svc.emergency_flatten(SpyClient())
            self.assertEqual(res["status"], "COMPLETE")
            self.assertTrue(captured["reduce_only"])
            self.assertTrue(bool(captured["client_order_id"]))

    def test_bitunix_live_balance_and_close_fail_closed(self) -> None:
        client = BitunixClient()
        client.is_live = True
        with patch("requests.get", side_effect=RuntimeError("network down")):
            bal = client.fetch_balance()
            self.assertEqual(bal["mode"], "BITUNIX_ERROR")
            self.assertIsNone(bal["total_equity"])
            self.assertIsNone(bal["free_balance"])

        with patch.object(client, "fetch_positions", side_effect=RuntimeError("api down")):
            res = client.close_all_positions()
            self.assertFalse(res["success"])

    def test_taiwan_safe_float_parses_commas_and_rejects_nan(self) -> None:
        self.assertEqual(_safe_float("1,050.00"), 1050.0)
        self.assertEqual(_safe_float("+12,345.5"), 12345.5)
        self.assertIsNone(_safe_float("--"))
        self.assertIsNone(_safe_float(float("nan")))
        self.assertIsNone(_safe_float(float("inf")))

    def test_taiwan_report_generator_no_html_shadowing(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "taiwan_stock/report_generator.py").read_text(encoding="utf-8")
        self.assertIn('html_out = f"""', src)
        self.assertIn("html.escape(", src)

    def test_cloud_configs_disable_auto_submit_and_fix_static_patch_url(self) -> None:
        root = Path(__file__).resolve().parents[1]
        render_yaml = (root / "render.yaml").read_text(encoding="utf-8")
        self.assertIn('BRAIN_AUTO_SUBMIT_ENABLED\n        value: "false"', render_yaml)
        self.assertIn('TRADING_MODE\n        value: "LOCAL_PAPER"', render_yaml)

        workflow = (root / ".github/workflows/brain_miner.yml").read_text(encoding="utf-8")
        self.assertIn('BRAIN_AUTO_SUBMIT_ENABLED: "false"', workflow)

        index_html = (root / "web_ui/static/index.html").read_text(encoding="utf-8")
        self.assertIn('src="/static_snapshot_patch.js"', index_html)
        self.assertNotIn('src="/static/static_snapshot_patch.js"', index_html)

    def test_brain_job_queue_stores_metrics_before_cache_and_checks_429_in_worker(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "worldquant_brain/brain_job_queue.py").read_text(encoding="utf-8")
        idx_metrics = src.index('job.metrics = {\n                                "sharpe": sharpe,')
        idx_cache = src.index("self._store_simulation_cache(job)", idx_metrics - 200)
        self.assertLess(idx_metrics, idx_cache)
        self.assertIn("wait_until = self._rate_limit_until", src)

    def test_web_ui_server_wires_auth_and_removes_wildcard_cors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = (root / "web_ui/server.py").read_text(encoding="utf-8")
        self.assertNotIn('"Access-Control-Allow-Origin", "*"', src)
        self.assertIn("def _check_write_auth(", src)
        self.assertIn("from web_ui.auth import Role, auth_manager", src)


if __name__ == "__main__":
    unittest.main()
