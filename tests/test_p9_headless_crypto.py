"""Unit tests for P9 Headless Crypto Bot Bundle."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.headless_crypto_bot import run_headless_crypto


class TestP9HeadlessCrypto(unittest.TestCase):
    def test_headless_crypto_writes_atomic_pnl_snapshot_without_ui(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.status_summary.side_effect = [
            [
                {"symbol": "BTC/USDT", "balance_usdt": 10000.0, "position": 0.0, "entry_price": 0.0, "position_side": "FLAT"},
                {"symbol": "ETH/USDT", "balance_usdt": 10000.0, "position": 0.0, "entry_price": 0.0, "position_side": "FLAT"},
            ],
            [
                {"symbol": "BTC/USDT", "balance_usdt": 10250.0, "position": 0.0, "entry_price": 0.0, "position_side": "FLAT"},
                {"symbol": "ETH/USDT", "balance_usdt": 9950.0, "position": 0.0, "entry_price": 0.0, "position_side": "FLAT"},
            ],
        ]
        with tempfile.TemporaryDirectory() as td:
            status_file = Path(td) / "crypto_pnl_status.json"
            snap = run_headless_crypto(
                symbols=["BTC/USDT", "ETH/USDT"],
                interval_sec=0.01,
                iterations=1,
                status_path=status_file,
                manager=mock_mgr,
            )
            self.assertTrue(status_file.exists())
            saved = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["initial_equity_usdt"], 20000.0)
            self.assertEqual(saved["total_equity_usdt"], 20200.0)
            self.assertEqual(saved["pnl_usdt"], 200.0)
            self.assertAlmostEqual(saved["pnl_pct"], 1.0, places=4)
            self.assertEqual(len(saved["symbols"]), 2)
            self.assertEqual(snap["pnl_usdt"], 200.0)
            mock_mgr.step_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
