import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.regime_filter import HierarchicalRegimeFilter
from crypto_trading.risk_manager import RiskManager
from crypto_trading.testnet_client import BinanceTestnetClient
from data_persistence.gdrive_sync import GDrivePersistenceSync


def create_sample_bars(n: int = 100) -> pd.DataFrame:
    np.random.seed(123)
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    p = 65000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, size=n)))
    return pd.DataFrame(
        {
            "Open": p * 0.999,
            "High": p * 1.002,
            "Low": p * 0.998,
            "Close": p,
            "Volume": np.random.uniform(50, 200, size=n),
        },
        index=dates,
    )


class TestArchitecturalUpgrades(unittest.TestCase):
    def setUp(self):
        self.temp_db = "data_persistence/test_temp_state.db"
        self.sync = GDrivePersistenceSync(db_path=self.temp_db)

    def tearDown(self):
        if os.path.exists(self.temp_db):
            try:
                os.remove(self.temp_db)
            except Exception:
                pass

    def test_gdrive_sync_and_post_mortem_logging(self):
        """驗證 Google Drive / 本地持久化資料庫結構與 Post-Mortem 檢討寫入/查詢"""
        self.sync.save_state("active_mode", "BINANCE_TESTNET")
        self.assertEqual(self.sync.get_state("active_mode"), "BINANCE_TESTNET")

        self.sync.record_post_mortem(
            symbol="BTC/USDC",
            entry_price=68000.0,
            exit_price=66500.0,
            loss_amount=-150.0,
            loss_pct=-2.2,
            cause_analysis="觸發 ATR 移動停損",
        )
        pms = self.sync.get_recent_post_mortems(limit=5)
        self.assertGreaterEqual(len(pms), 1)
        self.assertEqual(pms[0]["symbol"], "BTC/USDC")
        self.assertIn("ATR", pms[0]["cause_analysis"])

        # 測試快照匯出
        snap_path = self.sync.export_snapshot_json()
        self.assertTrue(os.path.exists(snap_path))

    def test_usdc_liquidity_guard(self):
        """驗證 USDC 流動性守護：價差過大時拒絕市價單"""
        client = BinanceTestnetClient(settlement="USDC")
        # 測試常態檢查
        guard = client.check_liquidity_guard("BTC/USDT", max_spread_pct=0.15)
        self.assertIn("passed", guard)
        self.assertIn("spread_pct", guard)

        # 測試嚴苛價差 (0.000001%) 攔截
        tight_guard = client.check_liquidity_guard("BTC/USDT", max_spread_pct=0.000001)
        self.assertFalse(tight_guard["passed"])

    def test_exchange_reconciliation(self):
        """驗證交易所狀態機與部位同調 (Idempotency & Reconciliation)"""
        client = BinanceTestnetClient(settlement="USDC")
        # 模擬本地部位 0.05，若遠端為 0.05 則一致
        client._mock_positions = [{"symbol": "BTC/USDC:USDC", "side": "LONG", "contracts": 0.05}]
        res_sync = client.reconcile_with_exchange(local_position=0.05, symbol="BTC/USDC")
        self.assertTrue(res_sync["in_sync"])

        # 模擬本地部位 0.0 (漏記)，偵測到偏離並同調校正
        res_mismatch = client.reconcile_with_exchange(local_position=0.0, symbol="BTC/USDC")
        self.assertFalse(res_mismatch["in_sync"])
        self.assertEqual(res_mismatch["remote_position"], 0.05)

    def test_hierarchical_regime_filter(self):
        """驗證高階時框 (1h/1d) 市場狀態檢驗與微觀交易信號過濾"""
        rf = HierarchicalRegimeFilter()
        df = create_sample_bars(80)
        regime = rf.detect_regime(df)
        self.assertIn("regime", regime)
        self.assertIn("sizing_multiplier", regime)

        # 模擬高階空頭狀態下，微觀多單信號 (raw_signal=1) 被過濾為 0
        regime_bear = {
            "allow_long": False,
            "allow_short": True,
            "sizing_multiplier": 0.8,
        }
        filtered_sig = rf.filter_micro_signal(raw_signal=1, regime_info=regime_bear)
        self.assertEqual(filtered_sig, 0)

        # 模擬熔斷狀態 (sizing_multiplier=0)，所有信號均置為 0
        regime_shock = {
            "allow_long": False,
            "allow_short": False,
            "sizing_multiplier": 0.0,
        }
        self.assertEqual(rf.filter_micro_signal(1, regime_shock), 0)
        self.assertEqual(rf.filter_micro_signal(-1, regime_shock), 0)

    def test_circuit_breaker(self):
        """驗證 RiskManager 最大回撤停機熔斷保護"""
        rm = RiskManager(initial_capital=10000.0)
        # 資本回撤至 9100 (-9%)，超過 8% 熔斷門檻
        is_safe = rm.update_capital(9100.0)
        self.assertFalse(is_safe)
        self.assertTrue(rm.is_halted)
        pos = rm.calculate_position_size(65000.0, 1000.0)
        self.assertEqual(pos["size"], 0.0)


if __name__ == "__main__":
    unittest.main()
