import unittest
import json
import os
import sys
import threading
import time
from urllib.request import urlopen, Request

# 確保路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui.server import run_server, APP_STATE
from worldquant_brain.alpha_vault import AlphaVault
from crypto_trading.testnet_client import BinanceTestnetClient
from crypto_trading.bitunix_client import BitunixClient
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.feature_engineer import CryptoFeatureEngineer

class TestWebUIAndServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 啟動測試用後端 Server 於 8999 端口
        cls.test_port = 8999
        cls.server_thread = threading.Thread(
            target=run_server,
            kwargs={"port": cls.test_port, "host": "127.0.0.1"},
            daemon=True
        )
        cls.server_thread.start()
        time.sleep(1.5)  # 等候服務啟動

    def test_01_alpha_vault_qualification(self):
        """測試 Alpha Vault 門檻驗證與存取功能"""
        vault = AlphaVault(db_path=":memory:")
        # 測試不符合門檻
        res_fail = vault.save_alpha(
            name="Low_Sharpe_Alpha",
            expression="close / open",
            sharpe=0.8,
            fitness=0.7,
            turnover=15.0
        )
        self.assertFalse(res_fail["success"])

        # 測試符合門檻
        res_ok = vault.save_alpha(
            name="Super_Alpha",
            expression="rank(ts_argmax(high, 20))",
            sharpe=1.65,
            fitness=1.35,
            turnover=22.0
        )
        self.assertTrue(res_ok["success"])
        alphas = vault.list_alphas()
        self.assertEqual(len(alphas), 1)
        self.assertEqual(alphas[0]["name"], "Super_Alpha")

        # 測試分潤公式
        payout = vault.calculate_payout_projection(50000.0, 1.85, 32.0, is_consultant=True)
        self.assertEqual(payout["final_estimated_payout_usd"], 4800.0)

    def test_02_binance_testnet_client(self):
        """測試幣安測試網 Client 行情與 USDC 0費率掛單"""
        client = BinanceTestnetClient()
        ticker = client.fetch_ticker("BTC/USDT")
        self.assertIn("last", ticker)
        self.assertGreater(ticker["last"], 0)
        self.assertTrue(ticker.get("zero_fee_usdc"))

        # 下單
        order_res = client.create_order("BTC/USDT", "buy", "market", 0.01)
        self.assertTrue(order_res["success"])

        # 持倉
        pos = client.fetch_positions()
        self.assertIsInstance(pos, list)

    def test_03_bitunix_client(self):
        """測試 Bitunix 實盤對接客戶端"""
        client = BitunixClient()
        ticker = client.fetch_ticker("BTC/USDT")
        self.assertEqual(ticker["exchange"], "bitunix")
        self.assertGreater(ticker["last"], 0)

        bal = client.fetch_balance()
        self.assertIn("total_equity", bal)

        ord_res = client.create_order("BTC/USDT", "BUY", "MARKET", 0.01)
        self.assertTrue(ord_res["success"])

    def test_04_multi_asset_features(self):
        """測試多週期多幣種資料加載與特徵工程"""
        loader = CryptoDataLoader()
        df_1m = loader.fetch_ohlcv(limit=30, symbol="BTC/USDT", timeframe="1m")
        df_1h = loader.fetch_ohlcv(limit=30, symbol="ETH/USDT", timeframe="1h")
        self.assertEqual(len(df_1m), 30)
        self.assertEqual(len(df_1h), 30)

        fe = CryptoFeatureEngineer()
        btc_df = loader.fetch_ohlcv(limit=50, symbol="BTC/USDT")
        eth_df = loader.fetch_ohlcv(limit=50, symbol="ETH/USDT")
        sol_df = loader.fetch_ohlcv(limit=50, symbol="SOL/USDT")
        cross = fe.compute_multi_asset_cross_features({
            "BTC": btc_df, "ETH": eth_df, "SOL": sol_df
        })
        self.assertIn("ETH_corr_BTC_24", cross.columns)
        self.assertIn("SOL_to_ETH_ratio", cross.columns)

    def test_05_web_server_endpoints(self):
        """測試 Web Server 核心 API 路由響應"""
        base_url = f"http://127.0.0.1:{self.test_port}"

        # 1. 首頁 index.html
        with urlopen(f"{base_url}/") as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("Antigravity Quant", html)

        # 2. 系統狀態
        with urlopen(f"{base_url}/api/status") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "online")

        # 3. 加密帳戶與行情
        with urlopen(f"{base_url}/api/crypto/account") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("total_equity", data)

        # 4. Alpha Vault 列表與變現教學
        with urlopen(f"{base_url}/api/brain/alphas") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("roadmap", data)
            self.assertIn("payout_sample", data)

        # 5. 台股晨報 HTML 路由
        with urlopen(f"{base_url}/api/taiwan/report") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("html_content", data)

        # 6. TradingView Lightweight Charts K 線即時行情 API (支援加密與台股)
        with urlopen(f"{base_url}/api/crypto/klines?symbol=BTC/USDT&interval=1m") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("candles", data)
            self.assertIn("overlay", data)
            self.assertIn("markers", data)
            self.assertGreater(len(data["candles"]), 0)

        with urlopen(f"{base_url}/api/crypto/klines?symbol=2330.TW&interval=1d") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("candles", data)

    def test_06_crypto_order_via_api(self):
        """測試透過 API 提交下單與平倉指令"""
        base_url = f"http://127.0.0.1:{self.test_port}"

        # 下單
        req = Request(
            f"{base_url}/api/crypto/order",
            data=json.dumps({"side": "BUY", "amount": 0.01, "symbol": "BTC/USDT"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])

        # 一鍵平倉
        req_close = Request(
            f"{base_url}/api/crypto/close_all",
            data=b"{}",
            headers={"Content-Type": "application/json"}
        )
        with urlopen(req_close) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])

if __name__ == "__main__":
    unittest.main()
