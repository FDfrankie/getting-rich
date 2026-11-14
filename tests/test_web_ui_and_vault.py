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

    def test_02_binance_testnet_client(self):
        """測試幣安測試網 Client 行情與 Mock 掛單"""
        client = BinanceTestnetClient()
        ticker = client.fetch_ticker("BTC/USDT")
        self.assertIn("last", ticker)
        self.assertGreater(ticker["last"], 0)

        # 模擬掛單
        order_res = client.create_order("BTC/USDT", "buy", "market", 0.01)
        self.assertTrue(order_res["success"])

        # 持倉
        pos = client.fetch_positions()
        self.assertIsInstance(pos, list)

    def test_03_web_server_endpoints(self):
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

        # 4. Alpha Vault 列表
        with urlopen(f"{base_url}/api/brain/alphas") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertGreaterEqual(len(data["alphas"]), 1)

    def test_04_crypto_order_via_api(self):
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
