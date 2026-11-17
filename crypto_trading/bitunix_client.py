"""
Bitunix 實盤交易撮合對接客戶端 (Bitunix Live Trading REST API Client)
支援：
1. Bitunix V1/V2 規範之 HMAC-SHA256 安全簽名機制
2. 查詢帳戶合約與現貨資產餘額 (Account Balance & Margin)
3. 查詢目前持倉 (Positions) 與訂單狀態 (Orders)
4. 市價單與限價單撮合 (Market & Limit Orders)
5. 一鍵清倉與撤單
6. 與幣安模擬/測試網接口標準統一，支援「幣安跑模擬，最終實盤用 Bitunix」的無縫架構
"""

import hashlib
import hmac
import json
import os
import time
from typing import Any

from crypto_trading.settings import Settings, TradingMode, get_settings


class BitunixClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://api.bitunix.com",
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self.api_key = api_key if api_key is not None else self._settings.bitunix_api_key
        self.api_secret = api_secret if api_secret is not None else self._settings.bitunix_api_secret
        self.base_url = base_url.rstrip("/")
        # 由 Settings 注入判定 is_live，不得再用 bool(api_key) 判定
        self.is_live = self._settings.trading_mode == TradingMode.BITUNIX_LIVE

        # 實盤尚未填寫 Key 時的沙盒保險箱
        self._simulated_balance = {"USDT": 20000.0, "BTC": 0.0, "ETH": 0.0}
        self._simulated_positions = []
        self._simulated_orders = []

    def _generate_signature(self, timestamp: str, nonce: str, query_params: str, body: str = "") -> str:
        """
        Bitunix HMAC-SHA256 簽名算法:
        Signature = HMAC-SHA256(secret, timestamp + nonce + api_key + query_params + body)
        """
        raw_payload = f"{timestamp}{nonce}{self.api_key}{query_params}{body}"
        sig = hmac.new(self.api_secret.encode("utf-8"), raw_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    def _get_headers(self, query_params: str = "", body: str = "") -> dict[str, str]:
        """產生 Bitunix 要求的 HTTP Header"""
        timestamp = str(int(time.time() * 1000))
        nonce = hashlib.md5(f"{timestamp}_{os.urandom(8)}".encode()).hexdigest()[:16]
        signature = self._generate_signature(timestamp, nonce, query_params, body)

        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "timestamp": timestamp,
            "nonce": nonce,
            "signature": signature,
        }

    def fetch_ticker(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """Return a stable, fail-closed ticker schema.

        Public market data is queried only in BITUNIX_LIVE mode. In every
        non-live mode the method returns an explicit OFFLINE result instead of
        synthesizing a tradable price.
        """
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

    def fetch_balance(self) -> dict[str, Any]:
        """獲取 Bitunix 帳戶資產與可用保證金"""
        if not self.is_live:
            return {
                "mode": "BITUNIX_STANDBY_MODE (未配置實盤 API Key)",
                "exchange": "bitunix",
                "total_equity": None,
                "free_balance": None,
                "balances": {},
                "error": "BITUNIX_STANDBY_MODE",
                "notice": "請在系統環境變數或 config.py 中配置 BITUNIX_API_KEY 與 BITUNIX_API_SECRET 以啟動真金白銀實盤撮合。",
            }

        try:
            import requests

            url = f"{self.base_url}/api/v1/futures/account"
            headers = self._get_headers()
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                d = resp.json().get("data", {})
                equity = float(d.get("totalMargin", 0.0))
                free = float(d.get("availableMargin", 0.0))
                return {
                    "mode": "BITUNIX_LIVE_CONNECTED",
                    "exchange": "bitunix",
                    "total_equity": round(equity, 2),
                    "free_balance": round(free, 2),
                    "data": d,
                }
        except Exception as e:
            return {
                "mode": "BITUNIX_ERROR",
                "error": str(e),
                "total_equity": self._simulated_balance["USDT"],
                "free_balance": self._simulated_balance["USDT"],
            }

    def fetch_positions(self) -> list[dict[str, Any]]:
        """獲取 Bitunix 當前合約持倉"""
        if not self.is_live:
            return self._simulated_positions

        try:
            import requests

            url = f"{self.base_url}/api/v1/futures/position"
            headers = self._get_headers()
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                raw_positions = resp.json().get("data", [])
                results = []
                for p in raw_positions:
                    contracts = float(p.get("total", 0.0))
                    if contracts > 0:
                        results.append(
                            {
                                "symbol": p.get("symbol"),
                                "side": "LONG" if p.get("holdSide") == "1" else "SHORT",
                                "contracts": contracts,
                                "entry_price": float(p.get("openPrice", 0.0)),
                                "unrealized_pnl": float(p.get("unrealizedPnl", 0.0)),
                                "leverage": int(p.get("leverage", 1)),
                                "exchange": "bitunix",
                            }
                        )
                return results
        except Exception:
            pass
        return []

    def create_order(
        self,
        symbol: str = "BTC/USDT",
        side: str = "buy",
        order_type: str = "market",
        amount: float = 0.01,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """
        向 Bitunix 下達實時合約訂單
        支援市價 (MARKET) 與限價 (LIMIT)
        """
        side = side.upper()
        order_type = order_type.upper()
        clean_symbol = symbol.replace("/", "").upper()

        if not self.is_live:
            ticker = self.fetch_ticker(symbol)
            exec_price = price if (order_type == "LIMIT" and price) else ticker.get("last")
            if exec_price is None:
                return {
                    "success": False,
                    "mode": "BITUNIX_STANDBY_TICKER_UNAVAILABLE",
                    "error": "TICKER_UNAVAILABLE",
                    "message": "Bitunix 行情不可用，拒絕以假價格模擬成交",
                }
            order_info = {
                "id": f"bitunix_sim_{int(time.time() * 1000)}",
                "exchange": "bitunix",
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "price": exec_price,
                "amount": amount,
                "reduceOnly": reduce_only,
                "status": "closed" if order_type == "MARKET" else "open",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if side == "BUY":
                self._simulated_balance["USDT"] -= exec_price * amount
                self._simulated_positions.append(
                    {
                        "symbol": symbol,
                        "side": "LONG",
                        "contracts": amount,
                        "entry_price": exec_price,
                        "unrealized_pnl": 0.0,
                        "leverage": 1,
                        "exchange": "bitunix",
                    }
                )
            else:
                self._simulated_balance["USDT"] += exec_price * amount
                self._simulated_positions = [p for p in self._simulated_positions if p["symbol"] != symbol]

            return {
                "success": True,
                "mode": "BITUNIX_SIMULATED",
                "order": order_info,
                "message": f"[Bitunix 預備撮合] {order_type} {side} {amount} {symbol} @ {exec_price:.2f} (實盤備便待命)",
            }

        # 實盤發送真實 API 委託
        try:
            import requests

            url = f"{self.base_url}/api/v1/futures/order"
            payload = {
                "symbol": clean_symbol,
                "side": 1 if side == "BUY" else 2,
                "type": 1 if order_type == "LIMIT" else 2,
                "qty": str(amount),
                "reduceOnly": 1 if reduce_only else 0,
            }
            if order_type == "LIMIT" and price:
                payload["price"] = str(price)

            body_str = json.dumps(payload)
            headers = self._get_headers(body=body_str)

            resp = requests.post(url, headers=headers, data=body_str, timeout=5)
            res_data = resp.json()
            if resp.status_code == 200 and res_data.get("code") == 0:
                return {
                    "success": True,
                    "mode": "BITUNIX_LIVE_EXECUTED",
                    "order_id": res_data.get("data", {}).get("orderId"),
                    "raw": res_data,
                    "message": f"[Bitunix 實盤] 成功下單 {order_type} {side} {amount} {clean_symbol}",
                }
            else:
                return {
                    "success": False,
                    "mode": "BITUNIX_LIVE_REJECTED",
                    "code": res_data.get("code"),
                    "message": f"Bitunix 拒絕委託: {res_data.get('msg')}",
                }
        except Exception as e:
            return {
                "success": False,
                "mode": "BITUNIX_LIVE_ERROR",
                "error": str(e),
                "message": f"Bitunix 下單請求異常: {e!s}",
            }

    def close_all_positions(self) -> dict[str, Any]:
        """Bitunix 一鍵平倉所有持倉 (平倉委託必須附帶 reduceOnly=True 防反向開倉)"""
        count = 0
        if not self.is_live:
            count = len(self._simulated_positions)
            self._simulated_positions.clear()
        else:
            try:
                positions = self.fetch_positions()
                for p in positions:
                    side = "sell" if p["side"] == "LONG" else "buy"
                    res = self.create_order(
                        p["symbol"],
                        side=side,
                        order_type="market",
                        amount=p["contracts"],
                        reduce_only=True,
                    )
                    if res.get("success"):
                        count += 1
            except Exception:
                pass

        return {
            "success": True,
            "exchange": "bitunix",
            "closed_count": count,
            "message": f"[Bitunix 實盤] 已結清所有部位 ({count} 筆持倉已處理)",
        }


if __name__ == "__main__":
    client = BitunixClient()
    print("=== Bitunix 實盤客戶端模組檢測 ===")
    print("Ticker:", client.fetch_ticker("BTC/USDT"))
    print("Balance:", client.fetch_balance())
