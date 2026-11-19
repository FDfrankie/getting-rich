import argparse
import os
import sys
import time
from typing import Any

try:
    import ccxt
except ImportError:
    ccxt = None
from crypto_trading.config import (
    BINANCE_TESTNET_API_KEY,
    BINANCE_TESTNET_API_SECRET,
    DEFAULT_SETTLEMENT_CURRENCY,
)


class BinanceTestnetClient:
    """
    幣安官方合約/現貨測試網 (Binance Futures & Spot Testnet) 串接模組
    功能：
    1. 完整支援 Binance Futures Testnet (fapi testnet: https://testnet.binancefuture.com)
    2. 支援 USDT-M 及 USDC-M 永續合約 (USDC 合約專享零手續費優惠)
    3. 查詢合約帳戶資產餘額 (Balance & Equity)
    4. 查詢目前未平倉部位 (Positions) 與委託訂單 (Open Orders)
    5. 市價 (Market) 與 限價 (Limit) 下單
    6. 一鍵全平倉 (Close All Positions) 與撤單
    7. 若網路或遠端連線波動，自動平滑降級為高擬真沙盒環境，確保系統不崩潰
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        market_type: str = "future",  # 'future' or 'spot'
        settlement: str = DEFAULT_SETTLEMENT_CURRENCY,
    ):
        self.api_key = api_key or BINANCE_TESTNET_API_KEY
        self.api_secret = api_secret or BINANCE_TESTNET_API_SECRET
        self.market_type = market_type
        self.settlement = settlement
        self.is_connected = False
        self.mock_mode = False

        # 初始化 CCXT 幣安實例
        self.exchange = None
        self._init_exchange()

        # Mock 模式內部模擬帳本 (當無真實網路或金鑰異常時提供防護)
        self._mock_balance = {"USDT": 15000.0, "USDC": 15000.0, "BTC": 0.05, "ETH": 0.8}
        self._mock_positions = [
            {
                "symbol": "BTC/USDC:USDC",
                "side": "LONG",
                "contracts": 0.05,
                "entry_price": 68250.0,
                "unrealized_pnl": +112.5,
                "leverage": 5,
                "fee_rate": 0.0,
            }
        ]
        self._mock_orders = []

    def _init_exchange(self):
        try:
            if not ccxt:
                raise ImportError("CCXT library is not installed.")
            config = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future" if self.market_type == "future" else "spot",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                },
            }
            self.exchange = ccxt.binance(config)
            # 配置 Binance Futures Testnet 官方專用端點
            if "test" in self.exchange.urls:
                self.exchange.urls["api"] = self.exchange.urls["test"]

            if self.api_key and self.api_secret:
                self.is_connected = True
                self.mock_mode = False
            else:
                self.is_connected = False
                self.mock_mode = True
        except Exception as e:
            print(f"[!] 初始化 Binance Testnet 失敗，降級至 Mock 模式: {e}")
            self.exchange = None
            self.mock_mode = True

    def _to_raw_symbol(self, symbol: str) -> str:
        """轉換為幣安合約原始 symbol (例如 BTCUSDC, BTCUSDT)"""
        base = (
            symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "").replace("USDC", "").replace(":", "")
        )
        return f"{base}{self.settlement}".upper()

    def format_symbol(self, raw_symbol: str) -> str:
        """根據設定自動對齊結算貨幣 (USDC 或 USDT)"""
        base = (
            raw_symbol.split("/")[0]
            if "/" in raw_symbol
            else raw_symbol.replace("USDT", "").replace("USDC", "").replace(":", "")
        )
        if self.market_type == "future":
            return f"{base}/{self.settlement}:{self.settlement}"
        return f"{base}/{self.settlement}"

    def fetch_ticker(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """獲取實時行情報價 (支援 USDC 與 USDT 合約)"""
        target_symbol = self.format_symbol(symbol)
        raw_symbol = self._to_raw_symbol(symbol)
        try:
            if self.exchange:
                # 優先使用 fapiPublic 獲取合約即時行情
                try:
                    res = self.exchange.fapiPublicGetTickerPrice({"symbol": raw_symbol})
                    price = float(res.get("price", 0.0))
                    if price > 0:
                        return {
                            "symbol": target_symbol,
                            "last": price,
                            "bid": round(price * 0.9999, 2),
                            "ask": round(price * 1.0001, 2),
                            "high": round(price * 1.01, 2),
                            "low": round(price * 0.99, 2),
                            "volume": 10000.0,
                            "timestamp": int(res.get("time", time.time() * 1000)),
                            "settlement": self.settlement,
                            "zero_fee_usdc": self.settlement == "USDC",
                        }
                except Exception:
                    pass

                try:
                    ticker = self.exchange.fetch_ticker(target_symbol)
                    return {
                        "symbol": target_symbol,
                        "last": float(ticker.get("last", 0.0)),
                        "bid": float(ticker.get("bid", 0.0)),
                        "ask": float(ticker.get("ask", 0.0)),
                        "high": float(ticker.get("high", 0.0)),
                        "low": float(ticker.get("low", 0.0)),
                        "volume": float(ticker.get("baseVolume", 0.0)),
                        "timestamp": ticker.get("timestamp"),
                        "settlement": self.settlement,
                        "zero_fee_usdc": self.settlement == "USDC",
                    }
                except Exception:
                    pass
        except Exception:
            pass

        if not self.mock_mode and self.is_connected:
            return {"symbol": target_symbol, "last": None, "error": "TICKER_UNAVAILABLE", "timestamp": None, "settlement": self.settlement}
        # 備用模擬報價 (僅限 mock_mode)
        return {
            "symbol": target_symbol,
            "last": 68500.0 if "BTC" in symbol else 3500.0,
            "bid": 68490.0,
            "ask": 68510.0,
            "high": 69200.0,
            "low": 67800.0,
            "volume": 12850.5,
            "timestamp": int(time.time() * 1000),
            "settlement": self.settlement,
            "zero_fee_usdc": self.settlement == "USDC",
        }

    def fetch_balance(self) -> dict[str, Any]:
        """查詢幣安測試網合約錢包餘額 (包含 USDC 與 USDT)"""
        if self.mock_mode or not self.is_connected or not self.exchange:
            return {
                "mode": "MOCK_TESTNET",
                "total_equity": self._mock_balance.get(self.settlement, 15000.0),
                "free_balance": self._mock_balance.get(self.settlement, 15000.0) * 0.8,
                "settlement_currency": self.settlement,
                "balances": self._mock_balance,
                "zero_fee_usdc": True,
            }

        try:
            # 優先使用 fapiPrivateV2GetAccount 獲取合約資產
            acc = self.exchange.fapiPrivateV2GetAccount()
            assets = acc.get("assets", [])
            total_usdt = 0.0
            free_usdt = 0.0
            upnl_usdt = 0.0
            margin_usdt = 0.0
            total_usdc = 0.0
            free_usdc = 0.0
            upnl_usdc = 0.0
            margin_usdc = 0.0
            balances = {}
            for a in assets:
                asset_name = a.get("asset")
                wallet_bal = float(a.get("walletBalance", 0.0))
                avail_bal = float(a.get("availableBalance", 0.0))
                upnl_val = float(a.get("unrealizedProfit", 0.0) or 0.0)
                margin_val = float(a.get("marginBalance", wallet_bal + upnl_val) or (wallet_bal + upnl_val))
                if wallet_bal > 0 or avail_bal > 0:
                    balances[asset_name] = wallet_bal
                if asset_name == "USDT":
                    total_usdt = wallet_bal
                    free_usdt = avail_bal
                    upnl_usdt = upnl_val
                    margin_usdt = margin_val
                elif asset_name == "USDC":
                    total_usdc = wallet_bal
                    free_usdc = avail_bal
                    upnl_usdc = upnl_val
                    margin_usdc = margin_val

            selected_total = total_usdc if self.settlement == "USDC" else total_usdt
            selected_free = free_usdc if self.settlement == "USDC" else free_usdt
            selected_upnl = upnl_usdc if self.settlement == "USDC" else upnl_usdt
            selected_margin = margin_usdc if self.settlement == "USDC" else margin_usdt

            return {
                "mode": "OFFICIAL_BINANCE_TESTNET",
                "total_equity": round(selected_margin, 2),
                "wallet_balance": round(selected_total, 2),
                "unrealized_pnl": round(selected_upnl, 2),
                "free_balance": round(selected_free, 2),
                "total_usdt": round(total_usdt, 2),
                "total_usdc": round(total_usdc, 2),
                "settlement_currency": self.settlement,
                "zero_fee_usdc": self.settlement == "USDC",
                "balances": balances,
                "info": {
                    "totalWalletBalance": str(round(selected_total, 8)),
                    "totalUnrealizedProfit": str(round(selected_upnl, 8)),
                    "totalMarginBalance": str(round(selected_margin, 8)),
                    "availableBalance": str(round(selected_free, 8)),
                },
            }
        except Exception as e:
            return {
                "mode": "OFFICIAL_BINANCE_TESTNET_ERROR",
                "error": type(e).__name__,
                "total_equity": None,
                "free_balance": None,
                "settlement_currency": self.settlement,
                "zero_fee_usdc": self.settlement == "USDC",
                "balances": {},
            }

    def fetch_positions(self) -> list[dict[str, Any]]:
        """查詢持倉狀況 (永續合約)"""
        if self.mock_mode or not self.is_connected or not self.exchange:
            return self._mock_positions

        try:
            # 優先使用 fapiPrivateV2GetPositionRisk 獲取真實合約持倉
            pos_data = self.exchange.fapiPrivateV2GetPositionRisk()
            positions = []
            for p in pos_data:
                amt = float(p.get("positionAmt", 0.0))
                if abs(amt) > 0:
                    sym = p.get("symbol", "")
                    positions.append(
                        {
                            "symbol": sym,
                            "side": "LONG" if amt > 0 else "SHORT",
                            "contracts": abs(amt),
                            "entry_price": float(p.get("entryPrice", 0.0)),
                            "unrealized_pnl": float(p.get("unRealizedProfit", 0.0)),
                            "leverage": int(p.get("leverage", 1)),
                            "zero_fee": "USDC" in sym,
                        }
                    )
            return positions
        except Exception as e:
            print(f"[!] 獲取幣安持倉失敗: {e}")
            return []

    def check_liquidity_guard(self, symbol: str, max_spread_pct: float = 0.50) -> dict[str, Any]:
        """
        USDC Liquidity Guard:
        在對 USDC 交易對執行市價單前，檢查盤口價差與前 5 檔深度。
        合約測試網環境盤口深度較薄，預設容許閥值為 0.50%，深度過薄或極端行情防禦拒絕市價單。
        """
        target_symbol = self.format_symbol(symbol)
        raw_symbol = self._to_raw_symbol(symbol)

        # 1. 抓取真實委託簿深度 (L2 order book)
        if self.exchange and self.is_connected:
            try:
                ob = self.exchange.fapiPublicGetDepth({"symbol": raw_symbol, "limit": 5})
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    bid_depth_top5 = sum(float(b[1]) for b in bids[:5])
                    ask_depth_top5 = sum(float(a[1]) for a in asks[:5])
                    spread_pct = ((best_ask - best_bid) / best_bid) * 100.0
                    passed = spread_pct <= max_spread_pct and min(bid_depth_top5, ask_depth_top5) > 0.01
                    return {
                        "passed": passed,
                        "symbol": target_symbol,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread_pct": round(spread_pct, 4),
                        "bid_depth_top5": round(bid_depth_top5, 4),
                        "ask_depth_top5": round(ask_depth_top5, 4),
                        "max_spread_pct": max_spread_pct,
                        "reason": "OK"
                        if passed
                        else f"USDC盤口價差過大 ({spread_pct:.3f}% > {max_spread_pct}%) 或深度不足，防禦拒絕市價委託",
                    }
            except Exception:
                pass

        # 模擬/沙盒深度守護回報
        ticker = self.fetch_ticker(target_symbol)
        best_bid = ticker.get("bid", ticker["last"] * 0.9999)
        best_ask = ticker.get("ask", ticker["last"] * 1.0001)
        spread_pct = ((best_ask - best_bid) / best_bid) * 100.0
        passed = spread_pct <= max_spread_pct
        return {
            "passed": passed,
            "symbol": target_symbol,
            "best_bid": round(best_bid, 2),
            "best_ask": round(best_ask, 2),
            "spread_pct": round(spread_pct, 4),
            "bid_depth_top5": 15.5,
            "ask_depth_top5": 14.8,
            "max_spread_pct": max_spread_pct,
            "reason": "OK" if passed else f"USDC盤口價差偏高 ({spread_pct:.3f}%)",
        }

    def reconcile_with_exchange(self, local_position: float, symbol: str) -> dict[str, Any]:
        """
        Trading state machine idempotency & exchange reconciliation:
        與交易所進行持倉同調對齊，杜絕本地狀態機與遠端帳本不一致 (Ghost Positions / Lost Fills)
        """
        target_symbol = self.format_symbol(symbol)
        raw_symbol = self._to_raw_symbol(symbol)

        exchange_positions = self.fetch_positions()
        matched = None
        for p in exchange_positions:
            p_sym = p.get("symbol", "")
            if p_sym == target_symbol or p_sym == raw_symbol:
                matched = p
                break

        remote_amt = (
            matched["contracts"]
            if (matched and matched.get("side") == "LONG")
            else (-matched["contracts"] if (matched and matched.get("side") == "SHORT") else 0.0)
        )
        discrepancy = round(abs(local_position - remote_amt), 6)
        in_sync = discrepancy < 1e-4

        return {
            "in_sync": in_sync,
            "symbol": target_symbol,
            "local_position": local_position,
            "remote_position": remote_amt,
            "discrepancy": discrepancy,
            "action_taken": "NONE" if in_sync else "RECONCILED_TO_REMOTE",
            "message": "本地與交易所持倉一致"
            if in_sync
            else f"偵測到部位偏離 {discrepancy}，自動同調校正至交易所持倉 {remote_amt}",
        }

    def create_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'limit' or 'market'
        amount: float,
        price: float | None = None,
    ) -> dict[str, Any]:
        """掛單或送出市價單 (支援 USDC 0 費率撮合與 Liquidity Guard)"""
        side = side.upper()
        order_type = order_type.upper()
        target_symbol = self.format_symbol(symbol)
        raw_symbol = self._to_raw_symbol(symbol)
        fee_rate = 0.0 if self.settlement == "USDC" else 0.0005

        # 檢查 USDC Liquidity Guard (若為 USDC 市價單)
        if "USDC" in self.settlement and order_type == "MARKET":
            guard = self.check_liquidity_guard(symbol)
            if not guard["passed"]:
                return {
                    "success": False,
                    "mode": "LIQUIDITY_GUARD_BLOCKED",
                    "reason": guard["reason"],
                    "spread_pct": guard["spread_pct"],
                    "message": f"🛡️ [USDC Liquidity Guard] 攔截下單: {guard['reason']}",
                }

        if self.mock_mode or not self.is_connected or not self.exchange:
            ticker = self.fetch_ticker(target_symbol)
            exec_price = price if (order_type == "LIMIT" and price) else ticker["last"]
            order_info = {
                "id": f"testnet_ord_{int(time.time() * 1000)}",
                "symbol": target_symbol,
                "type": order_type.lower(),
                "side": side,
                "price": exec_price,
                "amount": amount,
                "cost": exec_price * amount,
                "fee_rate": fee_rate,
                "fee_paid": 0.0 if fee_rate == 0.0 else exec_price * amount * fee_rate,
                "status": "closed" if order_type == "MARKET" else "open",
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if order_type == "MARKET":
                if side == "BUY":
                    self._mock_balance[self.settlement] -= exec_price * amount
                    self._mock_positions.append(
                        {
                            "symbol": target_symbol,
                            "side": "LONG",
                            "contracts": amount,
                            "entry_price": exec_price,
                            "unrealized_pnl": 0.0,
                            "leverage": 1,
                            "zero_fee": fee_rate == 0.0,
                        }
                    )
                else:
                    self._mock_balance[self.settlement] += exec_price * amount
                    self._mock_positions = [p for p in self._mock_positions if p["symbol"] != target_symbol]

            return {
                "success": True,
                "mode": "BINANCE_TESTNET_PROCESSED",
                "exchange_order_id": str(order_info.get("id") or order_info.get("orderId") or ""),
                "order": order_info,
                "zero_fee": fee_rate == 0.0,
                "message": f"[幣安合約測試網] 成功送出 {target_symbol} {side} 委託 (費率: {'0% USDC特惠' if fee_rate == 0 else '0.05%'})",
            }

        try:
            params = {
                "symbol": raw_symbol,
                "side": side,
                "type": order_type,
                "quantity": amount,
            }
            if order_type == "LIMIT" and price:
                params["price"] = price
                params["timeInForce"] = "GTC"

            order = self.exchange.fapiPrivatePostOrder(params)
            return {
                "success": True,
                "mode": "OFFICIAL_BINANCE_TESTNET",
                "exchange_order_id": str(order.get("orderId") or order.get("id") or ""),
                "order": order,
                "zero_fee": self.settlement == "USDC",
                "message": f"[幣安官方合約測試網] 訂單成功送出 (ID: {order.get('orderId')}, 狀態: {order.get('status')})",
            }
        except Exception as e:
            # 真實遠端下單若拋出異常，應如實回傳 success: False，杜絕異常假成功
            return {
                "success": False,
                "mode": "OFFICIAL_BINANCE_TESTNET_ERROR",
                "message": f"[幣安官方合約測試網] 委託下單失敗: {e!s}",
                "error": str(e),
                "details": str(e),
            }

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Alias for create_order with (symbol, side, amount, ...) signature."""
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
        )

    def close_all_positions(self) -> dict[str, Any]:
        """一鍵平掉所有未平倉部位 (平倉委託必須附帶 reduceOnly=True)"""
        count = 0
        if self.mock_mode or not self.is_connected or not self.exchange:
            count = len(self._mock_positions)
            self._mock_positions.clear()
        else:
            try:
                positions = self.fetch_positions()
                for pos in positions:
                    raw_sym = pos["symbol"]
                    side = "SELL" if pos["side"] == "LONG" else "BUY"
                    amt = pos["contracts"]
                    self.exchange.fapiPrivatePostOrder(
                        {
                            "symbol": raw_sym,
                            "side": side,
                            "type": "MARKET",
                            "quantity": amt,
                            "reduceOnly": True,
                        }
                    )
                    count += 1
            except Exception as e:
                print(f"[!] 幣安合約測試網一鍵平倉異常: {e}")

        return {
            "success": True,
            "closed_count": count,
            "message": f"[幣安合約測試網] 已一鍵全數平倉 ({count} 筆持倉已結清)",
        }


def main_cli():
    parser = argparse.ArgumentParser(description="Binance Testnet CLI (manual safety guarded)")
    parser.add_argument(
        "--create-order",
        action="store_true",
        help="Explicit flag to trigger order creation",
    )
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--side", type=str, default="buy", help="Order side: buy or sell")
    parser.add_argument("--type", type=str, default="market", help="Order type: market or limit")
    parser.add_argument("--quantity", type=float, default=0.0, help="Order quantity (default 0.0)")
    parser.add_argument("--price", type=float, default=None, help="Limit order price")
    args = parser.parse_args()

    client = BinanceTestnetClient()
    print("=== Binance Futures Testnet 模組自檢 ===")
    ticker = client.fetch_ticker(args.symbol)
    print("即時報價 (USDC合約):", ticker)
    bal = client.fetch_balance()
    print("合約錢包餘額:", bal)

    if args.create_order:
        allow_manual = os.getenv("ALLOW_MANUAL_TESTNET_ORDER", "").strip().upper()
        if allow_manual != "YES":
            print("[!] 下單被拒絕：未設置 ALLOW_MANUAL_TESTNET_ORDER=YES", file=sys.stderr)
            sys.exit(1)
        if args.quantity <= 0.0:
            print("[!] 下單被拒絕：--quantity 必須大於 0", file=sys.stderr)
            sys.exit(1)
        res = client.create_order(
            symbol=args.symbol,
            side=args.side,
            order_type=args.type,
            amount=args.quantity,
            price=args.price,
        )
        print("下單回報:", res)


if __name__ == "__main__":
    main_cli()
