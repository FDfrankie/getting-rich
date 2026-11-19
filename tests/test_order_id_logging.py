from __future__ import annotations

from crypto_trading.bot_runner import CryptoTradingBot
from crypto_trading.testnet_client import BinanceTestnetClient


def test_mock_testnet_order_sets_exchange_order_id() -> None:
    client = BinanceTestnetClient(settlement="USDC")
    client.mock_mode = True
    res = client.place_order("BTC/USDT", "BUY", 0.01, order_type="MARKET", price=65000.0)
    assert res["success"] is True
    assert res["exchange_order_id"]


def test_bot_runner_extracts_nested_binance_order_id() -> None:
    class _StubAdapter:
        def fetch_ticker(self, symbol: str):
            return {"last": 65000.0}

        def place_order(self, symbol: str, side: str, amount: float, **kwargs):
            return {
                "success": True,
                "price": 65000.0,
                "order": {"orderId": 987654321},
            }

    class _ReadyRecon:
        def is_ready_for_trading(self) -> bool:
            return True

    bot = CryptoTradingBot(symbol="BTC/USDT", exchange_adapter=_StubAdapter(), execution_mode="BINANCE_TESTNET")
    bot.reconciliation_service = _ReadyRecon()
    bot.execute_order("BUY", 0.01, 65000.0, reason="unit_test")
    assert any("987654321" in msg for msg in bot.trade_logs)
