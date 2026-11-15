import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from crypto_trading.testnet_client import BinanceTestnetClient


def check_live():
    client = BinanceTestnetClient()
    print("=== LIVE TESTNET ACCOUNT CHECK ===")
    balances = client.fetch_balance()
    print("Balances:", balances)
    positions = client.fetch_positions()
    print("Real Positions (positionAmt != 0):", positions)
    if client.exchange:
        try:
            raw_orders = client.exchange.fapiPrivateGetOpenOrders()
            print("Raw Open Orders from Binance Testnet:", raw_orders)
            all_pos = client.exchange.fapiPrivateV2GetPositionRisk()
            non_zero_pos = [p for p in all_pos if float(p.get("positionAmt", 0)) != 0]
            print("Real Non-Zero Position Risk entries:", non_zero_pos)
        except Exception as e:
            print("Error checking raw endpoints:", e)


if __name__ == "__main__":
    check_live()
