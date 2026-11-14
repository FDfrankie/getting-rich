import pandas as pd
import numpy as np
from typing import Dict, Any
from crypto_trading.config import RISK_PARAMS

class RiskManager:
    """
    量化交易風控核心模組
    職責：
    1. 部位規模精確計算 (基於固定風險比例與 ATR 波動度)
    2. 動態追蹤停損 (Trailing Stop) 與固定目標停利 (Take Profit)
    3. 全域帳戶回撤熔斷監控 (Max Drawdown Circuit Breaker)
    """
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_limit = RISK_PARAMS["max_drawdown_stop_pct"]
        self.risk_per_trade = RISK_PARAMS["max_risk_per_trade_pct"]
        self.is_halted = False

    def update_capital(self, new_capital: float) -> bool:
        """更新資金並檢查是否觸發回撤熔斷"""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown >= self.max_drawdown_limit:
            self.is_halted = True
            print(f"[🚨 警報] 帳戶累計回撤達到 {drawdown*100:.2f}%，超過最大容忍限制 {self.max_drawdown_limit*100}%！啟動交易熔斷保護！")
            return False
        return True

    def calculate_position_size(self, current_price: float, atr: float) -> Dict[str, float]:
        """
        利用 ATR 與帳戶風險限額動態計算最佳下單數量與停損利點位
        公式：
        停損距離 = ATR * atr_stop_multiplier
        可承擔虧損金額 = 總資金 * risk_per_trade
        下單數量 = 可承擔虧損金額 / 停損距離
        """
        if self.is_halted:
            return {"size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}

        stop_distance = max(atr * RISK_PARAMS["atr_stop_multiplier"], current_price * 0.01)
        risk_amount = self.current_capital * self.risk_per_trade
        position_size = risk_amount / stop_distance

        # 限制單筆持倉名義價值不超過目前總資產的 50% (槓桿控制)
        max_notional = self.current_capital * 0.5
        if position_size * current_price > max_notional:
            position_size = max_notional / current_price

        return {
            "size": round(position_size, 4),
            "stop_loss_distance": round(stop_distance, 2),
            "stop_loss_long": round(current_price - stop_distance, 2),
            "stop_loss_short": round(current_price + stop_distance, 2),
            "take_profit_long": round(current_price + (stop_distance * 1.5), 2),
            "take_profit_short": round(current_price - (stop_distance * 1.5), 2)
        }

if __name__ == "__main__":
    rm = RiskManager(10000.0)
    pos_info = rm.calculate_position_size(current_price=65000.0, atr=1200.0)
    print("=== 風控部位與防守點位計算 ===")
    for k, v in pos_info.items():
        print(f"{k}: {v}")
