import time
from datetime import datetime
from typing import Optional
from crypto_trading.config import PAPER_TRADING, DEFAULT_SYMBOL, INITIAL_BALANCE_USDT
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy, BaseStrategy
from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
from crypto_trading.risk_manager import RiskManager
from crypto_trading.regime_filter import HierarchicalRegimeFilter
from data_persistence.gdrive_sync import GDrivePersistenceSync

class CryptoTradingBot:
    """
    虛擬貨幣自動操盤機器人核心
    支援：
    1. Paper Trading (高擬真模擬搓合帳戶)
    2. 即時價格監聽與技術/ML/RL 指標計算
    3. 自動風險管控與動態停損停利更新
    4. 支援經典動量、LightGBM/XGBoost ML Alpha、以及 PyTorch PPO RL 策略無縫切換
    """
    def __init__(self, symbol: str = DEFAULT_SYMBOL, strategy: Optional[BaseStrategy] = None):
        self.symbol = symbol
        self.is_paper = PAPER_TRADING
        self.loader = CryptoDataLoader(symbol=symbol)
        self.strategy = strategy or MomentumBreakoutStrategy()
        self.risk_manager = RiskManager(initial_capital=INITIAL_BALANCE_USDT)
        self.regime_filter = HierarchicalRegimeFilter(high_tf="1h", low_tf="1m")
        self.sync = GDrivePersistenceSync()

        # 模擬帳戶狀態
        self.balance_usdt = INITIAL_BALANCE_USDT
        self.position = 0.0  # 持倉標的數量
        self.position_side = "FLAT"  # FLAT, LONG, SHORT
        self.entry_price = 0.0
        self.trade_logs = []

    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    def execute_paper_order(self, side: str, amount: float, current_price: float, reason: str):
        """執行模擬訂單撮合"""
        cost = amount * current_price
        fee = cost * 0.0005  # 模擬 0.05% 手續費

        if side == "BUY":
            self.balance_usdt -= (cost + fee)
            self.position += amount
            self.position_side = "LONG"
            self.entry_price = current_price
            self.log(f"🟢 [模擬買入] {amount:.4f} {self.symbol} @ {current_price:.2f} | 總花費: {cost:.2f} USDT (手續費: {fee:.2f}) | 原因: {reason}")
        elif side == "SELL":
            revenue = cost - fee
            self.balance_usdt += revenue
            pnl = (current_price - self.entry_price) * self.position if self.position_side == "LONG" else 0.0
            pnl_pct = ((current_price - self.entry_price) / self.entry_price * 100.0) if self.entry_price > 0 else 0.0
            self.log(f"🔴 [模擬賣出/平倉] {self.position:.4f} {self.symbol} @ {current_price:.2f} | 損益 (PnL): {pnl:+.2f} USDT | 原因: {reason}")
            
            # 若發生實質虧損，自動記入 Post-Mortem 檢討資料庫
            if pnl < 0:
                self.sync.record_post_mortem(
                    symbol=self.symbol,
                    entry_price=self.entry_price,
                    exit_price=current_price,
                    loss_amount=pnl,
                    loss_pct=pnl_pct,
                    cause_analysis=f"平倉觸發原因: {reason}"
                )
            
            self.position = 0.0
            self.position_side = "FLAT"
            self.entry_price = 0.0

    def step(self):
        """單次決策迴圈步進 (整合高階趨勢濾網 Hierarchical Regime Filter)"""
        # 1. 抓取最新 K 線與行情
        df = self.loader.fetch_ohlcv(limit=60)
        df_strat = self.strategy.generate_signals(df)
        curr = df_strat.iloc[-1]
        current_price = float(curr['Close'])
        raw_signal = int(curr['signal'])
        atr = float(curr.get('ATR', current_price * 0.02))

        # 1.1 高階時框 (1h/1d) 市場狀態檢驗與信號濾波
        regime_info = self.regime_filter.detect_regime(df)
        signal = self.regime_filter.filter_micro_signal(raw_signal, regime_info)

        # 計算當前總資產淨值
        total_equity = self.balance_usdt + (self.position * current_price)
        self.risk_manager.update_capital(total_equity)

        # 2. 判斷是否有停損或平倉信號
        if self.position_side == "LONG":
            pos_risk = self.risk_manager.calculate_position_size(self.entry_price, atr)
            # 觸發動態停損
            if current_price <= pos_risk['stop_loss_long']:
                self.execute_paper_order("SELL", self.position, current_price, reason=f"觸發 ATR 移動停損 ({pos_risk['stop_loss_long']})")
                return
            # 觸發目標停利
            elif current_price >= pos_risk['take_profit_long']:
                self.execute_paper_order("SELL", self.position, current_price, reason=f"達到 1.5R 目標停利 ({pos_risk['take_profit_long']})")
                return

        # 3. 處理策略開平倉信號
        if signal == 1 and self.position_side == "FLAT":
            # 產生做多信號，計算部位並進場
            pos_plan = self.risk_manager.calculate_position_size(current_price, atr)
            if pos_plan['size'] > 0:
                self.execute_paper_order("BUY", pos_plan['size'], current_price, reason="布林帶動量突破向上")
        elif signal == 0 and self.position_side != "FLAT":
            # 產生離場信號
            self.execute_paper_order("SELL", self.position, current_price, reason="策略均線中軸平倉信號")

    def run_loop(self, iterations: int = 5, sleep_sec: int = 2):
        """啟動監聽迴圈 (預設先進行 5 次測試步進)"""
        self.log(f"🤖 虛擬貨幣量化操盤機器人已啟動！")
        self.log(f"⚙️ 模式: {'【模擬沙盒交易 Paper Trading】' if self.is_paper else '【真實實盤交易 Live Trading】'}")
        self.log(f"💰 當前初始資金: {self.balance_usdt:.2f} USDT | 監控交易對: {self.symbol}")
        self.log("=" * 60)

        for step_i in range(1, iterations + 1):
            self.log(f"--- 執行心跳檢測 Cycle {step_i}/{iterations} ---")
            self.step()
            total_equity = self.balance_usdt + (self.position * self.entry_price if self.position > 0 else 0)
            self.log(f"📊 當前帳戶淨值: {total_equity:.2f} USDT | 倉位狀態: {self.position_side} ({self.position:.4f} {self.symbol})")
            if step_i < iterations:
                time.sleep(sleep_sec)

        self.log("✅ 機器人週期自檢完畢，處於隨時可常駐監聽狀態。")

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run_loop(iterations=3, sleep_sec=1)
