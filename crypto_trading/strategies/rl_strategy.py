"""
Reinforcement Learning Policy Strategy Integration.
Connects trained PPO Agent into the BacktestEngine and BotRunner.
Supports continuous position weight [-1.0, 1.0] discretization or direct sizing.
"""

import numpy as np
import pandas as pd
from typing import Optional
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from crypto_trading.rl_agent import PPOAgent
from crypto_trading.feature_engineer import CryptoFeatureEngineer

class RLAlphaStrategy(BaseStrategy):
    """
    深度強化學習 (PPO) 自適應部位交易策略
    - 觀察當前多因子狀態、微觀結構以及自身帳戶部位損益狀態
    - 輸出最優持倉權重 [-1.0, 1.0]
    - 自動依據手續費與市場波動度調節換手率
    """
    def __init__(self, agent_weights_path: Optional[str] = None, feature_engineer: Optional[CryptoFeatureEngineer] = None):
        self.fe = feature_engineer or CryptoFeatureEngineer()
        self.agent = None
        self.weights_path = agent_weights_path

    def _ensure_agent(self, state_dim: int):
        if self.agent is None:
            self.agent = PPOAgent(state_dim=state_dim, action_dim=1)
            if self.weights_path:
                self.agent.load(self.weights_path)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        processed = self.fe.compute_features(df)
        feature_cols = self.fe.feature_columns

        # State dim = feature_cols + [position, unrealized_pnl]
        state_dim = len(feature_cols) + 2
        self._ensure_agent(state_dim)

        signals = np.zeros(len(df))
        current_pos = 0.0
        entry_price = df['Close'].iloc[0]

        # 逐筆模擬決策
        for i in range(len(df)):
            curr_row = processed.iloc[i]
            if curr_row[feature_cols].isna().any():
                signals[i] = 0.0
                continue

            curr_price = curr_row['Close']
            unrealized_pnl = 0.0
            if current_pos != 0 and entry_price > 0:
                unrealized_pnl = current_pos * (curr_price - entry_price) / entry_price

            feat_vec = curr_row[feature_cols].values.astype(np.float32)
            feat_vec = np.nan_to_num(feat_vec, nan=0.0)
            obs = np.concatenate([feat_vec, [current_pos, unrealized_pnl]])

            # PPO 評估模式生成動作
            action, _, _ = self.agent.select_action(obs, evaluate=True)
            
            # 將連續動作轉換為持倉 (-1, 0, 1) 或依門檻濾波
            if action > 0.3:
                pos_signal = 1.0
            elif action < -0.3:
                pos_signal = -1.0
            else:
                pos_signal = 0.0

            if pos_signal != current_pos:
                current_pos = pos_signal
                entry_price = curr_price

            signals[i] = pos_signal

        df['signal'] = signals
        return df
