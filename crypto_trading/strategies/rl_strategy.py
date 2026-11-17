"""
Reinforcement Learning Policy Strategy Integration.
Connects trained PPO Agent into the BacktestEngine and BotRunner.
Supports continuous position weight [-1.0, 1.0] discretization or direct sizing.
"""

import json
import os

import numpy as np
import pandas as pd

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.strategies.momentum_breakout import BaseStrategy

try:
    from crypto_trading.rl_agent import PPOAgent
except ImportError:
    PPOAgent = None


class RLAlphaStrategy(BaseStrategy):
    """
    深度強化學習 (PPO) 自適應部位交易策略
    - 觀察當前多因子狀態、微觀結構以及自身帳戶部位損益狀態
    - 輸出最優持倉權重 [-1.0, 1.0]
    - 自動依據手續費與市場波動度調節換手率
    """

    def __init__(
        self,
        agent_weights_path: str | None = None,
        feature_engineer: CryptoFeatureEngineer | None = None,
        allow_unverified: bool = False,
    ):
        self.fe = feature_engineer or CryptoFeatureEngineer()
        self.agent = None
        self.weights_path = agent_weights_path
        self.allow_unverified = allow_unverified
        self.model_status = "MODEL_UNAVAILABLE"
        self.is_ready = False
        if self.weights_path and os.path.exists(self.weights_path):
            self.is_ready = True

    def _ensure_agent(self, state_dim: int):
        if PPOAgent is None:
            return False
        if self.agent is None:
            if not self.is_ready:
                return False
            # 檢查是否有對應 manifest 動態對齊特徵長度
            manifest_path = self.weights_path.replace(".pt", "_manifest.json")
            meta = {}
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "state_dim" in meta:
                        state_dim = int(meta["state_dim"])
                    elif "feature_columns" in meta:
                        state_dim = len(meta["feature_columns"]) + 2
                except Exception:
                    meta = {}
            if not self.allow_unverified:
                if not os.path.exists(manifest_path) or meta.get("is_synthetic") or meta.get("deployment_stage") == "RESEARCH_ONLY":
                    self.is_ready = False
                    self.model_status = "UNVERIFIED_MODEL_QUARANTINED"
                    return False
            self.agent = PPOAgent(state_dim=state_dim, action_dim=1)
            self.agent.load(self.weights_path)
            return True
        return True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not self.is_ready:
            df["signal"] = 0.0
            df["rl_action"] = 0.0
            df["model_status"] = getattr(self, "model_status", "MODEL_UNAVAILABLE")
            return df

        processed = self.fe.compute_features(df)
        feature_cols = self.fe.feature_columns

        # State dim = feature_cols + [position, unrealized_pnl]
        state_dim = len(feature_cols) + 2
        if not self._ensure_agent(state_dim):
            df["signal"] = 0.0
            df["rl_action"] = 0.0
            df["model_status"] = getattr(self, "model_status", "MODEL_UNAVAILABLE")
            return df

        signals = np.zeros(len(df))
        current_pos = 0.0
        entry_price = df["Close"].iloc[0]

        # 逐筆模擬決策
        for i in range(len(df)):
            curr_row = processed.iloc[i]
            if curr_row[feature_cols].isna().any():
                signals[i] = 0.0
                continue

            curr_price = curr_row["Close"]
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

        df["signal"] = signals
        return df
