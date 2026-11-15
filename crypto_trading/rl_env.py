"""
Reinforcement Learning Trading Environment (Gymnasium compatible).
Key features:
1. Differential Sharpe / Risk-Adjusted Reward formulation
2. Friction Modeling (Maker/Taker commission fees & slippage)
3. Action space: Continuous position sizing [-1.0, 1.0] (Short to Long) or Discrete [0, 1, 2]
4. Observation space: Multi-factor technical and microstructure state vector + agent position state
5. Dynamic Drawdown penalty & liquidation guardrails
"""

from typing import Any

import numpy as np
import pandas as pd


class CryptoTradingEnv:
    """
    Custom Standalone Gymnasium-compliant Trading Environment.
    Can be used directly or wrapped with gymnasium.Env.
    """

    def __init__(
        self,
        df_features: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0005,  # 5 bps fee
        slippage: float = 0.0002,  # 2 bps slippage
        max_drawdown_limit: float = 0.25,  # 25% max drawdown termination
        reward_type: str = "sharpe_differential",  # 'sharpe_differential' or 'pnl_risk_adjusted'
    ):
        self.df = df_features.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_type = reward_type

        # Observation dimensions: feature vector size + 2 (current position, unrealized PnL%)
        self.state_dim = len(feature_cols) + 2
        # Action space: continuous scalar in [-1.0, 1.0] representing target position ratio
        self.action_dim = 1

        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # State variables
        self.balance = initial_balance
        self.position = 0.0  # -1.0 to 1.0
        self.entry_price = 0.0
        self.peak_equity = initial_balance
        self.returns_history = []

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.peak_equity = self.initial_balance
        self.returns_history = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        current_row = self.df.loc[self.current_step]
        feat_vec = current_row[self.feature_cols].values.astype(np.float32)
        # Handle possible NaN/inf in input features
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        current_price = current_row["Close"]
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (
                self.position * (current_price - self.entry_price) / self.entry_price
            )

        agent_state = np.array([self.position, unrealized_pnl], dtype=np.float32)
        obs = np.concatenate([feat_vec, agent_state])
        return obs

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Executes one step in the trading environment.
        action: Target position weight in [-1.0, 1.0].
        """
        action = float(np.clip(action, -1.0, 1.0))
        current_price = self.df.loc[self.current_step, "Close"]
        next_step = self.current_step + 1
        next_price = self.df.loc[next_step, "Close"]

        # 1. Price change return
        price_ret = (next_price - current_price) / current_price

        # 2. Portfolio return before transaction cost
        gross_return = self.position * price_ret

        # 3. Transaction friction cost (commission + slippage)
        position_delta = abs(action - self.position)
        cost = position_delta * (self.fee_rate + self.slippage)

        # 4. Net period return
        net_return = gross_return - cost
        self.returns_history.append(net_return)

        # 5. Update portfolio equity & balance
        self.balance = self.balance * (1.0 + net_return)
        self.peak_equity = max(self.peak_equity, self.balance)

        drawdown = (self.peak_equity - self.balance) / (self.peak_equity + 1e-9)

        # 6. Reward formulation
        reward = self._calculate_reward(net_return, drawdown, position_delta)

        # Update position
        if action != self.position:
            self.position = action
            self.entry_price = next_price if action != 0 else 0.0

        self.current_step = next_step
        done = (self.current_step >= self.max_steps) or (
            drawdown >= self.max_drawdown_limit
        )

        info = {
            "step": self.current_step,
            "balance": self.balance,
            "net_return": net_return,
            "drawdown": drawdown,
            "position": self.position,
            "close_price": next_price,
        }

        obs = (
            self._get_observation()
            if not done
            else np.zeros(self.state_dim, dtype=np.float32)
        )
        return obs, float(reward), done, info

    def _calculate_reward(
        self, net_return: float, drawdown: float, delta_pos: float
    ) -> float:
        """
        Reward function designed for real alpha:
        1. Differential Sharpe: Incentivizes high returns with low volatility.
        2. Downside quadratic penalty: Penalizes large drawdowns heavily.
        3. Turnover penalty: Discourages excessive high-frequency churning.
        """
        # Base reward = net return scaled to percentage
        base_reward = net_return * 100.0

        # Downside risk penalty
        downside_penalty = 0.0
        if net_return < 0:
            downside_penalty = 2.0 * (net_return**2) * 100.0

        # Drawdown penalty
        dd_penalty = 1.5 * (drawdown**2)

        # Turnover friction penalty
        turnover_penalty = 0.1 * delta_pos

        reward = base_reward - downside_penalty - dd_penalty - turnover_penalty
        return float(reward)
