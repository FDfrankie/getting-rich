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
        reward_type: str = "sharpe_differential",
        position_deadband: float = 0.15,
        min_holding_bars: int = 30,
        rebalance_cooldown: int = 15,
        funding_rate_per_bar: float = 0.0,
        turnover_penalty_rate: float = 0.1,
        drawdown_penalty_rate: float = 1.5,
    ):
        self.df = df_features.reset_index(drop=True)
        self._feat_arr = np.nan_to_num(
            self.df[feature_cols].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        self._close_arr = self.df["Close"].to_numpy(dtype=np.float64)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_type = reward_type
        self.position_deadband = float(position_deadband)
        self.min_holding_bars = int(min_holding_bars)
        self.rebalance_cooldown = int(rebalance_cooldown)
        self.funding_rate_per_bar = float(funding_rate_per_bar)
        self.turnover_penalty_rate = float(turnover_penalty_rate)
        self.drawdown_penalty_rate = float(drawdown_penalty_rate)

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
        self.bars_in_position = 0
        self.bars_since_rebalance = self.rebalance_cooldown
        self._dsr_eta: float = 0.01
        self._dsr_a: float = 0.0
        self._dsr_b: float = 0.0
        self._prev_drawdown: float = 0.0
        self.impact_coeff: float = 0.0003

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.peak_equity = self.initial_balance
        self.returns_history = []
        self.bars_in_position = 0
        self.bars_since_rebalance = self.rebalance_cooldown
        self._dsr_eta: float = 0.01
        self._dsr_a: float = 0.0
        self._dsr_b: float = 0.0
        self._prev_drawdown: float = 0.0
        self.impact_coeff: float = 0.0003
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        feat_vec = self._feat_arr[self.current_step]
        current_price = float(self._close_arr[self.current_step])
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price

        agent_state = np.array([self.position, unrealized_pnl], dtype=np.float32)
        obs = np.concatenate([feat_vec, agent_state])
        return obs

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Executes one step in the trading environment.
        action: Target position weight in [-1.0, 1.0].
        """
        proposed_action = float(np.clip(action, -1.0, 1.0))
        delta = proposed_action - self.position
        action = proposed_action
        blocked_reason = None
        if abs(delta) < self.position_deadband:
            action = self.position
            blocked_reason = "DEADBAND"
        elif self.position != 0.0 and self.bars_in_position < self.min_holding_bars:
            action = self.position
            blocked_reason = "MIN_HOLDING"
        elif self.bars_since_rebalance < self.rebalance_cooldown:
            action = self.position
            blocked_reason = "COOLDOWN"
        current_price = float(self._close_arr[self.current_step])
        next_step = self.current_step + 1
        next_price = float(self._close_arr[next_step])

        # 1. Price change return
        price_ret = (next_price - current_price) / current_price

        # 2. Transaction friction + nonlinear market impact cost (Almgren-Chriss proxy)
        position_delta = abs(action - self.position)
        linear_cost = position_delta * (self.fee_rate + self.slippage)
        impact_cost = self.impact_coeff * (position_delta ** 1.5)
        trading_cost = linear_cost + impact_cost
        funding_cost = abs(action) * self.funding_rate_per_bar
        cost = trading_cost + funding_cost

        # 3. Causal portfolio return on executed target position
        gross_return = action * price_ret
        net_return = gross_return - cost
        self.returns_history.append(net_return)

        # 4. Update portfolio equity, balance & incremental drawdown
        self.balance = self.balance * (1.0 + net_return)
        self.peak_equity = max(self.peak_equity, self.balance)
        drawdown = (self.peak_equity - self.balance) / (self.peak_equity + 1e-9)
        delta_dd = max(0.0, drawdown - self._prev_drawdown)
        self._prev_drawdown = drawdown

        # 5. Reward formulation (online DSR + incremental drawdown penalty)
        reward = self._calculate_reward(net_return, delta_dd, position_delta)

        # Update position
        if action != self.position:
            self.position = action
            self.entry_price = next_price if action != 0 else 0.0
            self.bars_in_position = 0
            self.bars_since_rebalance = 0
        else:
            self.bars_since_rebalance += 1
            if self.position != 0.0:
                self.bars_in_position += 1

        self.current_step = next_step
        done = (self.current_step >= self.max_steps) or (drawdown >= self.max_drawdown_limit)

        info = {
            "step": self.current_step,
            "balance": self.balance,
            "net_return": net_return,
            "drawdown": drawdown,
            "position": self.position,
            "close_price": next_price,
            "proposed_position": proposed_action,
            "executed_position": self.position,
            "action_blocked_reason": blocked_reason,
            "turnover": position_delta,
            "trading_cost": trading_cost,
            "funding_cost": funding_cost,
            "bars_in_position": self.bars_in_position,
            "bars_since_rebalance": self.bars_since_rebalance,
        }

        obs = self._get_observation() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return obs, float(reward), done, info

    def _calculate_reward(self, net_return: float, delta_dd: float, delta_pos: float) -> float:
        """Online Differential Sharpe Ratio (Moody & Saffell) + incremental drawdown & turnover penalty."""
        if self.reward_type == "sharpe_differential":
            dA = net_return - self._dsr_a
            dB = (net_return ** 2) - self._dsr_b
            denom = max(self._dsr_b - (self._dsr_a ** 2), 1e-8) ** 1.5
            dsr = (self._dsr_b * dA - 0.5 * self._dsr_a * dB) / denom if self.current_step > 5 else net_return * 100.0
            self._dsr_a += self._dsr_eta * dA
            self._dsr_b += self._dsr_eta * dB
            base_reward = float(np.clip(dsr, -10.0, 10.0))
        else:
            base_reward = net_return * 100.0

        dd_penalty = self.drawdown_penalty_rate * delta_dd * 100.0
        turnover_penalty = self.turnover_penalty_rate * delta_pos
        return float(base_reward - dd_penalty - turnover_penalty)
