"""
Reinforcement Learning PPO (Proximal Policy Optimization) Agent for Alpha Trading.
Implemented in pure PyTorch for universal compatibility (CPU / CUDA / Kaggle).
Architecture:
- Actor-Critic Shared Feature Extractor or Dual Network with LayerNorm & Tanh/Gaussian head
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective with Value Loss and Entropy Bonus
- Continuous action space [-1.0, 1.0] for position sizing
"""

import os

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        # Common feature backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        )

        # Actor head: Outputs continuous mean [-1.0, 1.0]
        self.actor_mean = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Mish(), nn.Linear(64, action_dim), nn.Tanh())
        # Learnable log standard deviation for exploration
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Critic head: Outputs state value scalar V(s)
        self.critic = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Mish(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        value = self.critic(features)
        return dist, value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c_val: float = 0.5,
        c_ent: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c_val = c_val
        self.c_ent = c_ent

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> tuple[float, float, float]:
        """
        Returns (action, log_prob, value)
        If evaluate=True, takes deterministic action (distribution mean).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.policy_old(state_t)
            if evaluate:
                action = dist.mean
            else:
                action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            float(action.cpu().numpy()[0, 0]),
            float(log_prob.cpu().numpy()[0]),
            float(value.cpu().numpy()[0, 0]),
        )

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        next_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0
        values_ext = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_ext[step + 1] * (1 - dones[step]) - values_ext[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def train_step(
        self,
        states: list[np.ndarray],
        actions: list[float],
        old_log_probs: list[float],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        k_epochs: int = 4,
        batch_size: int = 64,
    ) -> dict[str, float]:
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(-1).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.unsqueeze(-1)
        returns = returns.unsqueeze(-1)

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0

        n_samples = states_t.size(0)
        for _ in range(k_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = indices[start : start + batch_size]
                b_states = states_t[idx]
                b_actions = actions_t[idx]
                b_old_log_probs = old_log_probs_t[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]

                dist, value = self.policy(b_states)
                log_prob = dist.log_prob(b_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                ratios = torch.exp(log_prob - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value, b_returns)
                loss = policy_loss + self.c_val * value_loss - self.c_ent * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_accum += loss.item()
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        num_updates = max(1, k_epochs * (n_samples // batch_size + 1))

        return {
            "loss": total_loss_accum / num_updates,
            "policy_loss": policy_loss_accum / num_updates,
            "value_loss": value_loss_accum / num_updates,
        }

    def save(self, filepath: str, metadata: dict | None = None):
        import hashlib
        import json
        from datetime import datetime, timezone
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        sha = h.hexdigest()

        manifest = {
            "target": "crypto_ppo",
            "model_name": "crypto_ppo_agent",
            "version": "1.0.0",
            "sha256": sha,
            "weights_sha256": sha,
            "framework": "pytorch_ppo",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            manifest.update(metadata)
            manifest["sha256"] = sha
            manifest["weights_sha256"] = sha

        mf_path = os.path.splitext(filepath)[0] + "_manifest.json"
        with open(mf_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest

    def load(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
