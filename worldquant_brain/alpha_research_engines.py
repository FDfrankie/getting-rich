"""Research-backed Alpha Discovery Engines for slow, high-precision BRAIN mining.

Implements:
1. Sliding-Window UCB Multi-Armed Bandit (DomainBanditAllocator) over 7 data domains
   (ref: https://arxiv.org/abs/2406.07451).
2. MCTS-style UCT Formula Tree Search (FormulaMCTSSearch)
   (ref: Alpha-Jungle https://arxiv.org/html/2505.11122v1).
3. Warm-Start Structural Mutation & DiCo Originality/Complementarity Penalty
   (ref: Warm-Start GP https://arxiv.org/html/2412.00896v1,
         AlphaAgent https://arxiv.org/html/2502.16789v2,
         QuantEvolver https://arxiv.org/html/2605.15412v1).
"""
from __future__ import annotations

import math
import os
import random
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence


DEFAULT_DOMAINS: tuple[str, ...] = (
    "FUNDAMENTAL_CF_EV",
    "FUNDAMENTAL_ROA_DELEVERAGE",
    "ANALYST_EPS_DISPERSION",
    "SENTIMENT_BUZZ_MOMENTUM",
    "OPTIONS_IV_SLOPE",
    "PV_RESIDUAL_HUMP",
    "VOLATILITY_LIQ_DECAY",
)


def tokenize_expression(expr: str) -> set[str]:
    """Extract operator and data-field tokens for Jaccard originality/DiCo similarity checks."""
    return set(re.findall(r"[a-z_][a-z0-9_]*", (expr or "").lower()))


def compute_dico_similarity_penalty(
    candidate_expr: str,
    existing_expressions: Sequence[str],
    *,
    max_penalty: float = 45.0,
) -> float:
    """QuantEvolver DiCo + AlphaAgent Originality Regularization:
    Penalize candidates that share high token/structural overlap with existing vault/pool formulas.
    """
    cand_tokens = tokenize_expression(candidate_expr)
    if not cand_tokens or not existing_expressions:
        return 0.0
    max_sim = 0.0
    for ex in existing_expressions:
        ex_toks = tokenize_expression(ex)
        if not ex_toks:
            continue
        inter = len(cand_tokens & ex_toks)
        union = len(cand_tokens | ex_toks)
        if union > 0:
            sim = inter / float(union)
            if sim > max_sim:
                max_sim = sim
    # Only penalize strongly once similarity exceeds 0.60
    if max_sim <= 0.60:
        return round(max_sim * 10.0, 2)
    excess = (max_sim - 0.60) / 0.40
    return round(6.0 + excess * (max_penalty - 6.0), 2)


def warm_start_structural_mutate(
    seed_expr: str,
    *,
    domain: str = "",
) -> str | None:
    """Warm-Start GP (arXiv:2412.00896): swap equivalent fields/windows inside a proven structure
    without increasing tree depth or nesting ts_decay_linear."""
    expr = (seed_expr or "").strip()
    if not expr or "ts_decay_linear(ts_decay_linear(" in expr.lower():
        return None
    field_swaps = {
        "operating_cashflow": ("operating_income", "ebitda"),
        "operating_income": ("operating_cashflow", "ebitda"),
        "ebitda": ("operating_cashflow", "operating_income"),
        "subindustry": ("industry",),
        "industry": ("subindustry",),
        "returns": ("ts_delta(close, 1) / (close + 0.001)",),
    }
    mutated = expr
    swapped = False
    for src, targets in field_swaps.items():
        if re.search(rf"\b{re.escape(src)}\b", mutated):
            mutated = re.sub(rf"\b{re.escape(src)}\b", random.choice(targets), mutated, count=1)
            swapped = True
            break
    # Also perturb window constants safely within [10, 20, 40, 60, 120]
    nums = re.findall(r"\b(10|20|40|60|120)\b", mutated)
    if nums:
        old_w = nums[0]
        choices = [w for w in ("10", "20", "40", "60", "120") if w != old_w]
        if choices:
            mutated = re.sub(rf"\b{old_w}\b", random.choice(choices), mutated, count=1)
            swapped = True
    return mutated if (swapped and mutated != expr) else None


class DomainBanditAllocator:
    """Sliding-Window UCB Multi-Armed Bandit for allocating 3-minute BRAIN simulations across data domains."""

    def __init__(
        self,
        db_path: str | None = None,
        domains: Sequence[str] = DEFAULT_DOMAINS,
        window_size: int = 40,
        exploration_c: float = 0.85,
    ) -> None:
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "alpha_vault.db")
        self.domains = tuple(domains)
        self.window_size = int(window_size)
        self.exploration_c = float(exploration_c)
        self._init_db()

    def _init_db(self) -> None:
        if self.db_path == ":memory:":
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS brain_domain_bandit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    reward REAL NOT NULL,
                    sharpe REAL,
                    passed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_outcome(self, domain: str, *, sharpe: float | None, passed: bool, self_corr: float | None = None) -> float:
        if domain not in self.domains:
            domain = self.domains[0]
        sr = float(sharpe) if sharpe is not None else 0.0
        reward = 0.0
        if passed:
            reward += 0.60
        if sr >= 1.25:
            reward += min(0.35, (sr - 1.0) * 0.25)
        if self_corr is not None and self_corr >= 0.70:
            reward -= 0.40
        reward = max(0.0, min(1.0, reward))
        if self.db_path != ":memory:":
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    "INSERT INTO brain_domain_bandit (domain, reward, sharpe, passed, created_at) VALUES (?, ?, ?, ?, ?)",
                    (domain, reward, sr, 1 if passed else 0, now),
                )
        return round(reward, 4)

    def select_domain(self) -> str:
        if self.db_path == ":memory:" or not os.path.exists(self.db_path):
            return random.choice(self.domains)
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT domain, reward FROM brain_domain_bandit ORDER BY id DESC LIMIT ?",
                (self.window_size,),
            ).fetchall()
        counts: dict[str, int] = {d: 0 for d in self.domains}
        sums: dict[str, float] = {d: 0.0 for d in self.domains}
        for dom, rew in rows:
            if dom in counts:
                counts[dom] += 1
                sums[dom] += float(rew)
        # Ensure every domain is explored at least once in the sliding window
        unexplored = [d for d in self.domains if counts[d] == 0]
        if unexplored:
            return unexplored[0]
        total_n = max(1, sum(counts.values()))
        best_dom = self.domains[0]
        best_ucb = -1.0
        for d in self.domains:
            mean_r = sums[d] / counts[d]
            ucb = mean_r + self.exploration_c * math.sqrt(math.log(total_n) / counts[d])
            if ucb > best_ucb:
                best_ucb = ucb
                best_dom = d
        return best_dom


class FormulaMCTSSearch:
    """Lightweight UCT Monte Carlo Tree Search over (domain, transform, neutralization) nodes."""

    TRANSFORMS: tuple[str, ...] = ("WINSORIZE_HUMP", "TS_REGRESSION_RESID", "ZSCORE_RANK")
    NEUTRALIZATIONS: tuple[str, ...] = ("subindustry", "industry")

    def __init__(self, exploration_weight: float = 1.1) -> None:
        self.exploration_weight = float(exploration_weight)
        self.visits: dict[tuple[str, str, str], int] = {}
        self.rewards: dict[tuple[str, str, str], float] = {}

    def select_branch(self, domain: str) -> tuple[str, str, str]:
        branches = [(domain, t, n) for t in self.TRANSFORMS for n in self.NEUTRALIZATIONS]
        unvisited = [b for b in branches if self.visits.get(b, 0) == 0]
        if unvisited:
            return random.choice(unvisited)
        total_v = sum(self.visits.get(b, 0) for b in branches)
        best_b = branches[0]
        best_uct = -1.0
        for b in branches:
            v = self.visits[b]
            q = self.rewards[b] / v
            uct = q + self.exploration_weight * math.sqrt(math.log(max(1, total_v)) / v)
            if uct > best_uct:
                best_uct = uct
                best_b = b
        return best_b

    def backpropagate(self, branch: tuple[str, str, str], reward: float) -> None:
        self.visits[branch] = self.visits.get(branch, 0) + 1
        self.rewards[branch] = self.rewards.get(branch, 0.0) + float(reward)
