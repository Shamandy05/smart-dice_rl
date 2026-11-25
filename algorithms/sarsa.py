"""Tabular SARSA (on-policy TD control). Implemented from scratch (not part of Stable-Baselines3)."""
from __future__ import annotations

from typing import Optional

from algorithms.tabular_base import TabularAgent


class SarsaAgent(TabularAgent):
    """Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ].

    On-policy: a' is the action actually chosen by the epsilon-greedy behaviour policy, so the
    learned values include the cost of future exploratory mistakes (e.g. stepping near snakes).
    """
    name = "SARSA"

    def update(self, s: int, a: int, r: float, s_next: int, done: bool,
               a_next: Optional[int] = None) -> float:
        if not done and a_next is None:
            raise ValueError("SARSA requires the next action a' for non-terminal transitions")
        target = r if done else r + self.gamma * self.q[s_next, a_next]
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        return float(td_error)
