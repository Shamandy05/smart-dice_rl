"""Tabular Q-Learning (off-policy TD control)."""
from __future__ import annotations

from typing import Optional

from algorithms.tabular_base import TabularAgent


class QLearningAgent(TabularAgent):
    """Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ].

    Off-policy: the bootstrap target uses the greedy action in s', regardless of which action
    the epsilon-greedy behaviour policy will actually take next.
    """
    name = "Q-Learning"

    def update(self, s: int, a: int, r: float, s_next: int, done: bool,
               a_next: Optional[int] = None) -> float:
        target = r if done else r + self.gamma * self.q[s_next].max()
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        return float(td_error)
