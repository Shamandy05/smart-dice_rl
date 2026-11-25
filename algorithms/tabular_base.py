"""Shared machinery for tabular TD control (epsilon-greedy, Q-table, persistence)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class TabularAgent:
    """Tabular action-value learner with epsilon-greedy exploration.

    Subclasses implement :meth:`update`. ``epsilon`` decays multiplicatively once per episode
    via :meth:`decay_epsilon` down to ``epsilon_min``.
    """
    name = "tabular"

    def __init__(self, n_states: int, n_actions: int = 6, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.999,
                 seed: Optional[int] = None):
        if not (0 < alpha <= 1 and 0 <= gamma <= 1):
            raise ValueError("alpha must be in (0,1] and gamma in [0,1]")
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon_start, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)

    def greedy_action(self, state: int) -> int:
        """argmax_a Q(s,a) with random tie-breaking."""
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy: random action w.p. epsilon, otherwise greedy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, s: int, a: int, r: float, s_next: int, done: bool,
               a_next: Optional[int] = None) -> float:
        raise NotImplementedError

    def policy(self) -> np.ndarray:
        """Greedy policy (deterministic, first-index tie-break) for evaluation."""
        return self.q.argmax(axis=1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q)

    def load(self, path: str | Path) -> None:
        q = np.load(path)
        if q.shape != self.q.shape:
            raise ValueError(f"Q-table shape mismatch: {q.shape} vs {self.q.shape}")
        self.q = q
