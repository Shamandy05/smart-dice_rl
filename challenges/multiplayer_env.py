"""Challenge 4: two-player race. The learning agent chooses its dice; the opponent follows a rule."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.board_generator import Board
from environments.dice_board_env import DEFAULT_REWARDS, DiceBoardEnv


def greedy_opponent_dice(board: Board, pos: int, rng: np.random.Generator, kind: str = "greedy") -> int:
    """Rule-based opponent. 'random': uniform valid dice. 'greedy': goal > ladder > largest safe move."""
    valid = [d for d in range(1, 7) if pos + d <= board.size]
    if kind == "random":
        return int(rng.choice(valid))
    best, best_score = valid[0], -1e9
    for d in valid:
        t = pos + d
        if t == board.size:
            return d
        if t in board.ladders:
            score = 1000 + board.ladders[t]
        elif t in board.snakes:
            score = board.snakes[t] - 1000
        else:
            score = t
        if score > best_score:
            best, best_score = d, score
    return best


class TwoPlayerRaceEnv(DiceBoardEnv):
    """First player to reach square N wins. After each agent move the opponent moves.

    Observation: Discrete(N*N) = (agent_pos-1) * N + (opp_pos-1). A random player starts each game.
    Reward: official per-event reward for the agent's move; +0 extra on win (goal already +100);
    additional -100 if the opponent reaches the goal first (episode terminates).
    """

    def __init__(self, board: Board, opponent: str = "greedy", max_episode_steps: int = 200,
                 loss_penalty: float = -100.0, rewards: Optional[dict] = None):
        super().__init__(board, max_episode_steps, rewards)
        self.opponent, self.loss_penalty = opponent, loss_penalty
        self.opp_pos = 1
        self.observation_space = spaces.Discrete(self.n * self.n)
        self.agent_won: Optional[bool] = None

    def _obs(self) -> int:
        return (self.position - 1) * self.n + (self.opp_pos - 1)

    def _opp_move(self) -> bool:
        d = greedy_opponent_dice(self.board, self.opp_pos, self.np_random, self.opponent)
        self.opp_pos, _, reached, _ = self._move(self.opp_pos, d)
        return reached

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed)
        self.opp_pos, self.agent_won = 1, None
        if self.np_random.random() < 0.5 and self._opp_move():   # opponent starts (cannot win in 1 move)
            self.agent_won = False
        return self._obs(), {**info, "opp_position": self.opp_pos}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            self.agent_won = True
        elif not truncated and self._opp_move():
            terminated, reward, self.agent_won = True, reward + self.loss_penalty, False
        info = {**info, "is_success": bool(self.agent_won), "opp_position": self.opp_pos}
        return self._obs(), float(reward), terminated, truncated, info
