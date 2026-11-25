"""Core Smart Dice Board Game environment (agent chooses the dice value)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.board_generator import Board, validate_board

DEFAULT_REWARDS = {"goal": 100.0, "ladder": 20.0, "normal": -1.0, "snake": -20.0, "invalid": -10.0}


class DiceBoardEnv(gym.Env):
    """Snakes-and-ladders MDP where the agent *chooses* the dice outcome.

    Observation: Discrete(N). Observation index ``o`` represents board square ``o + 1``
        (Gymnasium Discrete spaces start at 0; square 1 -> 0, goal N -> N-1).
    Action: Discrete(6). Action ``a`` means dice value ``a + 1`` (0->1 ... 5->6).
    Invalid move (target > N): reward ``invalid``, position unchanged, episode continues
        (until ``max_episode_steps`` triggers truncation).
    Termination: reaching square N. Truncation: step budget exhausted.
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, board: Board, max_episode_steps: int = 200,
                 rewards: Optional[Dict[str, float]] = None, render_mode: Optional[str] = None):
        super().__init__()
        validate_board(board)
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        self.board = board
        self.n = board.size
        self.max_episode_steps = max_episode_steps
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(self.n)
        self.action_space = spaces.Discrete(6)
        self._jump = board.transitions()
        self.position = 1
        self.steps = 0

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def action_to_dice(action: int) -> int:
        return int(action) + 1

    def _obs(self) -> int:
        return self.position - 1

    def _info(self, **extra: Any) -> Dict[str, Any]:
        return {"position": self.position, "steps": self.steps, **extra}

    def _move(self, position: int, dice: int) -> Tuple[int, float, bool, str]:
        """Pure transition: returns (new_position, reward, goal_reached, event)."""
        target = position + dice
        if target > self.n:
            return position, self.rewards["invalid"], False, "invalid"
        if target == self.n:
            return target, self.rewards["goal"], True, "goal"
        if target in self.board.ladders:
            return self.board.ladders[target], self.rewards["ladder"], False, "ladder"
        if target in self.board.snakes:
            return self.board.snakes[target], self.rewards["snake"], False, "snake"
        return target, self.rewards["normal"], False, "normal"

    # ---- Gymnasium API -------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.position, self.steps = 1, 0
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action}; expected an integer in 0..5")
        dice = self.action_to_dice(action)
        self.position, reward, terminated, event = self._move(self.position, dice)
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_episode_steps
        return self._obs(), float(reward), terminated, truncated, self._info(event=event, dice=dice, is_success=bool(terminated))

    def render(self):
        if self.render_mode == "rgb_array":
            from visualization.board_visualizer import render_board_array
            return render_board_array(self.board, self.position)
        row = f"Position {self.position}/{self.n} | steps {self.steps}"
        if self.render_mode == "ansi":
            return row
        print(row)

    def close(self) -> None:
        pass
