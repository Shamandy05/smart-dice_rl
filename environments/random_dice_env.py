"""Challenge 1 (random dice), Challenge 2 (limited special action), Challenge 3 (dynamic board).

Random dice MDP reformulation
-----------------------------
The dice value is no longer an action but a stochastic component of the transition kernel
P(s' | s, a). ``RandomDiceEnv`` offers a decision at each turn so the MDP remains non-trivial:
Without special actions the agent has no decision at all (a Markov reward process); with the
shield (Challenge 2) it decides, after seeing the roll, whether to spend its single shield.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from environments.board_generator import Board, generate_board
from environments.dice_board_env import DiceBoardEnv


class RandomDiceEnv(DiceBoardEnv):
    """Environment-supplied random dice (+ optional limited snake shield, Challenge 2).

    The dice value is part of the environment's randomness, not an action. To make the shield
    decision meaningful the *rolled* dice value is shown to the agent before it acts:

    * ``special_actions=False``: action space Discrete(1) (no decisions; a Markov reward process),
      observation Discrete(N) = position.
    * ``special_actions=True``: action 0 = "move as rolled", action 1 = "use shield if the roll
      lands on a snake" (cancels that snake once; the player stays on the snake head square and
      pays the normal step cost). Action 1 is a no-op when no snake is hit or no shield remains.
      Observation Discrete(N * 6 * (shields+1)) encoding (shields_left, position, rolled dice):
      ``(shields_left * N + (position-1)) * 6 + (dice-1)``. The remaining special-action count is
      therefore part of the state, as required by the project description.
    """

    def __init__(self, board: Board, max_episode_steps: int = 200, rewards=None,
                 special_actions: bool = False, shields: int = 1, render_mode=None):
        super().__init__(board, max_episode_steps, rewards, render_mode)
        self.special_actions = special_actions
        self.initial_shields = shields if special_actions else 0
        self.shields = self.initial_shields
        self.next_dice = 1
        self.action_space = spaces.Discrete(2 if special_actions else 1)
        n_obs = self.n * 6 * (self.initial_shields + 1) if special_actions else self.n
        self.observation_space = spaces.Discrete(n_obs)

    def _obs(self) -> int:
        if not self.special_actions:
            return self.position - 1
        return (self.shields * self.n + (self.position - 1)) * 6 + (self.next_dice - 1)

    def _roll(self) -> None:
        self.next_dice = int(self.np_random.integers(1, 7))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.shields = self.initial_shields
        self._roll()
        return self._obs(), {"position": 1, "steps": 0, "shields": self.shields, "dice_rolled": self.next_dice}

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action {action}")
        dice = self.next_dice
        target = self.position + dice
        if action == 1 and self.shields > 0 and target in self.board.snakes:
            self.shields -= 1                      # shield cancels this snake (stay on the head)
            self.position, reward, terminated, event = target, self.rewards["normal"], False, "shield"
        else:
            self.position, reward, terminated, event = self._move(self.position, dice)
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_episode_steps
        self._roll()
        return (self._obs(), float(reward), terminated, truncated,
                self._info(event=event, dice=dice, shields=self.shields, is_success=bool(terminated)))


class DynamicBoardEnv(DiceBoardEnv):
    """Challenge 3: a new random board every episode (agent chooses dice).

    The observation is a Box so the agent can *see* the board: position/N plus, for the 6 squares
    ahead, a code for what each reachable target holds (+1 ladder, -1 snake, 0 normal, -2 invalid)
    and the relative jump length. This gives the policy enough information to generalise.
    """

    def __init__(self, size: int, n_snakes: int, n_ladders: int, base_seed: int = 0,
                 max_episode_steps: int = 200, rewards=None, fixed_pool: Optional[int] = None):
        board = generate_board(size, n_snakes, n_ladders, seed=base_seed)
        super().__init__(board, max_episode_steps, rewards)
        self.size_, self.ns, self.nl, self.base_seed = size, n_snakes, n_ladders, base_seed
        self.fixed_pool = fixed_pool          # if set, boards are drawn from a pool of this size
        self.episode_count = 0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(13,), dtype=np.float32)

    def _obs(self):  # type: ignore[override]
        o = np.zeros(13, dtype=np.float32)
        o[0] = (self.position - 1) / (self.n - 1)
        for d in range(1, 7):
            t = self.position + d
            if t > self.n:
                o[d], o[6 + d] = -1.0, 0.0
            elif t in self.board.ladders:
                o[d], o[6 + d] = 1.0, (self.board.ladders[t] - t) / self.n
            elif t in self.board.snakes:
                o[d], o[6 + d] = -0.5, (self.board.snakes[t] - t) / self.n
        return o

    def reset(self, *, seed=None, options=None):
        gym_seed = seed
        if gym_seed is not None:
            self.episode_count = 0
        k = self.episode_count % self.fixed_pool if self.fixed_pool else self.episode_count
        self.board = generate_board(self.size_, self.ns, self.nl, seed=self.base_seed + 1000 + k)
        self._jump = self.board.transitions()
        self.episode_count += 1
        return super().reset(seed=seed, options=options)
