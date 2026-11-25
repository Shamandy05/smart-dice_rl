"""Evaluation of policies on *separate* environments (never the training env)."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def evaluate_policy(env, act_fn: Callable[[int], int], n_episodes: int, seed: int) -> Dict[str, float]:
    """Roll out ``act_fn`` for ``n_episodes`` on ``env`` (reset seeded once with ``seed``).

    Returns mean moves (all episodes), mean moves among successful games, success rate (%),
    mean cumulative reward and their standard deviations over episodes.
    """
    moves: List[int] = []
    rewards: List[float] = []
    successes: List[bool] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        done, total, steps, ok = False, 0.0, 0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(int(act_fn(obs)))
            total += r
            steps += 1
            ok = ok or terminated
            done = terminated or truncated
        moves.append(steps)
        rewards.append(total)
        successes.append(ok)
    m, s = np.array(moves, float), np.array(successes, bool)
    return {"mean_moves": float(m.mean()), "std_moves": float(m.std()),
            "mean_moves_success": float(m[s].mean()) if s.any() else float("nan"),
            "success_rate": float(s.mean() * 100.0),
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "n_episodes": n_episodes}


def random_policy(env, n_episodes: int, seed: int) -> Dict[str, float]:
    """Uniform-random dice selection baseline (the 'before training' policy)."""
    rng = np.random.default_rng(seed)
    return evaluate_policy(env, lambda _o: int(rng.integers(env.action_space.n)), n_episodes, seed)


def sb3_act_fn(model) -> Callable[[int], int]:
    """Deterministic (greedy) action function for a Stable-Baselines3 model."""
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def tabular_act_fn(agent) -> Callable[[int], int]:
    q = agent.q
    return lambda obs: int(q[int(obs)].argmax())
