"""Episode-based training for tabular Q-Learning and SARSA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np

from algorithms import QLearningAgent, SarsaAgent
from algorithms.tabular_base import TabularAgent
from environments.board_generator import Board
from evaluation.metrics import evaluate_policy, tabular_act_fn
from training.config import ROOT, get_board, load_config, make_env

AGENTS: Dict[str, Type[TabularAgent]] = {"Q-Learning": QLearningAgent, "SARSA": SarsaAgent}


def train_tabular(algo: str, board: Board, cfg: Dict[str, Any], exp: str, seed: int,
                  episodes: int | None = None, verbose: bool = False, max_timesteps: int | None = None,
                  train_env_factory=None, epsilon_decay: float | None = None):
    """Train ``algo`` for ``episodes`` episodes. Returns (agent, history).

    The training env and the evaluation env are separate objects with different seeds.
    History holds per-training-episode logs (epsilon-greedy behaviour) and periodic greedy
    evaluation points on the evaluation env.
    """
    t = cfg["tabular"]
    episodes = episodes or t["episodes"][exp]
    decay = epsilon_decay or t.get("epsilon_decay_by_board", {}).get(exp, t["epsilon_decay"])
    # Optional hooks (used by the challenges): a different *training* env (e.g. shaped rewards) and a
    # timestep budget. Evaluation always uses the official env/rewards from ``cfg``.
    train_env = train_env_factory() if train_env_factory else make_env(board, cfg)
    eval_env = make_env(board, cfg)
    if max_timesteps:
        episodes = episodes or 10**9
    agent = AGENTS[algo](board.size, 6, t["alpha"], t["gamma"], t["epsilon_start"], t["epsilon_min"],
                         decay, seed=seed)
    hist: Dict[str, Any] = {"algo": algo, "exp": exp, "seed": seed, "episode_reward": [], "episode_moves": [],
                            "episode_success": [], "epsilon": [], "eval": []}
    ev = cfg["evaluation"]
    start = time.time()
    for ep in range(1, episodes + 1):
        s, _ = train_env.reset(seed=seed * 100003 + ep)
        a = agent.select_action(s)
        total, steps, done, ok = 0.0, 0, False, False
        while not done:
            s2, r, terminated, truncated, _ = train_env.step(a)
            # Bootstrapping stops only on true termination (goal); truncation is a time limit.
            a2 = None if terminated else agent.select_action(s2)
            agent.update(s, a, r, s2, terminated, a2)
            total += r
            steps += 1
            ok = ok or terminated
            done = terminated or truncated
            s, a = s2, a2
        hist["episode_reward"].append(total)
        hist["episode_moves"].append(steps)
        hist["episode_success"].append(bool(ok))
        hist["epsilon"].append(agent.epsilon)
        agent.decay_epsilon()
        out_of_budget = bool(max_timesteps) and int(np.sum(hist["episode_moves"])) >= max_timesteps
        if ep % ev["interval_episodes"] == 0 or ep == episodes or out_of_budget:
            m = evaluate_policy(eval_env, tabular_act_fn(agent), ev["curve_episodes"], ev["seed"])
            hist["eval"].append({"episodes": ep, "timesteps": int(np.sum(hist["episode_moves"])), **m})
            if verbose:
                print(f"[{algo}] ep {ep}/{episodes} eps={agent.epsilon:.3f} "
                      f"eval success={m['success_rate']:.0f}% moves={m['mean_moves']:.1f}")
        if out_of_budget:
            break
    hist["train_seconds"] = time.time() - start
    return agent, hist


def main() -> None:
    p = argparse.ArgumentParser(description="Train tabular Q-Learning / SARSA")
    p.add_argument("--algo", choices=list(AGENTS), required=True)
    p.add_argument("--exp", choices=["exp1", "exp2", "exp3"], default="exp1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    board = get_board(cfg, a.exp)
    agent, hist = train_tabular(a.algo, board, cfg, a.exp, a.seed, a.episodes, verbose=True)
    out = ROOT / "results" / "trained_models"
    agent.save(out / f"{a.exp}_{a.algo}_seed{a.seed}_qtable.npy")
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs" / f"{a.exp}_{a.algo}_seed{a.seed}.json").write_text(json.dumps(hist))
    print("saved Q-table and history")


if __name__ == "__main__":
    main()
