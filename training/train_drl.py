"""Timestep-based training for DQN / PPO / A2C with Stable-Baselines3.

Relationship between timesteps and episodes: one *timestep* = one dice choice (env.step).
One *episode* = one game (<= max_episode_steps timesteps). Episodes completed is therefore
approximately timesteps / (average game length) and is logged explicitly by the callback so that
every curve can be plotted against training episodes as well as timesteps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from algorithms import BUILDERS
from environments.board_generator import Board
from evaluation.metrics import evaluate_policy, sb3_act_fn
from training.config import ROOT, get_board, load_config, make_env


class HistoryCallback(BaseCallback):
    """Logs every finished training episode and runs periodic greedy evaluation on a separate env."""

    def __init__(self, eval_env, eval_interval: int, curve_episodes: int, eval_seed: int):
        super().__init__()
        self.eval_env, self.eval_interval = eval_env, eval_interval
        self.curve_episodes, self.eval_seed = curve_episodes, eval_seed
        self.history: Dict[str, Any] = {"episode_reward": [], "episode_moves": [], "episode_success": [],
                                        "episode_timestep": [], "eval": []}
        self._next_eval = eval_interval

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.history["episode_reward"].append(float(info["episode"]["r"]))
                self.history["episode_moves"].append(int(info["episode"]["l"]))
                self.history["episode_success"].append(bool(info.get("is_success", False)))
                self.history["episode_timestep"].append(self.num_timesteps)
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_interval
            m = evaluate_policy(self.eval_env, sb3_act_fn(self.model), self.curve_episodes, self.eval_seed)
            self.history["eval"].append({"episodes": len(self.history["episode_reward"]),
                                         "timesteps": self.num_timesteps, **m})
        return True


def train_drl(algo: str, board: Board, cfg: Dict[str, Any], exp: str, seed: int,
              timesteps: int | None = None):
    """Train an SB3 agent. Returns (model, history). Training/eval envs are separate objects."""
    timesteps = timesteps or cfg["drl"]["timesteps"][exp]
    train_env = Monitor(make_env(board, cfg), info_keywords=("is_success",))
    eval_env = make_env(board, cfg)
    model = BUILDERS[algo](train_env, cfg["drl"][algo.lower()], seed, cfg["drl"].get("device", "cpu"))
    ev = cfg["evaluation"]
    cb = HistoryCallback(eval_env, ev["interval_timesteps"], ev["curve_episodes"], ev["seed"])
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    hist = {"algo": algo, "exp": exp, "seed": seed, "total_timesteps": timesteps,
            "train_seconds": time.time() - start, **cb.history}
    # Make sure the final policy is always represented in the learning curve.
    m = evaluate_policy(eval_env, sb3_act_fn(model), ev["curve_episodes"], ev["seed"])
    hist["eval"].append({"episodes": len(hist["episode_reward"]), "timesteps": model.num_timesteps, **m})
    return model, hist


def main() -> None:
    p = argparse.ArgumentParser(description="Train DQN / PPO / A2C")
    p.add_argument("--algo", choices=list(BUILDERS), required=True)
    p.add_argument("--exp", choices=["exp1", "exp2", "exp3"], default="exp1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    board = get_board(cfg, a.exp)
    model, hist = train_drl(a.algo, board, cfg, a.exp, a.seed, a.timesteps)
    model.save(ROOT / "results" / "trained_models" / f"{a.exp}_{a.algo}_seed{a.seed}")
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs" / f"{a.exp}_{a.algo}_seed{a.seed}.json").write_text(json.dumps(hist))
    last = hist["eval"][-1]
    print(f"{a.algo} {a.exp}: final greedy eval success={last['success_rate']:.0f}% moves={last['mean_moves']:.1f}")


if __name__ == "__main__":
    main()
