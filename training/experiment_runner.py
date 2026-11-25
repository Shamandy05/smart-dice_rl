"""Runs the full experiment grid: boards x algorithms x seeds, with resume support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.bfs_baseline import bfs_min_moves, bfs_shortest_path, optimal_return_dp, path_return
from evaluation.convergence import find_convergence
from training.config import ROOT, get_board, load_config
from training.evaluate import final_evaluation, random_baseline
from training.train_drl import train_drl
from training.train_tabular import train_tabular

TABULAR = ["Q-Learning", "SARSA"]
DRL = ["DQN", "PPO", "A2C"]


def run_one(algo: str, exp: str, seed: int, cfg: Dict[str, Any], results_dir: Path, scale: float = 1.0,
            verbose: bool = False) -> Dict[str, Any]:
    """Train + evaluate one (algorithm, board, seed). ``scale`` shrinks the budget for smoke tests."""
    board = get_board(cfg, exp, save=True)
    res_path = results_dir / "metrics" / f"{exp}_{algo}_seed{seed}.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    conv_cfg = cfg["convergence"]
    if algo in TABULAR:
        episodes = max(10, int(cfg["tabular"]["episodes"][exp] * scale))
        agent, hist = train_tabular(algo, board, cfg, exp, seed, episodes, verbose)
        final = final_evaluation(agent, "tabular", board, cfg)
        agent.save(results_dir / "trained_models" / f"{exp}_{algo}_seed{seed}_qtable.npy")
    else:
        steps = max(500, int(cfg["drl"]["timesteps"][exp] * scale))
        model, hist = train_drl(algo, board, cfg, exp, seed, steps)
        final = final_evaluation(model, "sb3", board, cfg)
        model.save(results_dir / "trained_models" / f"{exp}_{algo}_seed{seed}")
    conv = find_convergence(hist["eval"], conv_cfg["window"], conv_cfg["success_threshold"],
                            conv_cfg["moves_tolerance"])
    out = {"algo": algo, "exp": exp, "seed": seed, "final": final, "convergence": conv,
           "converged": conv is not None, "train_seconds": hist["train_seconds"],
           "training_episodes_total": len(hist["episode_reward"]),
           "smoke": scale != 1.0}
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "logs" / f"{exp}_{algo}_seed{seed}.json").write_text(json.dumps(hist))
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(out, indent=1))
    return out


def baselines(exp: str, cfg: Dict[str, Any], results_dir: Path) -> Dict[str, Any]:
    """BFS (min moves), DP max return, and random policy for one board."""
    board = get_board(cfg, exp)
    path = bfs_shortest_path(board)
    out = {"exp": exp, "bfs_min_moves": len(path), "bfs_path": path,
           "bfs_path_return": path_return(board, path, cfg["env"]["rewards"]),
           "dp_max_return": optimal_return_dp(board, cfg["env"]["max_episode_steps"], cfg["env"]["rewards"]),
           "random": random_baseline(board, cfg)}
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics" / f"{exp}_baselines.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run experiments (resumable)")
    p.add_argument("--exps", nargs="+", default=["exp1", "exp2", "exp3"])
    p.add_argument("--algos", nargs="+", default=TABULAR + DRL)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--scale", type=float, default=1.0, help="budget multiplier (use <1 for smoke tests)")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    rd = Path(a.results_dir)
    seeds: List[int] = a.seeds if a.seeds is not None else cfg["seeds"]
    for exp in a.exps:
        b = baselines(exp, cfg, rd)
        print(f"{exp}: BFS min moves={b['bfs_min_moves']}, random success={b['random']['success_rate']:.0f}%")
        for algo in a.algos:
            for seed in seeds:
                r = run_one(algo, exp, seed, cfg, rd, a.scale)
                f = r["final"]
                print(f"  {exp} {algo:10s} seed{seed}: success={f['success_rate']:.0f}% moves={f['mean_moves']:.1f} "
                      f"reward={f['mean_reward']:.1f} converged={r['converged']} ({r['train_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
