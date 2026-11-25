"""Interactive-style demonstration (PDF Section 8). Everything shown is computed live.

    python demo.py --exp exp1                    # board, random play, training, learned play, Q vs SARSA
    python demo.py --exp exp3 --drl              # additionally trains DQN/PPO/A2C (slower)

Outputs (results/demo/): board PNG, before/after GIFs, learning-curve PNG, printed comparison tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.bfs_baseline import bfs_min_moves, bfs_shortest_path
from evaluation.metrics import evaluate_policy, random_policy, sb3_act_fn, tabular_act_fn
from training.config import ROOT, get_board, load_config, make_env
from training.train_drl import train_drl
from training.train_tabular import train_tabular
from visualization.board_visualizer import save_board_png
from visualization.game_animation import play_episode, save_episode_gif


def row(name, m):
    return f"{name:22s} success={m['success_rate']:5.1f}%  moves={m['mean_moves']:6.2f}  reward={m['mean_reward']:8.1f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="exp1", choices=["exp1", "exp2", "exp3"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drl", action="store_true", help="also train and compare DQN / PPO / A2C")
    a = p.parse_args()
    cfg = load_config()
    out = ROOT / "results" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    board = get_board(cfg, a.exp)
    n_eval, eval_seed = cfg["evaluation"]["episodes"], cfg["evaluation"]["seed"]

    print("=== 1. Board configuration ===")
    print(f"N={board.size}  snakes(head->tail)={board.snakes}  ladders(bottom->top)={board.ladders}")
    save_board_png(board, out / f"{a.exp}_board.png")

    print("\n=== 2. BEFORE training: random policy ===")
    rand = random_policy(make_env(board, cfg), n_eval, eval_seed)
    print(row("random policy", rand))
    pos, tot, ok, _ = play_episode(make_env(board, cfg), lambda o, r=np.random.default_rng(1): int(r.integers(6)), seed=1)
    save_episode_gif(board, pos, out / f"{a.exp}_before_training.gif", "BEFORE training (random)")
    print(f"one random game: {len(pos) - 1} moves, reward {tot:.0f} -> {out / (a.exp + '_before_training.gif')}")

    print("\n=== 3. Training (Q-Learning vs SARSA, epsilon-greedy, separate eval env) ===")
    results, hists, agents = {}, {}, {}
    for algo in ["Q-Learning", "SARSA"]:
        ag, h = train_tabular(algo, board, cfg, a.exp, a.seed, verbose=True)
        agents[algo], hists[algo] = ag, h
        results[algo] = evaluate_policy(make_env(board, cfg), tabular_act_fn(ag), n_eval, eval_seed)

    print("\n=== 4. AFTER training: learned greedy policy ===")
    pos, tot, ok, _ = play_episode(make_env(board, cfg), tabular_act_fn(agents["Q-Learning"]), seed=1)
    save_episode_gif(board, pos, out / f"{a.exp}_after_training.gif", "AFTER training (Q-Learning)")
    print(f"one learned game: {len(pos) - 1} moves, reward {tot:.0f}, squares visited {pos}")
    save_board_png(board, out / f"{a.exp}_learned_path.png", position=board.size, trail=pos,
                   title=f"Learned Q-Learning trajectory ({len(pos) - 1} moves)")

    if a.drl:
        for algo in ["DQN", "PPO", "A2C"]:
            print(f"training {algo} ...")
            model, h = train_drl(algo, board, cfg, a.exp, a.seed)
            hists[algo] = h
            results[algo] = evaluate_policy(make_env(board, cfg), sb3_act_fn(model), n_eval, eval_seed)

    print("\n=== 5. Before vs after, and algorithm comparison (same eval seed, same board) ===")
    print(row("random (before)", rand))
    for k, v in results.items():
        print(row(k, v))
    print(f"{'BFS optimum (reference)':22s} moves={bfs_min_moves(board)}  path={bfs_shortest_path(board)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algo in ["Q-Learning", "SARSA"]:
        h = hists[algo]
        r = np.convolve(h["episode_reward"], np.ones(25) / 25, mode="valid")
        m = np.convolve(h["episode_moves"], np.ones(25) / 25, mode="valid")
        axes[0].plot(r, label=algo); axes[1].plot(m, label=algo)
    axes[0].set(title="Reward per training episode (rolling 25)", xlabel="Training episode", ylabel="Total reward")
    axes[1].set(title="Moves per training episode (rolling 25)", xlabel="Training episode", ylabel="Moves")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{a.exp}_learning_curves.png", dpi=200)
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
