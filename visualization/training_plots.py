"""All required result plots, generated exclusively from files written by real experiment runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGOS = ["Q-Learning", "SARSA", "DQN", "PPO", "A2C"]
COLORS = {"Q-Learning": "#1f77b4", "SARSA": "#ff7f0e", "DQN": "#2ca02c", "PPO": "#d62728",
          "A2C": "#9467bd", "Random": "#7f7f7f", "BFS": "#000000"}
EXPS = ["exp1", "exp2", "exp3"]
EXP_LABEL = {"exp1": "Exp1 (N=30)", "exp2": "Exp2 (N=50)", "exp3": "Exp3 (N=100)"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def load_results(results_dir: Path):
    metrics = [json.loads(p.read_text()) for p in sorted((results_dir / "metrics").glob("*_seed*.json"))]
    logs: Dict[tuple, dict] = {}
    for p in (results_dir / "logs").glob("*_seed*.json"):
        h = json.loads(p.read_text())
        logs[(h["exp"], h["algo"], h["seed"])] = h
    baselines = {p.stem.replace("_baselines", ""): json.loads(p.read_text())
                 for p in (results_dir / "metrics").glob("*_baselines.json")}
    return metrics, logs, baselines


def _rolling(y: np.ndarray, w: int) -> np.ndarray:
    w = max(1, min(w, len(y)))
    return np.convolve(y, np.ones(w) / w, mode="valid")


def _mean_curve(series: List[np.ndarray], grid_len: int = 200):
    """Average several (possibly different-length) curves on a common normalised x grid (in episodes)."""
    L = min(len(s) for s in series)
    xs = np.linspace(0, 1, grid_len)
    ys = [np.interp(xs, np.linspace(0, 1, len(s)), s) for s in series]
    return xs * L, np.mean(ys, axis=0), np.std(ys, axis=0)


def _train_curves(logs, exp, algo, key, window):
    seeds = [h for (e, a, _), h in logs.items() if e == exp and a == algo]
    ser = []
    for h in seeds:
        y = np.array(h[key], dtype=float)
        if len(y) >= 2:
            ser.append(_rolling(y, window))
    return ser


def _panel_grid(logs, algos, key, ylabel, title, path, window=50, pct=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, exp in zip(axes, EXPS):
        for algo in algos:
            ser = _train_curves(logs, exp, algo, key, window)
            if not ser:
                continue
            x, m, s = _mean_curve(ser)
            m, s = (m * 100, s * 100) if pct else (m, s)
            ax.plot(x, m, color=COLORS[algo], label=algo, lw=1.8)
            ax.fill_between(x, m - s, m + s, color=COLORS[algo], alpha=0.15)
        ax.set_title(EXP_LABEL[exp])
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(title + f" (rolling mean, window={window} episodes; mean ± std over seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_curves(logs, out: Path, algos=ALGOS, prefix: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    _panel_grid(logs, algos, "episode_reward", "Total reward per episode",
                "Total Reward vs Training Episodes", out / f"{prefix}01_reward_vs_episodes.png")
    _panel_grid(logs, algos, "episode_moves", "Moves per episode",
                "Average Number of Moves vs Training Episodes", out / f"{prefix}02_moves_vs_episodes.png")
    _panel_grid(logs, algos, "episode_success", "Success rate (%)",
                "Success Rate vs Training Episodes", out / f"{prefix}03_success_vs_episodes.png", pct=True)


def plot_q_vs_sarsa(logs, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys = [("episode_reward", "Total reward per episode"), ("episode_moves", "Moves per episode"),
            ("episode_success", "Success rate (%)")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for r, (key, ylabel) in enumerate(keys):
        for c, exp in enumerate(EXPS):
            ax = axes[r, c]
            for algo in ["Q-Learning", "SARSA"]:
                ser = _train_curves(logs, exp, algo, key, 50)
                if not ser:
                    continue
                x, m, s = _mean_curve(ser)
                if key == "episode_success":
                    m, s = m * 100, s * 100
                ax.plot(x, m, color=COLORS[algo], label=algo, lw=1.8)
                ax.fill_between(x, m - s, m + s, color=COLORS[algo], alpha=0.15)
            ax.set_title(EXP_LABEL[exp] if r == 0 else "")
            ax.set_xlabel("Training episode")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
    fig.suptitle("Q-Learning vs SARSA (training episodes, epsilon-greedy behaviour; rolling 50, mean ± std)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "04_q_learning_vs_sarsa.png", dpi=200)
    plt.close(fig)


def plot_drl_comparison(logs, out: Path) -> None:
    """DQN vs PPO vs A2C using *greedy evaluation* points on the separate eval env."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for c, exp in enumerate(EXPS):
        for algo in ["DQN", "PPO", "A2C"]:
            hs = [h for (e, a, _), h in logs.items() if e == exp and a == algo]
            if not hs:
                continue
            for r, (key, yl) in enumerate([("success_rate", "Greedy eval success (%)"),
                                           ("mean_moves", "Greedy eval moves")]):
                grid = np.linspace(0, max(h["eval"][-1]["timesteps"] for h in hs), 60)
                ys = [np.interp(grid, [p["timesteps"] for p in h["eval"]], [p[key] for p in h["eval"]]) for h in hs]
                m, s = np.mean(ys, 0), np.std(ys, 0)
                ax = axes[r, c]
                ax.plot(grid, m, color=COLORS[algo], label=algo, lw=1.8)
                ax.fill_between(grid, m - s, m + s, color=COLORS[algo], alpha=0.15)
                ax.set_xlabel("Training timesteps")
                ax.set_ylabel(yl)
                ax.set_title(EXP_LABEL[exp] if r == 0 else "")
                ax.legend(fontsize=8)
    fig.suptitle("DQN vs PPO vs A2C (greedy evaluation on separate env; mean ± std over seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "05_dqn_ppo_a2c_comparison.png", dpi=200)
    plt.close(fig)


def _final_stat(metrics, exp, algo, key):
    v = [m["final"][key] for m in metrics if m["exp"] == exp and m["algo"] == algo]
    return (np.mean(v), np.std(v)) if v else (np.nan, 0.0)


def plot_bars_by_board(metrics, baselines, out: Path) -> None:
    specs = [("mean_reward", "Average evaluation reward", "06_reward_by_board.png", "Average Evaluation Reward by Board Size"),
             ("success_rate", "Success rate (%)", "07_success_by_board.png", "Success Rate by Board Size"),
             ("mean_moves", "Average moves per episode", "08_moves_by_board.png", "Average Moves by Board Size")]
    for key, yl, fname, title in specs:
        fig, ax = plt.subplots(figsize=(11, 5))
        names = ALGOS + ["Random"]
        w = 0.8 / len(names)
        for i, a in enumerate(names):
            vals, errs = [], []
            for e in EXPS:
                if a == "Random":
                    b = baselines.get(e)
                    vals.append(b["random"][key] if b else np.nan); errs.append(0)
                else:
                    m, s = _final_stat(metrics, e, a, key)
                    vals.append(m); errs.append(s)
            ax.bar(np.arange(3) + i * w, vals, w, yerr=errs, capsize=3, color=COLORS[a], label=a)
        if key == "mean_moves":
            for j, e in enumerate(EXPS):
                if e in baselines:
                    ax.hlines(baselines[e]["bfs_min_moves"], j - 0.05, j + 0.85, color="k", ls="--",
                              label="BFS optimum" if j == 0 else None)
        ax.set_xticks(np.arange(3) + 0.4 - w / 2)
        ax.set_xticklabels([EXP_LABEL[e] for e in EXPS])
        ax.set_ylabel(yl)
        ax.set_title(title + " (greedy policy, mean ± std over seeds)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)


def plot_convergence(metrics, out: Path, logs=None) -> None:
    """Episodes to converge (criterion in evaluation/convergence.py). Unconverged runs are marked."""
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.8 / len(ALGOS)
    for i, a in enumerate(ALGOS):
        for j, e in enumerate(EXPS):
            ms = [m for m in metrics if m["exp"] == e and m["algo"] == a]
            conv = [m["convergence"]["episodes"] for m in ms if m["converged"]]
            x = j + i * w
            if conv:
                ax.bar(x, np.mean(conv), w, yerr=np.std(conv), capsize=3, color=COLORS[a],
                       label=a if j == 0 else None)
            else:
                ax.bar(x, 0, w, color=COLORS[a], label=a if j == 0 else None)
            ax.text(x, (np.mean(conv) if conv else 0) * 1.02 + 1, f"{len(conv)}/{len(ms)}",
                    ha="center", fontsize=7)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels([EXP_LABEL[e] for e in EXPS])
    ax.set_yscale("symlog")
    ax.set_ylabel("Training episodes to converge (symlog)")
    ax.set_title("Training Convergence Comparison (labels: converged seeds / total seeds; 0 = never converged)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "09_convergence_comparison.png", dpi=200)
    plt.close(fig)


def make_all_plots(results_dir: Path) -> None:
    metrics, logs, baselines = load_results(results_dir)
    out = results_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_training_curves(logs, out)
    plot_q_vs_sarsa(logs, out)
    plot_drl_comparison(logs, out)
    plot_bars_by_board(metrics, baselines, out)
    plot_convergence(metrics, out)
