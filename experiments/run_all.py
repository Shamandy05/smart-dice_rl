"""One command for everything: experiments -> summary tables -> plots -> board figures.

    python -m experiments.run_all --full           # final experiments (config.yaml budgets, all seeds)
    python -m experiments.run_all --smoke          # 5 % budget, 1 seed -> results_smoke/ (NOT final results)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_configs import FULL, SMOKE
from training.config import ROOT, get_board, load_config
from training.experiment_runner import DRL, TABULAR, baselines, run_one
from visualization.board_visualizer import save_board_png
from visualization.training_plots import EXPS, load_results, make_all_plots


def summarize(results_dir: Path) -> pd.DataFrame:
    """Aggregate final-evaluation metrics over seeds (mean ± std) into CSV + Markdown tables."""
    metrics, _, base = load_results(results_dir)
    rows = []
    for e in EXPS:
        if e in base:
            b = base[e]
            rows.append({"board": e, "algorithm": "Random policy", "seeds": 1,
                         "moves": f"{b['random']['mean_moves']:.2f}", "success_%": f"{b['random']['success_rate']:.1f}",
                         "reward": f"{b['random']['mean_reward']:.1f}", "converged_seeds": "-", "episodes_to_converge": "-",
                         "timesteps_to_converge": "-"})
            rows.append({"board": e, "algorithm": "BFS optimum (reference)", "seeds": "-",
                         "moves": str(b["bfs_min_moves"]), "success_%": "100.0",
                         "reward": f"{b['bfs_path_return']:.1f} (path) / {b['dp_max_return']:.1f} (max)",
                         "converged_seeds": "-", "episodes_to_converge": "-", "timesteps_to_converge": "-"})
        for a in TABULAR + DRL:
            ms = [m for m in metrics if m["exp"] == e and m["algo"] == a]
            if not ms:
                continue
            f = lambda k: np.array([m["final"][k] for m in ms])
            conv = [m["convergence"] for m in ms if m["converged"]]
            pm = lambda v, d=2: f"{v.mean():.{d}f} ± {v.std():.{d}f}"
            rows.append({"board": e, "algorithm": a, "seeds": len(ms), "moves": pm(f("mean_moves")),
                         "success_%": pm(f("success_rate"), 1), "reward": pm(f("mean_reward"), 1),
                         "converged_seeds": f"{len(conv)}/{len(ms)}",
                         "episodes_to_converge": (f"{np.mean([c['episodes'] for c in conv]):.0f} ± {np.std([c['episodes'] for c in conv]):.0f}" if conv else "not converged"),
                         "timesteps_to_converge": (f"{np.mean([c['timesteps'] for c in conv]):.0f}" if conv else "not converged")})
    df = pd.DataFrame(rows)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "metrics" / "summary.csv", index=False)
    (results_dir / "metrics" / "summary.md").write_text(df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False))
    return df


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--smoke", action="store_true")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    preset = FULL if a.full else SMOKE
    cfg = load_config(a.config)
    rd = ROOT / ("results" if a.full else "results_smoke")
    seeds = preset["seeds"] or cfg["seeds"]
    for exp in EXPS:
        b = baselines(exp, cfg, rd)
        print(f"{exp}: BFS={b['bfs_min_moves']} moves; random: {b['random']['mean_moves']:.1f} moves", flush=True)
        for algo in TABULAR + DRL:
            for seed in seeds:
                r = run_one(algo, exp, seed, cfg, rd, preset["scale"])
                f = r["final"]
                print(f"  {exp} {algo:10s} seed{seed}: success={f['success_rate']:.0f}% moves={f['mean_moves']:.1f} "
                      f"reward={f['mean_reward']:.1f} converged={r['converged']} ({r['train_seconds']:.0f}s)", flush=True)
    for exp in EXPS:
        save_board_png(get_board(cfg, exp), rd / "plots" / f"board_{exp}.png")
    make_all_plots(rd)
    print(summarize(rd).to_string(index=False))


if __name__ == "__main__":
    main()
