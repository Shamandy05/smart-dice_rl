"""Optional advanced challenges (PDF Section 7). Every number is produced by actually running code.

    python -m challenges.run_challenges --which 1 2 3 4 5 6 [--quick]

Results -> results/challenges/*.json and results/plots/challenge_*.png. All use tabular agents or SB3 with
the configuration in config/config.yaml; ``--quick`` shrinks budgets for a smoke test (written to a
separate directory so it can never be mistaken for final results).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Callable, Dict, List

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor

from algorithms import BUILDERS
from algorithms.q_learning import QLearningAgent
from challenges.multiplayer_env import TwoPlayerRaceEnv
from environments.board_generator import generate_board
from environments.dice_board_env import DiceBoardEnv
from environments.random_dice_env import DynamicBoardEnv, RandomDiceEnv
from evaluation.bfs_baseline import bfs_min_moves, optimal_return_dp
from evaluation.convergence import find_convergence
from evaluation.metrics import evaluate_policy, sb3_act_fn
from training.config import ROOT, get_board, load_config, make_env
from training.evaluate import final_evaluation
from training.train_tabular import train_tabular

OUT: Path
SCALE = 1.0
C5_SIZES = [100, 200, 400]


def _save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float))


def _bar(names: List[str], series: Dict[str, List[float]], title: str, ylabel: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(series)
    for i, (k, v) in enumerate(series.items()):
        ax.bar(np.arange(len(names)) + i * w, v, w, label=k)
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plots = OUT.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / fname, dpi=200)
    plt.close(fig)


def _train_q(env, n_states: int, episodes: int, seed: int, alpha=0.05, gamma=0.99, decay=0.9995) -> QLearningAgent:
    """Generic Q-learning loop for any discrete-observation env (used by challenges 2 and 4)."""
    ag = QLearningAgent(n_states, env.action_space.n, alpha, gamma, 1.0, 0.05, decay, seed=seed)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 7919 + ep)
        done = False
        while not done:
            a = ag.select_action(s)
            s2, r, term, trunc, _ = env.step(a)
            ag.update(s, a, r, s2, term)
            s, done = s2, term or trunc
        ag.decay_epsilon()
    return ag


# --------------------------------------------------------------------------------------------- C1
def challenge1(cfg) -> None:
    """Random dice: no decision exists -> Markov reward process. Report win probability and game length."""
    res = {}
    for exp in ["exp1", "exp2", "exp3"]:
        board = get_board(cfg, exp, save=False)
        env = RandomDiceEnv(board, cfg["env"]["max_episode_steps"], cfg["env"]["rewards"])
        m = evaluate_policy(env, lambda o: 0, int(5000 * SCALE) + 50, 2024)
        # Probability of finishing within k = BFS-optimal number of rolls is not defined without control;
        # report the controlled-dice optimum next to the random-dice expectation.
        res[exp] = {**m, "bfs_min_moves_if_agent_chooses_dice": bfs_min_moves(board)}
        print(f"C1 {exp}: random dice -> win {m['success_rate']:.1f}% within {cfg['env']['max_episode_steps']} turns, "
              f"E[turns]={m['mean_moves']:.1f} (controlled-dice optimum {res[exp]['bfs_min_moves_if_agent_chooses_dice']})")
    _save("challenge1_random_dice", res)


# --------------------------------------------------------------------------------------------- C2
def challenge2(cfg) -> None:
    """One-shot snake shield: state = (shields left, position, rolled dice). Tabular Q-learning."""
    res = {}
    n_eval = int(3000 * SCALE) + 50
    for exp in ["exp2", "exp3"]:
        board = get_board(cfg, exp, save=False)
        mk = lambda special: RandomDiceEnv(board, cfg["env"]["max_episode_steps"], cfg["env"]["rewards"], special_actions=special)
        env = mk(True)
        ag = _train_q(env, env.observation_space.n, int(20000 * SCALE) + 200, seed=0)
        learned = evaluate_policy(mk(True), lambda o: int(ag.q[o].argmax()), n_eval, 99)
        never = evaluate_policy(mk(True), lambda o: 0, n_eval, 99)
        always = evaluate_policy(mk(True), lambda o: 1, n_eval, 99)
        res[exp] = {"learned_shield_policy": learned, "never_use_shield": never, "shield_at_first_snake": always}
        print(f"C2 {exp}: reward learned={learned['mean_reward']:.2f} never={never['mean_reward']:.2f} "
              f"first-snake={always['mean_reward']:.2f}; turns {learned['mean_moves']:.2f}/{never['mean_moves']:.2f}/{always['mean_moves']:.2f}")
    _save("challenge2_special_action", res)
    _bar(list(res), {k: [res[e][k]["mean_reward"] for e in res] for k in ["never_use_shield", "shield_at_first_snake", "learned_shield_policy"]},
         "Challenge 2: mean return with a one-shot snake shield (random dice)", "mean return", "challenge_2_shield.png")


# --------------------------------------------------------------------------------------------- C3
def _eval_dynamic(act: Callable, n: int, size=30, ns=3, nl=3, eval_base=10**6) -> Dict[str, float]:
    env = DynamicBoardEnv(size, ns, nl, base_seed=eval_base)
    moves, succ, rew, bfs = [], [], [], []
    obs, _ = env.reset(seed=1)
    for k in range(n):
        if k:
            obs, _ = env.reset()
        bfs.append(bfs_min_moves(env.board))
        done, total, steps, ok = False, 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(int(act(env, obs)))
            total, steps, ok, done = total + r, steps + 1, ok or term, term or trunc
        moves.append(steps); succ.append(ok); rew.append(total)
    return {"mean_moves": float(np.mean(moves)), "success_rate": float(np.mean(succ) * 100),
            "mean_reward": float(np.mean(rew)), "mean_bfs_moves": float(np.mean(bfs))}


def challenge3(cfg) -> None:
    """Dynamic board: generalisation to *unseen* boards. Agents see a local view of the board."""
    steps = int(60000 * SCALE) + 2000
    n_test = 100
    res = {}
    for label, pool in [("PPO_fixed_board", 1), ("PPO_new_board_each_episode", None)]:
        env = Monitor(DynamicBoardEnv(30, 3, 3, base_seed=0, fixed_pool=pool, max_episode_steps=100))
        model = BUILDERS["PPO"](env, cfg["drl"]["ppo"], seed=0)
        model.learn(steps)
        res[label] = _eval_dynamic(lambda e, o: model.predict(o, deterministic=True)[0], n_test)
    # Tabular Q-learning (state = position only) on one fixed board, evaluated on unseen boards.
    fixed = DynamicBoardEnv(30, 3, 3, base_seed=0, fixed_pool=1, max_episode_steps=100)
    ag = QLearningAgent(30, 6, 0.1, 0.99, 1.0, 0.05, 0.995, seed=0)
    for ep in range(int(1500 * SCALE) + 50):
        o, _ = fixed.reset(seed=ep if ep == 0 else None)
        s, done = fixed.position - 1, False
        while not done:
            a = ag.select_action(s)
            o, r, term, trunc, _ = fixed.step(a)
            s2 = fixed.position - 1
            ag.update(s, a, r, s2, term)
            s, done = s2, term or trunc
        ag.decay_epsilon()
    res["Tabular_Q_fixed_board"] = _eval_dynamic(lambda e, o: ag.q[e.position - 1].argmax(), n_test)
    for k, v in res.items():
        print(f"C3 {k}: success={v['success_rate']:.0f}% moves={v['mean_moves']:.1f} (BFS avg {v['mean_bfs_moves']:.1f}) reward={v['mean_reward']:.1f}")
    _save("challenge3_dynamic_board", res)
    _bar(["mean moves on 100 unseen boards"], {k: [v["mean_moves"]] for k, v in res.items()},
         "Challenge 3: generalisation to unseen boards (N=30; BFS avg = %.1f)" % res["PPO_fixed_board"]["mean_bfs_moves"], "moves", "challenge_3_dynamic.png")


# --------------------------------------------------------------------------------------------- C4
def _win_rate(env, act: Callable[[int], int], n: int, seed: int) -> Dict[str, float]:
    """Win rate from info['is_success'] (terminated is also True when the *opponent* wins)."""
    wins, turns = [], []
    obs, _ = env.reset(seed=seed)
    for k in range(n):
        if k:
            obs, _ = env.reset()
        done, steps, won = False, 0, False
        while not done:
            obs, r, term, trunc, info = env.step(int(act(obs)))
            steps += 1
            done = term or trunc
            won = bool(info["is_success"])
        wins.append(won); turns.append(steps)
    return {"success_rate": float(np.mean(wins) * 100), "mean_moves": float(np.mean(turns))}


def challenge4(cfg) -> None:
    """Two-player race against a rule-based greedy opponent and against a random opponent."""
    res = {}
    for exp in ["exp1", "exp3"]:
        board = get_board(cfg, exp, save=False)
        for opp in ["greedy", "random"]:
            mk = lambda: TwoPlayerRaceEnv(board, opp, cfg["env"]["max_episode_steps"], rewards=cfg["env"]["rewards"])
            env = mk()
            ag = _train_q(env, env.observation_space.n, int(20000 * SCALE) + 200, seed=0, alpha=0.1, decay=0.9995)
            n = int(2000 * SCALE) + 50
            trained = _win_rate(mk(), lambda o: int(ag.q[o].argmax()), n, 5)
            rng = np.random.default_rng(5)
            rand = _win_rate(mk(), lambda o: int(rng.integers(6)), n, 5)
            res[f"{exp}_vs_{opp}"] = {"trained_win_rate": trained["success_rate"], "random_agent_win_rate": rand["success_rate"],
                                      "trained_mean_turns": trained["mean_moves"]}
            print(f"C4 {exp} vs {opp}: trained win {trained['success_rate']:.1f}% | random agent win {rand['success_rate']:.1f}%")
    _save("challenge4_multiplayer", res)
    _bar(list(res), {"trained agent": [v["trained_win_rate"] for v in res.values()],
                     "random-dice agent": [v["random_agent_win_rate"] for v in res.values()]},
         "Challenge 4: win rate in a two-player race (random player starts)", "win rate (%)", "challenge_4_multiplayer.png")


# --------------------------------------------------------------------------------------------- C5
def challenge5(cfg) -> None:
    """Equal-budget DQN vs tabular Q-Learning on increasingly large boards (single seed)."""
    budget = int(150000 * SCALE) + 3000
    res = {}
    results_file = OUT / "challenge5_dqn_vs_qlearning.json"
    if results_file.exists():                       # merge with sizes already run
        res = json.loads(results_file.read_text())
    for size in C5_SIZES:
        k = size // 12 if size < 400 else 20        # N=400: 20/20 (32/32 admits no reward-cycle-free board)
        board = generate_board(size, k, k, seed=size)
        c = copy.deepcopy(cfg)
        c["env"]["max_episode_steps"] = 4 * size
        c["evaluation"]["interval_episodes"] = 25
        c["evaluation"]["curve_episodes"] = 1
        ag, hist = train_tabular("Q-Learning", board, c, "exp3", 0, verbose=False, max_timesteps=budget, epsilon_decay=0.99)
        tab = final_evaluation(ag, "tabular", board, c)
        env = Monitor(make_env(board, c))
        model = BUILDERS["DQN"](env, c["drl"]["dqn"], seed=0)
        model.learn(budget)
        dqn = final_evaluation(model, "sb3", board, c)
        res[f"N={size}"] = {"bfs_min_moves": bfs_min_moves(board), "tabular_q": tab, "dqn": dqn,
                            "tabular_episodes": len(hist["episode_reward"]), "budget_timesteps": budget}
        print(f"C5 N={size} (BFS {bfs_min_moves(board)}): Q-Learning success={tab['success_rate']:.0f}% moves={tab['mean_moves']:.1f} "
              f"| DQN success={dqn['success_rate']:.0f}% moves={dqn['mean_moves']:.1f}  [budget {budget} steps]", flush=True)
    _save("challenge5_dqn_vs_qlearning", res)
    _bar(list(res), {"BFS optimum": [v["bfs_min_moves"] for v in res.values()],
                     "Q-Learning": [v["tabular_q"]["mean_moves"] for v in res.values()],
                     "DQN": [v["dqn"]["mean_moves"] for v in res.values()]},
         "Challenge 5: moves of greedy policy after equal training budget", "average moves", "challenge_5_dqn_vs_q.png")


# --------------------------------------------------------------------------------------------- C6
class PotentialShaping(gym.Wrapper):
    """r' = r + gamma*Phi(s') - Phi(s), Phi(s) = scale * position (Ng et al. 1999); Phi(terminal)=0."""

    def __init__(self, env, scale: float, gamma: float):
        super().__init__(env)
        self.scale, self.gamma = scale, gamma

    def step(self, action):
        old = self.env.position
        obs, r, term, trunc, info = self.env.step(action)
        phi_new = 0.0 if term else self.scale * self.env.position
        return obs, r + self.gamma * phi_new - self.scale * old, term, trunc, info


def challenge6(cfg) -> None:
    """Alternative reward functions, always *evaluated* with the official environment."""
    gamma = cfg["tabular"]["gamma"]
    official = cfg["env"]["rewards"]
    variants = {
        "official": lambda b: make_env(b, cfg),
        "goal_only (sparse)": lambda b: DiceBoardEnv(b, cfg["env"]["max_episode_steps"], {"goal": 100, "ladder": 0, "normal": 0, "snake": 0, "invalid": 0}),
        "heavy_step_penalty (-5)": lambda b: DiceBoardEnv(b, cfg["env"]["max_episode_steps"], {**official, "normal": -5.0}),
        "potential_shaping (0.5*pos)": lambda b: PotentialShaping(make_env(b, cfg), 0.5, gamma),
        "ladder_bonus (+60)": lambda b: DiceBoardEnv(b, cfg["env"]["max_episode_steps"], {**official, "ladder": 60.0}),
    }
    seeds = cfg["seeds"]
    res = {}
    for exp in ["exp3"]:
        board = get_board(cfg, exp, save=False)
        for name, fac in variants.items():
            runs = []
            for sd in seeds:
                ag, hist = train_tabular("Q-Learning", board, cfg, exp, sd, int(cfg["tabular"]["episodes"][exp] * max(SCALE, 0.1)),
                                         train_env_factory=lambda: fac(board))
                fin = final_evaluation(ag, "tabular", board, cfg)
                conv = find_convergence(hist["eval"], cfg["convergence"]["window"], cfg["convergence"]["success_threshold"], cfg["convergence"]["moves_tolerance"])
                runs.append({**fin, "converge_episodes": conv["episodes"] if conv else None})
            ce = [r["converge_episodes"] for r in runs if r["converge_episodes"] is not None]
            res[name] = {"mean_moves": float(np.mean([r["mean_moves"] for r in runs])), "success_rate": float(np.mean([r["success_rate"] for r in runs])),
                         "official_reward": float(np.mean([r["mean_reward"] for r in runs])),
                         "converged_seeds": f"{len(ce)}/{len(runs)}", "mean_converge_episodes": float(np.mean(ce)) if ce else None}
            print(f"C6 {exp} {name:30s} moves={res[name]['mean_moves']:.1f} success={res[name]['success_rate']:.0f}% "
                  f"official_reward={res[name]['official_reward']:.1f} converged={res[name]['converged_seeds']} at {res[name]['mean_converge_episodes']}", flush=True)
    # Unintended behaviour with the *official* rewards: a board with a reward-positive loop.
    bad = generate_board(50, 5, 5, seed=22, reject_reward_cycles=False)
    cfg2 = copy.deepcopy(cfg)
    runs = []
    for sd in seeds:
        ag, hist = train_tabular("Q-Learning", bad, cfg2, "exp2", sd)
        runs.append(final_evaluation(ag, "tabular", bad, cfg2))
    res["official_reward_on_exploitable_board_exp2_seed22"] = {
        "bfs_min_moves": bfs_min_moves(bad), "dp_max_return_in_200_steps": optimal_return_dp(bad, cfg["env"]["max_episode_steps"], official),
        "mean_moves": float(np.mean([r["mean_moves"] for r in runs])), "success_rate": float(np.mean([r["success_rate"] for r in runs])),
        "official_reward": float(np.mean([r["mean_reward"] for r in runs]))}
    print("C6 exploitable board:", res["official_reward_on_exploitable_board_exp2_seed22"])
    _save("challenge6_reward_optimization", res)
    names = [n for n in res if n in variants]
    _bar(names, {"mean moves": [res[n]["mean_moves"] for n in names], "success rate (%)": [res[n]["success_rate"] for n in names]},
         "Challenge 6: Q-Learning trained on different rewards, evaluated with official rewards (Exp3 board)", "value", "challenge_6_rewards.png")


def main() -> None:
    global OUT, SCALE, C5_SIZES
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--c5-sizes", nargs="+", type=int, default=[100, 200, 400])
    p.add_argument("--quick", action="store_true", help="tiny budgets; results go to results_smoke/ (not final)")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    SCALE = 0.05 if a.quick else 1.0
    C5_SIZES = a.c5_sizes
    OUT = ROOT / ("results_smoke" if a.quick else "results") / "challenges"
    fns = {1: challenge1, 2: challenge2, 3: challenge3, 4: challenge4, 5: challenge5, 6: challenge6}
    for k in a.which:
        fns[k](cfg)


if __name__ == "__main__":
    main()
