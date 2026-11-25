# Smart Dice Board Game — Learning to Win with Reinforcement Learning

Snakes-and-ladders where the **agent chooses the dice value** (action ∈ {1..6}). Five learners are compared on three boards
(N = 30 / 50 / 100): tabular **Q-Learning**, tabular **SARSA** (both implemented here) and **DQN, PPO, A2C** (Stable-Baselines3).
BFS / dynamic programming is used *only* as an offline reference — never as a playing policy.
All numbers below come from files in `results/` produced by the code in this repository.
The academic report is in [`REPORT.md`](REPORT.md).

## 1. Installation and commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # gymnasium, stable-baselines3, torch, numpy, matplotlib, pandas, PyYAML, pytest
python -m pytest -q tests                  # 40 tests
```
Verified with Python 3.12, Gymnasium 1.4.0, Stable-Baselines3 2.9.2, PyTorch 2.15 (CPU).

| Goal | Command |
|---|---|
| Train one tabular agent | `python -m training.train_tabular --algo Q-Learning --exp exp2 --seed 0` (`--algo SARSA` likewise) |
| Train one DRL agent | `python -m training.train_drl --algo DQN --exp exp2 --seed 0` (`PPO`, `A2C`) |
| All experiments + tables + plots | `python -m experiments.run_all --full` (≈25 min on 1 CPU; resumable) |
| Smoke test (5 % budget, 1 seed → `results_smoke/`, **not** final) | `python -m experiments.run_all --smoke` |
| Advanced challenges 1–6 | `python -m challenges.run_challenges --which 1 2 3 4 5 6` (≈40 min; `--quick` for smoke) |
| Demonstration | `python demo.py --exp exp1` (add `--drl` for DQN/PPO/A2C) |

Outputs: `results/metrics/summary.{md,csv}`, per-run `results/metrics/*.json`, training histories `results/logs/`,
models and Q-tables `results/trained_models/`, plots `results/plots/`, boards `results/boards/*.json`, challenge results `results/challenges/`,
demo files `results/demo/` (before/after GIFs, learned path, learning curves).

## 2. Requirements → implementation map
| PDF requirement | Where |
|---|---|
| Environment, state/action/reward/transition | `environments/dice_board_env.py` |
| 3 boards (30/3/3, 50/5/5, 100/8/8), reproducible, saved | `environments/board_generator.py`, `config/config.yaml`, `results/boards/` |
| ≥3 RL algorithms | `algorithms/{dqn,ppo,a2c}_agent.py`, `q_learning.py`, `sarsa.py` |
| ε-greedy | `algorithms/tabular_base.py` (tabular), SB3 DQN `exploration_*` (config) |
| Train / separate evaluation | `training/train_*.py`, `evaluation/metrics.py` (separate env objects and seeds) |
| Metrics (moves, success %, reward, convergence) | `evaluation/metrics.py`, `evaluation/convergence.py`, `experiments/run_all.py` |
| 4 required plots + extras | `visualization/training_plots.py` → `results/plots/01…09`, `board_*.png` |
| BFS comparison | `evaluation/bfs_baseline.py` |
| Challenges 1–6 | `environments/random_dice_env.py`, `challenges/` |
| Demonstration | `demo.py`, `visualization/game_animation.py` |
| Questions of Section 9 | `REPORT.md` §8–10 |

## 3. MDP and environment design
* **State** S = {1..N} (observation index = square − 1, Gymnasium `Discrete(N)`). **Action** `Discrete(6)`, action a ↔ dice a+1.
* **Transition:** target = s + dice. Target > N → *invalid*. Target = N → goal (terminal). Otherwise a ladder/snake at the target is applied immediately (no chains: endpoints are distinct).
* **Rewards (exactly as in the PDF):** goal +100, ladder +20, normal −1, snake −20, invalid −10 (configurable).
* **Invalid-move rule (documented):** reward −10, position unchanged, episode continues. Episodes are **truncated** after `max_episode_steps` (200) — `terminated` is only ever the goal.
* **Why an MDP:** next state and reward depend only on (current square, chosen dice) — Markov property — with a fixed transition/reward model and an episodic goal.
* **State vs action space:** the state describes *where the pawn is*; the action space is *what the agent may decide* (which dice value). They have different sizes (N vs 6).
* **Reward-farming loops (design decision):** with +20 ladder / −20 snake, a cycle such as ladder → ladder → snake has *positive* net return. On a first Exp-2 board dynamic programming gave a maximum return of 1420 within 200 steps versus ≈ 139 for finishing. The official rewards were kept; the generator now rejects boards with a reachable positive-return cycle (Bellman-Ford, `has_positive_return_cycle`) so "maximise return" and "reach the goal" agree. The rejected board is studied in Challenge 6.

Boards (head→tail snakes, bottom→top ladders; seeds in config):
* Exp1 N=30: snakes {29:5, 18:15, 21:2}, ladders {7:12, 4:17, 8:24} — BFS optimum 3 moves
* Exp2 N=50: 5/5 — BFS 4 moves. Exp3 N=100: 8/8 — BFS 5 moves (`results/boards/*.json`, pictures `results/plots/board_exp*.png`).

## 4. Algorithms
* **Q-Learning** (off-policy): `Q(s,a) ← Q + α[r + γ max_a' Q(s',a') − Q]`. Target uses the greedy action, independent of the exploratory action actually taken.
* **SARSA** (on-policy): `Q(s,a) ← Q + α[r + γ Q(s',a') − Q]` with a' drawn from the ε-greedy behaviour policy, so exploration risk is priced into the values. SARSA is **not** in Stable-Baselines3; it is implemented from scratch. Terminal transitions do not bootstrap; truncation does.
* **ε-greedy:** random action w.p. ε else greedy (random tie-breaks); ε decays multiplicatively per episode from 1.0 to a minimum of 0.05. Exploration matters because Q starts at 0 and the agent must discover ladders/snakes before it can exploit them.
* **DQN:** MLP Q-network (64×64, one-hot state), replay buffer (50 000), target network (hard update every 500 steps), ε-greedy decaying linearly 1.0→0.05 over the first 30 % of timesteps (SB3's mechanism, different from the per-episode multiplicative decay of the tabular agents), loss on the TD error `r + γ max Q_target(s',·) − Q(s,a)`. Versus tabular Q-learning: function approximation and sampled mini-batches instead of one table cell updated per step.
* **PPO:** actor-critic, GAE advantages (λ=0.95), clipped surrogate `min(rA, clip(r,1±0.2)A)` for stable updates, 10 epochs per rollout.
* **A2C:** synchronous actor-critic, advantage `A = R − V(s)` (n-step, n=16), policy-gradient + value loss + entropy bonus.
* **Timesteps vs episodes:** one timestep = one dice choice; one episode = one game. Every DRL run logs both (episodes ≈ timesteps / average game length), so all curves are plotted against *training episodes*.

## 5. Experimental protocol
* Training budgets (config): tabular 1500 / 3000 / 6000 episodes; DRL 30 k / 60 k / 120 k timesteps; **3 independent seeds** (0, 1, 2); same boards for every algorithm.
* Evaluation: separate environment object, fixed eval seed (12345), 200 episodes, **greedy policy** (`argmax Q`, `deterministic=True`). Because the core game and greedy policies are deterministic, the spread across evaluation episodes is zero; variation is only across training seeds (shown as ± std over 3 seeds).
* **Convergence criterion:** first evaluation point beginning a window of 5 consecutive evaluation points where greedy success ≥ 95 % and mean moves stay within ±10 % of the window mean. Evaluation points every 50 episodes (tabular) / 2000 timesteps (DRL); reported cost is the episode (and timestep) index at the window start. "Not converged" is reported when never met.
* Hyper-parameters: all in `config/config.yaml` (α=0.1, γ=0.99 for tabular; SB3 values listed there).

## 6. Results (actual runs: 3 seeds × 5 algorithms × 3 boards)
Mean ± std over seeds of the greedy policy. Reward = official reward.

| Board | Policy | Moves | Success % | Reward | Converged seeds | Episodes to converge | Timesteps to converge |
|---|---|---|---|---|---|---|---|
| Exp1 N=30 | Random | 19.88 | 100 | 36.4 | – | – | – |
| | BFS optimum | 3 | – | 119 (path) / 120 (max) | – | – | – |
| | Q-Learning | 3.67 ± 0.47 | 100 | 118.3 ± 0.5 | 3/3 | 50 ± 0 | 573 |
| | SARSA | 3.67 ± 0.47 | 100 | 118.3 ± 0.5 | 3/3 | 117 ± 94 | 1006 |
| | DQN | 3.00 ± 0.00 | 100 | 119.0 ± 0.0 | 2/3 | 3106 ± 1664 | 18000 |
| | PPO | 3.67 ± 0.47 | 100 | 119.7 ± 0.5 | 3/3 | 1206 ± 279 | 8667 |
| | A2C | 4.00 ± 0.00 | 100 | 118.0 ± 0.0 | 3/3 | 710 ± 368 | 6667 |
| Exp2 N=50 | Random | 17.27 | 100 | 51.4 | – | – | – |
| | BFS optimum | 4 | – | 139 (path) / 140 (max) | – | – | – |
| | Q-Learning | 4.00 ± 0 | 100 | 139.0 ± 0 | 3/3 | 67 ± 24 | 1012 |
| | SARSA | 4.00 ± 0 | 100 | 139.0 ± 0 | 3/3 | 100 ± 41 | 1411 |
| | DQN | 4.00 ± 0 | 100 | 139.0 ± 0 | 3/3 | 1620 ± 705 | 14667 |
| | PPO | 4.00 ± 0 | 100 | 139.0 ± 0 | 3/3 | 1030 ± 229 | 10000 |
| | A2C | 4.00 ± 0 | 100 | 139.0 ± 0 | 3/3 | 3256 ± 1012 | 24667 |
| Exp3 N=100 | Random | 39.91 | 100 | 3.4 | – | – | – |
| | BFS optimum | 5 | – | 117 (path) / 158 (max) | – | – | – |
| | Q-Learning | 7.00 ± 0 | 100 | 157.0 ± 0 | 3/3 | 133 ± 62 | 4290 |
| | SARSA | 7.67 ± 0.47 | 100 | 156.3 ± 0.5 | 3/3 | 217 ± 131 | 6578 |
| | DQN | 7.33 ± 0.47 | 100 | 157.3 ± 0.5 | 3/3 | 1234 ± 123 | 30667 |
| | PPO | 7.33 ± 0.47 | 100 | 157.3 ± 0.5 | 3/3 | 938 ± 616 | 16000 |
| | A2C | 7.33 ± 0.47 | 100 | 156.7 ± 0.5 | 3/3 | 1024 ± 873 | 17333 |

Full table: `results/metrics/summary.md`.

### Interpretation (what the data does and does not support)
* **Learning is real.** All five learners cut moves by 4–8× versus the random policy (e.g. Exp3: 39.9 → 7–8 moves) and reach returns within ~1 of the DP optimum (Exp1 120, Exp2 140, Exp3 158).
* **Success rate does not discriminate.** Because invalid moves do not end the game and the limit is 200 steps, even the random policy finishes 100 % of the time on these boards; moves and reward are the informative metrics. Success rate is still reported, as required.
* **Moves vs BFS:** on Exp3 the agents use 7–8 moves versus BFS's 5, yet earn *more* reward than the BFS path (≈157 vs 117). The two objectives differ (minimum rolls vs maximum return); the learners optimise return, so a 7-move route through extra ladders beats the 5-move path. Under the "minimum rolls" objective BFS is better by definition. Challenge 6 confirms this: training with a goal-only reward reached 5 moves.
* **Q-Learning vs SARSA:** near-identical here. Tabular curves (`04_q_learning_vs_sarsa.png`) overlap; SARSA needed more episodes to meet the criterion on all three boards (117 vs 50, 100 vs 67, 217 vs 133) and used 7.67 vs 7.0 moves on Exp3, but with 3 seeds and large std these differences are small and **not statistically tested**. The textbook difference (SARSA more cautious near penalties) is expected to matter more in boards with snakes adjacent to the optimal route, which these boards do not stress.
* **Sample efficiency:** tabular agents converged within 50–217 episodes; DRL agents needed roughly 10–50× more episodes (and 8 k–31 k timesteps), as expected for function approximation on a problem with ≤100 discrete states.
* **Board size:** moves, episodes-to-converge (tabular: 50 → 67 → 133 for Q-Learning) and the random baseline's cost all grow with N; the learned policies' success stays at 100 %.
* DQN Exp1: one seed never met the criterion (its greedy evaluation fluctuated), although its final policy was optimal (3 moves) — criterion strictness, not final performance.

Plots: `01_reward_vs_episodes`, `02_moves_vs_episodes`, `03_success_vs_episodes` (training-episode rolling means, ε-greedy behaviour for tabular), `04_q_learning_vs_sarsa`,
`05_dqn_ppo_a2c_comparison` (greedy eval on the separate env), `06_reward_by_board`, `07_success_by_board`, `08_moves_by_board`, `09_convergence_comparison`, `board_exp{1,2,3}`.

## 7. BFS comparison and the research question
BFS (`evaluation/bfs_baseline.py`) finds the minimum number of dice rolls on a **known deterministic** board: 3 / 4 / 5 moves. A finite-horizon DP gives the maximum return (120 / 140 / 158). The RL agents never see these; BFS is run offline for reference.
*Why RL if BFS exists?* BFS needs the full transition model and a fixed objective. RL learns from interaction when (a) dice are random (Challenge 1: expected game 19 / 17 / 41 turns — a policy, not a path, is needed), (b) boards change (Challenge 3: agents trained on one board succeed on only 57–61 % of unseen boards, one trained across varying boards 100 %), (c) the model is unknown or observations are partial, or (d) the objective is a return rather than path length. BFS is simpler, exact and instant when the board is known and deterministic and the goal is minimum rolls — as here. Using RL on the core task is didactic.

## 8. Advanced challenges (actual runs, single seed unless stated)
| # | Result |
|---|---|
| 1 Random dice | No decision exists (Markov reward process). Win within 200 turns 100 / 100 / 99.9 %; E[turns] 19.1 / 17.4 / 40.6 vs controlled-dice optimum 3 / 4 / 5 (5000 episodes each). |
| 2 One-shot snake shield | State = (shields left, square, rolled dice). Return learned / never-use / shield-first-snake: Exp2 62.15 / 50.94 / 62.15; Exp3 39.82 / −2.23 / 39.08. Shield clearly helps; learned policy barely beats the simple rule. |
| 3 Dynamic board (N=30) | 100 unseen boards. PPO trained on one board: 57 % success, 89 moves; tabular Q (one board): 61 %, 81 moves; PPO trained on a new board each episode (sees local board features): 100 %, 3.2 moves (BFS avg 2.9). |
| 4 Two players | Tabular Q vs rule-based opponent, random player starts. Win rate trained vs random-dice agent: Exp1 vs greedy 100 % vs 2.9 %; vs random 98.8 % vs 42.7 %; Exp3 vs greedy 100 % vs 4.2 %; vs random 89.6 % vs 45.1 %. (An earlier version of this challenge scored opponent wins as successes; fixed.) |
| 5 DQN vs Q-Learning, equal ≈153 k-step budget | Moves (Q-Learning / DQN / BFS): N=100 4.0 / 4.0 / 4; N=200 9.0 / 8.0 / 3; N=400 (20/20 features) 24.0 / 18.0 / 13; all 100 % success. DQN's edge appears on the larger boards (1 seed). |
| 6 Reward design (Q-Learning, Exp3, 3 seeds, evaluated with official rewards) | official: 7.0 moves, 100 %; goal-only: 5.0 moves, 100 %, converged in 50 eps; step penalty −5: 8.0 moves; potential shaping 0.5·pos: 7.7 moves; **ladder bonus +60: 0 % success (loops), never converged**. On the discarded exploitable board with *official* rewards: success only 33 %, mean return 927 (DP max 1420) — reward hacking. |

## 9. Limitations
* The randomly generated boards are easy (BFS 3–5 moves), so differences between algorithms are small; no controlled ablation of snake/ladder count or placement was run (Question 7 is answered only from the evidence above).
* 3 seeds for the main experiments, 1 seed for the challenges; no significance tests. Evaluation of deterministic policies has zero within-run variance.
* Convergence is tied to the stated criterion and evaluation interval.
* Challenge 5 N=400 uses 20/20 snakes/ladders (a 32/32 board without reward loops could not be generated).
* Torch was installed as the CPU-capable wheel in a disk-limited sandbox; behaviour is identical.
