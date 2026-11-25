# Smart Dice Board Game — Learning to Win Using Reinforcement Learning
*Academic report. All figures are taken from `results/` (see README for commands). Single-CPU runs; 3 seeds for the main experiments.*

## 1. Introduction
An agent plays a snakes-and-ladders board with N squares. In the core game it *chooses* the dice value (1–6), so the game is a controllable MDP. We compare tabular Q-Learning, tabular SARSA, DQN, PPO and A2C on boards of 30, 50 and 100 squares, against random play and the BFS/DP optimum, and study six optional challenges.

## 2. Problem Definition
Start at square 1, reach square N. A ladder moves the pawn forward, a snake backward, moves past N are invalid. Rewards: goal +100, ladder +20, normal −1, snake −20, invalid −10. Boards: (30, 3 snakes, 3 ladders), (50, 5, 5), (100, 8, 8).

## 3. Background
*Q-Learning* (off-policy TD control), *SARSA* (on-policy TD control), *ε-greedy exploration*, *DQN* (neural Q-function, replay, target network), *PPO* (clipped policy-gradient actor-critic with GAE), *A2C* (synchronous advantage actor-critic). BFS finds shortest paths in a known graph; finite-horizon DP finds the maximum return.

## 4. MDP Formulation
S = {1..N}; A = {1..6}; deterministic transition s' = f(s, a) (add dice, apply snake/ladder, invalid moves leave s unchanged); reward as above; episodic, terminal at N, truncated after 200 steps; γ = 0.99 for learning, undiscounted returns reported.
**Why an MDP:** (s, a) fully determines the distribution of (r, s') — the Markov property — and the model is stationary. **State vs action space:** the state is *where the pawn is* (N values); the action is *what the agent decides* (6 values). **Random-dice revision (Challenge 1):** the dice becomes part of the transition kernel P(s'|s); with no other decision the process is a Markov reward process, so we report win probability and expected game length; with the shield (Challenge 2) the state becomes (shields left, square, rolled dice) and the action is {move, use shield}.

## 5. Environment Implementation
`DiceBoardEnv` (Gymnasium; passes SB3 `check_env`): observation `Discrete(N)` (square−1), action `Discrete(6)`, invalid move = −10 with unchanged position and no termination, truncation after `max_episode_steps`. `board_generator.py` produces reproducible boards from seeds, with distinct endpoints, forward ladders/backward snakes, free start/goal squares and a BFS reachability check; boards are saved as JSON. **Reward-loop finding:** with the official rewards some boards contain positive-return cycles (e.g. ladder→ladder→snake); on one rejected Exp2 board DP gave 1420 vs ≈139 for finishing. The generator rejects such boards (Bellman-Ford) so that the official reward and the goal agree. 40 unit tests cover transitions, rewards, termination, board generation, TD updates, ε-greedy, model save/load and the challenge environments; all pass.

## 6. Algorithms
Tabular Q-Learning/SARSA: α=0.1, γ=0.99, ε 1.0→0.05 (multiplicative per-episode decay 0.995/0.998/0.999 for the three boards), episodes 1500/3000/6000. DRL (Stable-Baselines3, MLP 64×64, one-hot state): DQN lr 1e-3, buffer 50k, target update 500, ε linear 1→0.05 over 30 % of training; PPO lr 5e-4, n_steps 256, clip 0.2, GAE 0.95; A2C lr 7e-4, n_steps 16. Timesteps 30k/60k/120k. SARSA is implemented by hand (not in SB3). Full values: `config/config.yaml`.
*Q-Learning vs SARSA:* Q-Learning bootstraps from max Q(s',·) (the greedy policy, whatever was executed — off-policy); SARSA from Q(s',a') with a' actually sampled (on-policy). SARSA therefore learns the value of the ε-greedy policy including its exploratory slips; Q-Learning learns the greedy policy's value. *Q-Learning vs DQN:* DQN replaces the table by a network trained on replayed mini-batches with a frozen target.

## 7. Experimental Setup
Training and evaluation environments are separate objects with different seeds; same boards for all algorithms. Final evaluation: greedy policy, 200 episodes, seed 12345. Because the core environment and greedy policies are deterministic, within-run evaluation variance is zero; we report mean ± std over 3 training seeds. **Convergence criterion:** first evaluation point opening 5 consecutive evaluation points with success ≥ 95 % and mean moves within ±10 % of the window mean. Random-policy and BFS/DP references are computed on the same boards.

## 8. Results and Analysis
| Board | Random moves | BFS moves | Q-Learning | SARSA | DQN | PPO | A2C |
|---|---|---|---|---|---|---|---|
| Exp1 (30) | 19.88 | 3 | 3.67±0.47 | 3.67±0.47 | 3.00±0 | 3.67±0.47 | 4.00±0 |
| Exp2 (50) | 17.27 | 4 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| Exp3 (100) | 39.91 | 5 | 7.00 | 7.67±0.47 | 7.33±0.47 | 7.33±0.47 | 7.33±0.47 |

Average evaluation reward (±std): Exp1 118.3 / 118.3 / 119.0 / 119.7 / 118.0 (DP max 120); Exp2 139 for all (DP 140); Exp3 157.0 / 156.3 / 157.3 / 157.3 / 156.7 (DP 158; random 3.4). Success rate is 100 % for every learner and also for the random policy (the 200-step limit is generous), so it does not discriminate. Episodes (timesteps) to meet the convergence criterion:

| Board | Q-Learning | SARSA | DQN | PPO | A2C |
|---|---|---|---|---|---|
| Exp1 | 50 (573) | 117 (1006) | 3106 (18000), 2/3 seeds | 1206 (8667) | 710 (6667) |
| Exp2 | 67 (1012) | 100 (1411) | 1620 (14667) | 1030 (10000) | 3256 (24667) |
| Exp3 | 133 (4290) | 217 (6578) | 1234 (30667) | 938 (16000) | 1024 (17333) |

Plots: `results/plots/01…09`, `board_exp*.png`; demo outputs in `results/demo/`.

## 9. Algorithm Comparison
*Tabular vs DRL:* all end at essentially the same policy quality (within ≈ one reward unit of the DP optimum); tabular methods need 10–50× fewer episodes on these small state spaces. *Q-Learning vs SARSA:* nearly identical curves; SARSA converged later in all three boards and used 0.67 more moves on Exp3, but with 3 seeds and wide std this is suggestive, not significant. *DQN/PPO/A2C:* no consistent winner — PPO converged fastest on Exp2/Exp3 in episodes by a small margin, A2C slowest on Exp2; seed spread is large. We make no claim of superiority.

## 10. BFS Comparison
BFS (min rolls on the known deterministic board) = 3 / 4 / 5 moves. Learned policies match it on Exp2 and are within 0–1 (Exp1) or 2–3 (Exp3) moves. On Exp3 the learned 7-move routes return ≈157 whereas the 5-move BFS path returns 117 and the DP optimum is 158: the agents optimise **return**, BFS optimises **moves**, so the fair comparison depends on the stated objective (min rolls vs max return). A goal-only reward (Challenge 6) did recover 5 moves. BFS was never used as the agents' policy.
**Research question — why RL when BFS exists?** BFS is exact and instant when the full deterministic model is known and the goal is minimum rolls. RL is useful when the model is unknown or hidden, transitions are stochastic (a policy is needed, not a path), the board changes between episodes (generalisation), observations are partial, or the objective is a return that mixes several terms. Evidence from our runs: random dice make a game last 19–41 turns on average instead of 3–5 (Challenge 1); fixed-board agents succeeded on only 57–61 % of unseen boards, whereas an agent trained across boards succeeded 100 % (Challenge 3).

## 11. Advanced Challenges (single seed; see README §8 for numbers)
1. **Random dice:** no decision; win ≥ 99.9 % within 200 turns; E[turns] 19.1/17.4/40.6.
2. **Shield:** learned policy return 62.15 (Exp2) and 39.82 (Exp3) vs 50.94 and −2.23 without shield; simple "shield first snake" rule 62.15 and 39.08.
3. **Dynamic board:** generalisation requires state information about the board (local look-ahead features); see numbers above.
4. **Two players:** trained agent wins 89.6–100 % against rule-based/random opponents versus 2.9–45.1 % for a random-dice agent.
5. **DQN vs Q-Learning, equal ≈153k-step budget:** 

| Board | BFS moves | Q-Learning (moves / success) | DQN (moves / success) | Q-Learning episodes trained |
|---|---|---|---|---|
| N=100 | 4 | 4.0 / 100% | 4.0 / 100% | 6000 |
| N=200 | 3 | 9.0 / 100% | 8.0 / 100% | 6000 |
| N=400 | 13 | 24.0 / 100% | 18.0 / 100% | 5820 |
   DQN is better than tabular Q-Learning on the larger boards at equal budget (N=200: 8 vs 9 moves; N=400: 18 vs 24, BFS 13), while on N=100 both match BFS. Single seed, so the gap is indicative only.
6. **Reward design:** potential-based shaping neither helped nor hurt materially (7.7 vs 7.0 moves); a heavy step penalty gave 8.0 moves; goal-only reward gave 5.0 moves and fastest convergence (50 episodes); a +60 ladder bonus produced **unintended looping** (0 % success). On the exploitable board with official rewards, Q-Learning reached only 33 % success while collecting a mean return of 927.

## 12. Limitations
Boards are easy (BFS 3–5 moves) so algorithms barely separate; success rate is non-discriminating; 3 seeds (main) / 1 seed (challenges); no significance tests; no controlled study of snake/ladder count/placement; deterministic evaluation; N=400 board uses 20/20 snakes/ladders; convergence conclusions depend on the criterion.

## 13. Conclusion
All five algorithms genuinely learn to play, cutting game length by 4–8× versus random play and reaching returns within about one unit of the DP optimum, with tabular methods far more sample-efficient than DRL on these boards. The official reward function admits reward-farming loops on some boards, which we detect and avoid, and which Challenge 6 shows can derail learning. BFS remains the right tool for the known deterministic board; RL's value appears with stochasticity, changing boards and unknown models.

### Answers to the ten questions
1. **MDP?** Next square and reward depend only on (current square, chosen dice); stationary model; episodic (§4).
2. **State vs action space?** State = pawn position (N values); action = dice choice (6 values).
3. **Reward effect?** It defines what "good" means: official rewards made agents prefer high-return routes (7-move Exp3 route, return 157) over the 5-move BFS path (117); a goal-only reward gave 5 moves; a +60 ladder bonus caused looping and 0 % success.
4. **ε-greedy role?** Without exploration Q stays at its initial value for unvisited actions; ε-greedy samples ladders/snakes before exploiting them, and decaying ε lets the policy settle. Tabular ε: 1.0 → 0.05 multiplicative per episode; DQN: linear over 30 % of timesteps.
5. **Q-Learning vs SARSA?** Off-policy max-target vs on-policy sampled-next-action target (§6).
6. **Why different policies?** SARSA's values include the cost of exploratory slips, so it prefers routes that are safe under ε-greedy; Q-Learning evaluates the greedy policy. Here the policies were almost identical (SARSA +0.67 moves on Exp3, not tested for significance) because the optimal routes are not next to snakes.
7. **Number/location of snakes and ladders?** Only indirect evidence: the ladders 4→17 and 8→24 on Exp1 make a 3-move path; BFS is 3/4/5 on boards with 6/10/16 features; positive-return placements (ladder chains plus a snake) create loops. A controlled ablation was **not run** — pending.
8. **Fewer episodes?** Yes: tabular agents met the criterion after 50–217 episodes; DRL after ≈700–3300 episodes (8 k–31 k timesteps).
9. **Larger boards?** More moves (random 19.9 → 39.9; optimal 3 → 5 BFS), more episodes to converge (Q-Learning 50 → 67 → 133), and in Challenge 5 the gap to BFS widens (N=400: see table).
10. **vs BFS?** Equal or within 0–3 moves in moves, but ≥ BFS-path return; the objectives differ (§10).

## 14. References
Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.; Watkins & Dayan (1992) Q-learning; Rummery & Niranjan (1994) SARSA; Mnih et al. (2015) DQN; Schulman et al. (2017) PPO; Mnih et al. (2016) A3C/A2C; Ng, Harada & Russell (1999) potential-based shaping; Raffin et al. (2021) Stable-Baselines3; Towers et al. (2024) Gymnasium.
