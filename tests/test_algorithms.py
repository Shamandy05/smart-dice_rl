import numpy as np
import pytest
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from algorithms import QLearningAgent, SarsaAgent, BUILDERS
from evaluation.convergence import find_convergence
from evaluation.metrics import evaluate_policy, random_policy, sb3_act_fn
from tests.helpers import small_env


def test_q_learning_update_uses_max_of_next_state():
    ag = QLearningAgent(5, 6, alpha=0.5, gamma=0.9, seed=0)
    ag.q[3] = [1, 5, 2, 0, 0, 0]
    ag.update(0, 2, -1.0, 3, False)
    assert ag.q[0, 2] == pytest.approx(0.5 * (-1 + 0.9 * 5))


def test_sarsa_update_uses_actual_next_action():
    ag = SarsaAgent(5, 6, alpha=0.5, gamma=0.9, seed=0)
    ag.q[3] = [1, 5, 2, 0, 0, 0]
    ag.update(0, 2, -1.0, 3, False, a_next=2)           # exploratory a' = 2 (Q=2), not the max
    assert ag.q[0, 2] == pytest.approx(0.5 * (-1 + 0.9 * 2))
    with pytest.raises(ValueError):
        ag.update(0, 2, -1.0, 3, False)


@pytest.mark.parametrize("cls", [QLearningAgent, SarsaAgent])
def test_terminal_update_does_not_bootstrap(cls):
    ag = cls(5, 6, alpha=1.0, gamma=0.9, seed=0)
    ag.q[4] = 99
    ag.update(0, 0, 100.0, 4, True)
    assert ag.q[0, 0] == 100.0


def test_epsilon_greedy_and_decay():
    ag = QLearningAgent(3, 6, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5, seed=1)
    assert len({ag.select_action(0) for _ in range(200)}) == 6          # eps=1 -> explores all
    for _ in range(10):
        ag.decay_epsilon()
    assert ag.epsilon == pytest.approx(0.1)                              # floor respected
    ag.epsilon = 0.0
    ag.q[0, 4] = 1.0
    assert all(ag.select_action(0) == 4 for _ in range(20))              # pure exploitation


def test_q_table_save_load(tmp_path):
    ag = QLearningAgent(4, 6, seed=0); ag.q[1, 2] = 3.0
    ag.save(tmp_path / "q.npy")
    ag2 = QLearningAgent(4, 6); ag2.load(tmp_path / "q.npy")
    assert np.array_equal(ag.q, ag2.q)


def test_tabular_learns_small_board():
    from training.config import load_config
    from training.train_tabular import train_tabular
    cfg = load_config()
    cfg["evaluation"]["interval_episodes"] = 50
    board = small_env().board
    for algo in ["Q-Learning", "SARSA"]:
        agent, hist = train_tabular(algo, board, cfg, "exp1", seed=0, episodes=300)
        m = evaluate_policy(small_env(), lambda o: agent.q[o].argmax(), 5, 0)
        assert m["success_rate"] == 100.0 and m["mean_moves"] <= 5


@pytest.mark.parametrize("algo,cls", [("DQN", DQN), ("PPO", PPO), ("A2C", A2C)])
def test_drl_train_predict_save_load(algo, cls, tmp_path):
    from training.config import load_config
    cfg = load_config()
    hp = dict(cfg["drl"][algo.lower()])
    if algo == "DQN":
        hp["learning_starts"] = 50
    if algo == "PPO":
        hp.update(n_steps=64, batch_size=32)
    env = Monitor(small_env(), info_keywords=("is_success",))
    model = BUILDERS[algo](env, hp, seed=0)
    model.learn(300)
    obs, _ = env.reset(seed=0)
    a1 = int(model.predict(obs, deterministic=True)[0])
    model.save(tmp_path / "m")
    loaded = cls.load(tmp_path / "m")
    assert int(loaded.predict(obs, deterministic=True)[0]) == a1 and 0 <= a1 <= 5


def test_random_policy_and_metrics_keys():
    m = random_policy(small_env(), 20, seed=0)
    assert {"mean_moves", "success_rate", "mean_reward"} <= set(m)


def test_convergence_criterion():
    stable = [{"episodes": 50 * i, "timesteps": 100 * i, "success_rate": 100.0, "mean_moves": 4.0} for i in range(1, 10)]
    bad = [{"episodes": i, "timesteps": i, "success_rate": 100.0 if i % 2 else 0.0, "mean_moves": 4.0} for i in range(10)]
    assert find_convergence(stable, window=5)["episodes"] == 50
    assert find_convergence(bad, window=5) is None
