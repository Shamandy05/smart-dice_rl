"""DQN (Stable-Baselines3): neural Q-function + replay buffer + target network + epsilon-greedy."""
from __future__ import annotations

from stable_baselines3 import DQN


def build_dqn(env, hp: dict, seed: int, device: str = "cpu") -> DQN:
    """Create a DQN. Exploration: epsilon decays linearly from ``exploration_initial_eps`` to
    ``exploration_final_eps`` over ``exploration_fraction`` of training (SB3 built-in)."""
    hp = dict(hp)
    net_arch = hp.pop("net_arch", [64, 64])
    return DQN("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, seed=seed, device=device,
               verbose=0, **hp)
