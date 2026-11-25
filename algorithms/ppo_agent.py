"""PPO (Stable-Baselines3): actor-critic with clipped surrogate objective and GAE."""
from __future__ import annotations

from stable_baselines3 import PPO


def build_ppo(env, hp: dict, seed: int, device: str = "cpu") -> PPO:
    hp = dict(hp)
    net_arch = hp.pop("net_arch", [64, 64])
    return PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, seed=seed, device=device,
               verbose=0, **hp)
