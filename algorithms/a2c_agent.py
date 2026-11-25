"""A2C (Stable-Baselines3): synchronous advantage actor-critic."""
from __future__ import annotations

from stable_baselines3 import A2C


def build_a2c(env, hp: dict, seed: int, device: str = "cpu") -> A2C:
    hp = dict(hp)
    net_arch = hp.pop("net_arch", [64, 64])
    return A2C("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, seed=seed, device=device,
               verbose=0, **hp)
