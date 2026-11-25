import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from environments.board_generator import Board
from environments.dice_board_env import DiceBoardEnv
from tests.helpers import small_env


def test_initialization_and_spaces():
    env = small_env()
    obs, info = env.reset(seed=0)
    assert obs == 0 and info["position"] == 1          # square 1 -> observation 0
    assert env.observation_space.n == 20 and env.action_space.n == 6
    assert DiceBoardEnv.action_to_dice(0) == 1 and DiceBoardEnv.action_to_dice(5) == 6


def test_gymnasium_and_sb3_check_env():
    check_env(small_env(), warn=True, skip_render_check=True)


def test_invalid_action_value_rejected():
    env = small_env(); env.reset()
    with pytest.raises(ValueError):
        env.step(6)


def test_max_episode_length_truncates():
    env = small_env(max_episode_steps=3); env.reset()
    for i in range(3):
        _, _, term, trunc, _ = env.step(0)             # dice 1: 2,3(ladder)->12,13 ...
    assert not term and trunc


def test_reset_is_reproducible():
    e1, e2 = small_env(), small_env()
    assert e1.reset(seed=3)[0] == e2.reset(seed=3)[0]


def test_render_ansi():
    env = small_env(render_mode="ansi"); env.reset()
    assert "Position 1/20" in env.render()
