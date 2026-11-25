from environments.board_generator import generate_board
from environments.random_dice_env import DynamicBoardEnv, RandomDiceEnv
from stable_baselines3.common.env_checker import check_env


def test_random_dice_env_checks_and_shield():
    b = generate_board(30, 3, 3, 1)
    env = RandomDiceEnv(b, special_actions=True)
    check_env(env, warn=True, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert info["shields"] == 1 and env.observation_space.n == 30 * 6 * 2
    # force a snake hit: shield should cancel it exactly once; second time the snake bites
    head = next(iter(b.snakes))
    env.position, env.next_dice = head - 1, 1
    _, _, _, _, info = env.step(1)
    assert info["event"] == "shield" and info["position"] == head and info["shields"] == 0
    env.position, env.next_dice = head - 1, 1
    _, _, _, _, info = env.step(1)
    assert info["event"] == "snake" and info["position"] == b.snakes[head]


def test_dynamic_board_changes_and_obs_shape():
    env = DynamicBoardEnv(30, 3, 3, base_seed=0)
    check_env(env, warn=True, skip_render_check=True)
    env.reset(seed=1); b1 = env.board
    env.reset(); b2 = env.board
    assert b1 != b2 and env.observation_space.shape == (13,)
