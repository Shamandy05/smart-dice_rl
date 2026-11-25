from tests.helpers import small_env


def go(env, pos):
    env.reset(); env.position = pos


def test_normal_move():
    env = small_env(); go(env, 1)
    obs, r, term, trunc, info = env.step(0)            # dice 1 -> square 2
    assert info["position"] == 2 and obs == 1 and info["event"] == "normal" and not term


def test_ladder_transition_applied_immediately():
    env = small_env(); go(env, 1)
    _, _, _, _, info = env.step(1)                     # dice 2 -> square 3 -> ladder to 12
    assert info["position"] == 12 and info["event"] == "ladder"


def test_snake_transition():
    env = small_env(); go(env, 4)
    _, _, _, _, info = env.step(4)                     # 4+5=9 snake -> 2
    assert info["position"] == 2 and info["event"] == "snake"


def test_invalid_move_beyond_board_keeps_position():
    env = small_env(); go(env, 19)
    _, r, term, trunc, info = env.step(1)              # 19+2=21 > 20
    assert info["position"] == 19 and info["event"] == "invalid" and not term and not trunc


def test_goal_reached_terminates_exact_landing():
    env = small_env(); go(env, 14)
    _, _, term, trunc, info = env.step(5)              # 14+6=20
    assert term and not trunc and info["position"] == 20 and info["is_success"]


def test_ladder_and_snake_geometry_on_generated_boards():
    from environments.board_generator import generate_board
    for seed in range(20):
        b = generate_board(50, 5, 5, seed)
        assert all(t < h for h, t in b.snakes.items()) and all(t > s for s, t in b.ladders.items())
