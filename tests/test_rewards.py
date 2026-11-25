import pytest

from tests.helpers import small_env


@pytest.mark.parametrize("pos,action,expected", [
    (1, 0, -1.0),      # normal
    (1, 1, 20.0),      # ladder (3)
    (4, 4, -20.0),     # snake (9)
    (19, 1, -10.0),    # invalid
    (14, 5, 100.0),    # goal
])
def test_reward_table_matches_pdf(pos, action, expected):
    env = small_env(); env.reset(); env.position = pos
    assert env.step(action)[1] == expected


def test_custom_reward_override():
    env = small_env(rewards={"ladder": 5.0}); env.reset()
    assert env.step(1)[1] == 5.0
