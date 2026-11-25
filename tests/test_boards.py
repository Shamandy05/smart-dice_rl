import pytest

from environments.board_generator import Board, generate_board, validate_board
from evaluation.bfs_baseline import bfs_min_moves, bfs_shortest_path, has_positive_return_cycle, optimal_return_dp
from training.config import get_board, load_config


@pytest.mark.parametrize("n,s,l", [(30, 3, 3), (50, 5, 5), (100, 8, 8)])
def test_required_configurations(n, s, l):
    b = generate_board(n, s, l, seed=1)
    assert b.size == n and len(b.snakes) == s and len(b.ladders) == l
    pts = [*b.snakes, *b.snakes.values(), *b.ladders, *b.ladders.values()]
    assert len(set(pts)) == len(pts) and 1 not in pts and n not in pts
    assert not has_positive_return_cycle(b)


def test_seed_reproducibility_and_save_load(tmp_path):
    a, b = generate_board(50, 5, 5, 7), generate_board(50, 5, 5, 7)
    assert a == b and a != generate_board(50, 5, 5, 8)
    a.save(tmp_path / "b.json")
    assert Board.load(tmp_path / "b.json") == a


def test_validation_rejects_bad_boards():
    with pytest.raises(ValueError):
        validate_board(Board(20, snakes={5: 9}))                 # snake going forward
    with pytest.raises(ValueError):
        validate_board(Board(20, ladders={5: 12}, snakes={12: 3}))   # overlap/chain
    with pytest.raises(ValueError):
        validate_board(Board(20, ladders={5: 20}))               # ladder to goal square


def test_bfs_known_board():
    b = Board(20, snakes={9: 2, 17: 5}, ladders={3: 12, 6: 15})
    path = bfs_shortest_path(b)
    assert bfs_min_moves(b) == 2 and path == [(1, 5), (15, 5)]   # 1 -dice5-> 6 (ladder to 15) -dice5-> 20


def test_positive_cycle_detected_on_exploitable_board():
    # Cycle 9 -> 10(ladder to 30, +20) -> 31(ladder to 50, +20) -> 55(snake to 9, -20): net +20 per lap
    b = Board(60, ladders={10: 30, 31: 50}, snakes={55: 9})
    assert has_positive_return_cycle(b)
    assert optimal_return_dp(b, 200) > 500          # DP exploits the loop instead of finishing


def test_dp_return_at_least_bfs_path_return():
    cfg = load_config()
    b = get_board(cfg, "exp1", save=False)
    assert optimal_return_dp(b) >= 100
