from environments.board_generator import Board
from environments.dice_board_env import DiceBoardEnv


def small_board() -> Board:
    """N=20: ladder 3->12, ladder 6->15; snake 9->2, snake 17->5 (hand-made, all valid)."""
    return Board(20, snakes={9: 2, 17: 5}, ladders={3: 12, 6: 15})


def small_env(**kw) -> DiceBoardEnv:
    return DiceBoardEnv(small_board(), **kw)
