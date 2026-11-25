"""Reproducible, validated snake/ladder board generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Board:
    """Immutable board description (squares are numbered 1..size)."""
    size: int
    snakes: Dict[int, int] = field(default_factory=dict)   # head -> tail (tail < head)
    ladders: Dict[int, int] = field(default_factory=dict)  # bottom -> top (top > bottom)
    seed: int | None = None

    def transitions(self) -> Dict[int, int]:
        return {**self.snakes, **self.ladders}

    def to_dict(self) -> dict:
        return {"size": self.size, "seed": self.seed,
                "snakes": {str(k): v for k, v in self.snakes.items()},
                "ladders": {str(k): v for k, v in self.ladders.items()}}

    @staticmethod
    def from_dict(d: dict) -> "Board":
        return Board(int(d["size"]), {int(k): int(v) for k, v in d["snakes"].items()},
                     {int(k): int(v) for k, v in d["ladders"].items()}, d.get("seed"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "Board":
        return Board.from_dict(json.loads(Path(path).read_text()))


def validate_board(board: Board) -> None:
    """Raise ValueError if the board violates any rule. Also checks the goal is reachable."""
    n = board.size
    if n < 10:
        raise ValueError("Board size must be >= 10")
    endpoints = []
    for head, tail in board.snakes.items():
        if not (2 <= tail < head <= n - 1):
            raise ValueError(f"Invalid snake {head}->{tail}: must move backward, stay in 2..N-1")
        endpoints += [head, tail]
    for bottom, top in board.ladders.items():
        if not (2 <= bottom < top <= n - 1):
            raise ValueError(f"Invalid ladder {bottom}->{top}: must move forward, stay in 2..N-1")
        endpoints += [bottom, top]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Snake/ladder endpoints overlap (this would create chained transitions)")
    # Local import avoids a circular dependency; BFS proves the goal is reachable.
    from evaluation.bfs_baseline import bfs_shortest_path
    if bfs_shortest_path(board) is None:
        raise ValueError("Goal square is unreachable on this board")


def generate_board(size: int, n_snakes: int, n_ladders: int, seed: int = 0,
                   min_jump: int = 3, reject_reward_cycles: bool = True) -> Board:
    """Randomly place snakes and ladders. Squares 1 and N stay free; all endpoints are distinct.

    ``min_jump`` guarantees every snake/ladder spans at least that many squares.
    With ``reject_reward_cycles`` (default) boards that contain a reachable positive-return cycle
    under the official rewards (reward-farming loops) are re-drawn; see
    ``evaluation.bfs_baseline.has_positive_return_cycle``.
    """
    from evaluation.bfs_baseline import has_positive_return_cycle
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        free = set(range(2, size))  # excludes start (1) and goal (N)
        snakes, ladders, ok = {}, {}, True
        for kind, count in (("snake", n_snakes), ("ladder", n_ladders)):
            for _ in range(count):
                for _try in range(200):
                    a, b = sorted(rng.choice(sorted(free), size=2, replace=False).tolist())
                    if b - a >= min_jump:
                        break
                else:
                    ok = False
                    break
                free -= {a, b}
                if kind == "snake":
                    snakes[b] = a      # head high, tail low
                else:
                    ladders[a] = b     # bottom low, top high
            if not ok:
                break
        if not ok:
            continue
        board = Board(size, snakes, ladders, seed)
        try:
            validate_board(board)
        except ValueError:
            continue
        if reject_reward_cycles and has_positive_return_cycle(board):
            continue
        return board
    raise RuntimeError("Could not generate a valid board")
