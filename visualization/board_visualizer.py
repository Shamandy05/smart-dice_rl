"""Matplotlib rendering of the board (serpentine layout), snakes, ladders and the agent."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from environments.board_generator import Board

COLS = 10


def square_center(k: int, n: int) -> tuple[float, float]:
    """Centre of square k (1-based) in a boustrophedon layout with square 1 at the bottom-left."""
    row, idx = divmod(k - 1, COLS)
    col = idx if row % 2 == 0 else COLS - 1 - idx
    return col + 0.5, row + 0.5


def draw_board(board: Board, ax, position: Optional[int] = None, trail: Optional[Sequence[int]] = None,
               title: Optional[str] = None) -> None:
    n = board.size
    rows = int(np.ceil(n / COLS))
    for k in range(1, n + 1):
        x, y = square_center(k, n)
        shade = "#f4f1e8" if (k % 2) else "#e6e2d3"
        if k == 1:
            shade = "#cfe8cf"
        if k == n:
            shade = "#ffe28a"
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=shade, ec="#999999", lw=0.6))
        ax.text(x - 0.42, y + 0.38, str(k), fontsize=6 if n > 50 else 7, va="top", ha="left", color="#444444")
    for bottom, top in board.ladders.items():
        (x1, y1), (x2, y2) = square_center(bottom, n), square_center(top, n)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#1b8a3a", lw=2.2, alpha=0.85, zorder=3))
    for head, tail in board.snakes.items():
        (x1, y1), (x2, y2) = square_center(head, n), square_center(tail, n)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#c0392b", lw=2.2, alpha=0.85, ls="--",
                                     connectionstyle="arc3,rad=0.25", zorder=3))
    if trail:
        pts = np.array([square_center(k, n) for k in trail])
        ax.plot(pts[:, 0], pts[:, 1], "o-", color="#2c6fbb", ms=4, lw=1, alpha=0.6, zorder=4)
    if position is not None:
        x, y = square_center(position, n)
        ax.plot([x], [y], "o", ms=16, color="#2c6fbb", mec="white", mew=2, zorder=6)
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"Board N={n}  (green = ladder, red dashed = snake, S=1, goal={n})", fontsize=10)


def save_board_png(board: Board, path: str | Path, position: Optional[int] = None,
                   trail: Optional[Sequence[int]] = None, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.7 * np.ceil(board.size / COLS) + 0.8))
    draw_board(board, ax, position, trail, title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_board_array(board: Board, position: int, trail: Optional[Sequence[int]] = None,
                       title: Optional[str] = None) -> np.ndarray:
    """RGB array of the board with the agent drawn at ``position`` (used by env.render and GIFs)."""
    fig, ax = plt.subplots(figsize=(6, 0.6 * np.ceil(board.size / COLS) + 0.8), dpi=100)
    draw_board(board, ax, position, trail, title)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr
