"""Config loading and board/env factories shared by every entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from environments.board_generator import Board, generate_board
from environments.dice_board_env import DiceBoardEnv

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    with open(path or ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def get_board(cfg: Dict[str, Any], exp: str, save: bool = True) -> Board:
    """Deterministically build (and persist) the board for experiment ``exp``."""
    b = cfg["boards"][exp]
    board = generate_board(b["size"], b["snakes"], b["ladders"], seed=b["seed"])
    if save:
        board.save(ROOT / "results" / "boards" / f"{exp}.json")
    return board


def make_env(board: Board, cfg: Dict[str, Any], **kw) -> DiceBoardEnv:
    return DiceBoardEnv(board, max_episode_steps=cfg["env"]["max_episode_steps"],
                        rewards=cfg["env"]["rewards"], **kw)
