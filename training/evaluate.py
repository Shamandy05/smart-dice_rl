"""Final evaluation helpers: runs on a fresh environment seeded independently of training."""
from __future__ import annotations

from typing import Any, Dict

from environments.board_generator import Board
from evaluation.metrics import evaluate_policy, random_policy, sb3_act_fn, tabular_act_fn
from training.config import make_env


def final_evaluation(agent_or_model, kind: str, board: Board, cfg: Dict[str, Any]) -> Dict[str, float]:
    """kind: 'tabular' or 'sb3'. Uses cfg['evaluation'] episodes and seed (same for all algorithms)."""
    env = make_env(board, cfg)
    act = tabular_act_fn(agent_or_model) if kind == "tabular" else sb3_act_fn(agent_or_model)
    return evaluate_policy(env, act, cfg["evaluation"]["episodes"], cfg["evaluation"]["seed"])


def random_baseline(board: Board, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Random-policy performance with the same evaluation protocol (before-training reference)."""
    return random_policy(make_env(board, cfg), cfg["evaluation"]["episodes"], cfg["evaluation"]["seed"])
