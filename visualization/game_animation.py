"""Record an episode played by any policy and save it as an animated GIF."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image

from environments.board_generator import Board
from visualization.board_visualizer import render_board_array


def play_episode(env, act_fn: Callable[[int], int], seed: int = 0) -> Tuple[List[int], float, bool, List[str]]:
    """Play one episode; returns (positions incl. start, total reward, success, events)."""
    obs, info = env.reset(seed=seed)
    positions, events, total, done, ok = [info["position"]], [], 0.0, False, False
    while not done:
        obs, r, term, trunc, info = env.step(int(act_fn(obs)))
        positions.append(info["position"])
        events.append(info["event"])
        total += r
        ok = ok or term
        done = term or trunc
    return positions, total, ok, events


def save_episode_gif(board: Board, positions: List[int], path: str | Path, title: str,
                     fps: int = 3, max_frames: int = 60) -> None:
    """Save a GIF of the agent's trajectory (trail drawn behind the agent)."""
    positions = positions[:max_frames]
    frames = [Image.fromarray(render_board_array(board, p, positions[:i + 1],
                                                 f"{title} | move {i} | square {p}"))
              for i, p in enumerate(positions)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * 3,
                   duration=int(1000 / fps), loop=0)
