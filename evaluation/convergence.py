"""Convergence criterion based on stable greedy-evaluation performance.

Definition: training has converged at the first evaluation point ``i`` such that, for the
``window`` consecutive evaluation points ``i .. i+window-1``:
  (a) success rate >= ``success_threshold`` at every point, and
  (b) mean moves vary by at most ``moves_tolerance`` (relative) around the window mean.
The reported cost is the number of training episodes / timesteps at point ``i``.
Returns None if the criterion is never met (we then report "not converged", never a guess).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def find_convergence(history: List[Dict[str, float]], window: int = 5, success_threshold: float = 0.95,
                     moves_tolerance: float = 0.10) -> Optional[Dict[str, float]]:
    """``history``: list of eval points with keys episodes, timesteps, success_rate (0..100), mean_moves."""
    for i in range(len(history) - window + 1):
        w = history[i:i + window]
        if any(p["success_rate"] / 100.0 < success_threshold for p in w):
            continue
        mean_moves = sum(p["mean_moves"] for p in w) / window
        if all(abs(p["mean_moves"] - mean_moves) <= moves_tolerance * mean_moves for p in w):
            return {"episodes": w[0]["episodes"], "timesteps": w[0].get("timesteps", float("nan")), "index": i}
    return None
