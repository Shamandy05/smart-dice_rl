"""Classical planning baselines. Used ONLY for comparison, never as an agent's policy.

* ``bfs_shortest_path``   - minimum number of moves (objective: min rolls), needs the full model.
* ``optimal_return_dp``   - maximum achievable return within the episode step limit (objective:
                            max return), via finite-horizon dynamic programming.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from environments.board_generator import Board

REWARDS = {"goal": 100.0, "ladder": 20.0, "normal": -1.0, "snake": -20.0, "invalid": -10.0}


def _next(board: Board, pos: int, dice: int) -> Optional[int]:
    """Deterministic successor square, or None if the move is invalid (beyond the board)."""
    t = pos + dice
    if t > board.size:
        return None
    return board.transitions().get(t, t) if t != board.size else t


def bfs_shortest_path(board: Board) -> Optional[List[Tuple[int, int]]]:
    """Return the list of (square, dice) moves of a shortest path 1 -> N, or None if unreachable."""
    start, goal = 1, board.size
    parent: Dict[int, Tuple[int, int]] = {}
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dice in range(1, 7):
            nxt = _next(board, cur, dice)
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (cur, dice)
            q.append(nxt)
    if goal not in seen:
        return None
    path, node = [], goal
    while node != start:
        prev, dice = parent[node]
        path.append((prev, dice))
        node = prev
    return path[::-1]


def bfs_min_moves(board: Board) -> int:
    path = bfs_shortest_path(board)
    if path is None:
        raise ValueError("Unreachable goal")
    return len(path)


def optimal_return_dp(board: Board, max_steps: int = 200, rewards: Optional[dict] = None) -> float:
    """Maximum undiscounted return from square 1 within ``max_steps`` steps (finite-horizon DP)."""
    r = rewards or REWARDS
    n, jumps = board.size, board.transitions()
    neg_inf = float("-inf")
    value = [0.0] * (n + 1)                       # V_0 = 0 (budget exhausted)
    for _ in range(max_steps):
        new = [neg_inf] * (n + 1)
        new[n] = 0.0                              # terminal
        for s in range(1, n):
            best = neg_inf
            for d in range(1, 7):
                t = s + d
                if t > n:
                    q = r["invalid"] + value[s]
                elif t == n:
                    q = r["goal"]
                elif t in board.ladders:
                    q = r["ladder"] + value[jumps[t]]
                elif t in board.snakes:
                    q = r["snake"] + value[jumps[t]]
                else:
                    q = r["normal"] + value[t]
                best = max(best, q)
            new[s] = best
        value = new
    return value[1]


def path_return(board: Board, path: List[Tuple[int, int]], rewards: Optional[dict] = None) -> float:
    """Undiscounted return obtained by following a BFS path."""
    r = rewards or REWARDS
    total, jumps = 0.0, board.transitions()
    for pos, dice in path:
        t = pos + dice
        if t == board.size:
            total += r["goal"]
        elif t in board.ladders:
            total += r["ladder"]
        elif t in board.snakes:
            total += r["snake"]
        else:
            total += r["normal"]
    return total


def has_positive_return_cycle(board: Board, rewards: Optional[dict] = None) -> bool:
    """True if a reward-positive cycle is reachable from square 1 (Bellman-Ford, maximisation).

    With the official rewards (+20 ladder, -20 snake, -1 normal) a cycle such as
    ladder -> ladder -> snake can have positive net return, letting a return-maximising agent
    farm reward forever instead of finishing. Experiment boards are required to avoid this so that
    "maximise return" and "reach the goal quickly" agree.
    """
    r = rewards or REWARDS
    n, jumps = board.size, board.transitions()
    edges = []                                    # (from, to, reward); node n is terminal (no out-edges)
    for s in range(1, n):
        for d in range(1, 7):
            t = s + d
            if t > n:
                edges.append((s, s, r["invalid"]))
            elif t == n:
                edges.append((s, n, r["goal"]))
            elif t in board.ladders:
                edges.append((s, jumps[t], r["ladder"]))
            elif t in board.snakes:
                edges.append((s, jumps[t], r["snake"]))
            else:
                edges.append((s, t, r["normal"]))
    dist = {1: 0.0}
    for _ in range(n + 1):
        changed = False
        for u, v, w in edges:
            if u in dist and dist[u] + w > dist.get(v, float("-inf")) + 1e-9:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return False
    return True
