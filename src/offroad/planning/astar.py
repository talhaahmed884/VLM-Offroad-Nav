"""A* path planning on a 2D cost grid (pure functions, no file I/O).

Cells are [i, j] indices into an (nx, ny) cost grid (same layout as the cost map).
A cost of inf means the cell cannot be entered.
"""

import heapq
import math

import numpy as np

from offroad.mapping.elevation import GridSpec

SQRT2 = math.sqrt(2.0)
# (di, dj, step length in cells) for the 8 neighbors.
NEIGHBORS = [
    (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2),
]

Cell = tuple[int, int]


def astar(cost: np.ndarray, start: Cell, goal: Cell) -> tuple[list[Cell], float] | None:
    """Cheapest 8-connected path from start to goal.

    cost: (nx, ny) cost per cell, >= some positive minimum; inf = blocked.
    Moving from cell a to neighbor b costs step_length * (cost[a] + cost[b]) / 2,
    where step_length is 1 for side neighbors and sqrt(2) for diagonals.
    Diagonal moves may not cut the corner of a blocked cell.

    Returns (path, total_cost), with the path as a list of cells from start to goal,
    or None if the goal cannot be reached.
    """
    nx, ny = cost.shape
    if not (np.isfinite(cost[start]) and np.isfinite(cost[goal])):
        return None
    grid = cost.tolist()  # plain Python lists are much faster to index in the loop
    min_cost = float(cost[np.isfinite(cost)].min())

    def heuristic(cell: Cell) -> float:
        # Octile distance times the cheapest cell cost: never overestimates, so A* stays optimal.
        di, dj = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return min_cost * (di + dj + (SQRT2 - 2.0) * min(di, dj))

    best_cost = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    frontier = [(heuristic(start), 0.0, start)]
    done = set()

    while frontier:
        _, cost_so_far, cell = heapq.heappop(frontier)
        if cell == goal:
            return _reconstruct(came_from, goal), cost_so_far
        if cell in done:
            continue
        done.add(cell)

        i, j = cell
        for di, dj, step in NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or math.isinf(grid[ni][nj]):
                continue
            if di and dj and (math.isinf(grid[ni][j]) or math.isinf(grid[i][nj])):
                continue  # don't squeeze diagonally between two blocked cells
            new_cost = cost_so_far + step * (grid[i][j] + grid[ni][nj]) / 2.0
            if new_cost < best_cost.get((ni, nj), math.inf):
                best_cost[(ni, nj)] = new_cost
                came_from[(ni, nj)] = cell
                heapq.heappush(frontier, (new_cost + heuristic((ni, nj)), new_cost, (ni, nj)))
    return None


def _reconstruct(came_from: dict[Cell, Cell], goal: Cell) -> list[Cell]:
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return path[::-1]


def clear_footprint(cost: np.ndarray, spec: GridSpec, radius: float, free_cost: float = 1.0) -> np.ndarray:
    """Copy of the cost grid with every cell within `radius` meters of the FLU origin set to free_cost.

    The vehicle sits at the origin, so points there are the vehicle itself (or something
    touching it). Without this, the planner could start inside a blocked cell.
    """
    xs = spec.x_min + (np.arange(spec.shape[0]) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(spec.shape[1]) + 0.5) * spec.resolution
    inside = xs[:, None] ** 2 + ys[None, :] ** 2 <= radius**2
    cleared = cost.copy()
    cleared[inside] = free_cost
    return cleared


def nearest_free_cell(cost: np.ndarray, spec: GridSpec, x: float, y: float) -> Cell | None:
    """The cell with finite cost whose center is closest to the FLU point (x, y), or None if all are blocked."""
    xs = spec.x_min + (np.arange(spec.shape[0]) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(spec.shape[1]) + 0.5) * spec.resolution
    distance = (xs[:, None] - x) ** 2 + (ys[None, :] - y) ** 2
    distance[~np.isfinite(cost)] = np.inf
    if np.isinf(distance.min()):
        return None
    i, j = np.unravel_index(np.argmin(distance), cost.shape)
    return int(i), int(j)


def path_length(path: list[Cell], spec: GridSpec) -> float:
    """Length of a cell path in meters."""
    cells = np.array(path, dtype=float)
    return float(np.linalg.norm(np.diff(cells, axis=0), axis=1).sum() * spec.resolution)
