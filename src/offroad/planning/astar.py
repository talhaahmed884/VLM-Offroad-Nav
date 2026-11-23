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


def shortest_path_tree(cost: np.ndarray, start: Cell) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest paths from start to every cell at once (Dijkstra; same moves and costs as `astar`).

    Returns (come, parent):
        come: (nx, ny) cheapest path cost to each cell; inf where it can't be reached.
        parent: (nx, ny, 2) int, the previous cell on that path; -1 for the start and unreachable cells.
    One run gives the path to any goal via `path_from_tree`, which is much faster than one A* per goal.
    """
    nx, ny = cost.shape
    best = np.full(cost.shape, np.inf)
    parent = np.full((nx, ny, 2), -1, dtype=int)
    if not np.isfinite(cost[start]):
        return best, parent
    grid = cost.tolist()  # plain Python lists are much faster to index in the loop
    best_list = best.tolist()
    best_list[start[0]][start[1]] = 0.0
    parents: dict[Cell, Cell] = {}
    frontier = [(0.0, start)]
    while frontier:
        cost_so_far, (i, j) = heapq.heappop(frontier)
        if cost_so_far > best_list[i][j]:
            continue  # an older, more expensive entry for this cell
        for di, dj, step in NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or math.isinf(grid[ni][nj]):
                continue
            if di and dj and (math.isinf(grid[ni][j]) or math.isinf(grid[i][nj])):
                continue
            new_cost = cost_so_far + step * (grid[i][j] + grid[ni][nj]) / 2.0
            if new_cost < best_list[ni][nj]:
                best_list[ni][nj] = new_cost
                parents[(ni, nj)] = (i, j)
                heapq.heappush(frontier, (new_cost, (ni, nj)))
    for cell, previous in parents.items():
        parent[cell] = previous
    return np.array(best_list), parent


def path_from_tree(parent: np.ndarray, goal: Cell) -> list[Cell]:
    """Cells from the start to `goal`, read back from a `shortest_path_tree` parent array."""
    path = [goal]
    while True:
        previous = tuple(int(v) for v in parent[path[-1]])
        if previous[0] < 0:
            return path[::-1]
        path.append(previous)


def cost_to_come(cost: np.ndarray, start: Cell) -> np.ndarray:
    """Cheapest path cost from start to every cell (Dijkstra; same moves and costs as `astar`).

    Returns (nx, ny) float: inf for cells that cannot be reached. All inf if the start is blocked.
    """
    return shortest_path_tree(cost, start)[0]


def goal_from_costs(come: np.ndarray, spec: GridSpec, x: float, y: float, cost_weight: float) -> Cell | None:
    """The `choose_goal` rule on an already computed cost-to-come grid (see `choose_goal`)."""
    reachable = np.isfinite(come)
    if not reachable.any():
        return None
    xs = spec.x_min + (np.arange(spec.shape[0]) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(spec.shape[1]) + 0.5) * spec.resolution
    shortfall = np.hypot(xs[:, None] - x, ys[None, :] - y)
    # Score only reachable cells: multiplying an unreachable cell's inf cost by a weight of 0
    # would give NaN, which argmin would pick.
    score = np.full(come.shape, np.inf)
    score[reachable] = shortfall[reachable] + cost_weight * come[reachable] * spec.resolution
    i, j = np.unravel_index(np.argmin(score), come.shape)
    return int(i), int(j)


def choose_goal(
    cost: np.ndarray, spec: GridSpec, start: Cell, x: float, y: float, cost_weight: float
) -> Cell | None:
    """Pick the planning goal for a target at FLU point (x, y), meters.

    Every reachable cell is scored as
        distance from its center to the target (m) + cost_weight * cost to drive there (cost x m),
    and the lowest score wins. So the path gets as close to the target as it can, but only
    drives further for a gain that is worth it: with cost_weight = 0.1, getting 1 m closer is
    worth up to 10 m of extra smooth driving. With cost_weight = 0 it is simply the closest
    reachable cell. None only if the start cell is blocked.
    """
    return goal_from_costs(cost_to_come(cost, start), spec, x, y, cost_weight)


def path_length(path: list[Cell], spec: GridSpec) -> float:
    """Length of a cell path in meters."""
    cells = np.array(path, dtype=float)
    return float(np.linalg.norm(np.diff(cells, axis=0), axis=1).sum() * spec.resolution)
