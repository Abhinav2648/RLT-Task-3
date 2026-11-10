"""
maze_env.py
-----------
Grid-based Maze Environment for Reinforcement Learning.

The maze is represented as a 2-D NumPy array:
    0 = free cell
    1 = wall / obstacle
    2 = start  (also free cell)
    3 = goal   (terminal state)

Actions
-------
    0 = UP    (-1,  0)
    1 = DOWN  (+1,  0)
    2 = LEFT  ( 0, -1)
    3 = RIGHT ( 0, +1)

Rewards
-------
    GOAL reached   : +100  (encourage reaching the goal)
    Normal step    :  -1   (small penalty to prefer shorter paths)
    Hit wall/border:  -5   (discourage illegal moves)
"""

import numpy as np
from typing import Tuple, List, Optional


# ── Default 6×6 maze layout ──────────────────────────────────────────────────
# 0 = free, 1 = wall, 2 = start, 3 = goal
DEFAULT_MAZE = np.array([
    [2, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 3],
], dtype=np.int32)

# Action definitions
ACTION_NAMES   = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS  = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # (Δrow, Δcol)
N_ACTIONS      = 4

# Cell-type constants
CELL_FREE  = 0
CELL_WALL  = 1
CELL_START = 2
CELL_GOAL  = 3


class MazeEnv:
    """
    A configurable grid-based maze environment.

    Parameters
    ----------
    maze_layout : np.ndarray, optional
        2-D integer array describing the maze.  Defaults to DEFAULT_MAZE.
    reward_goal   : float  Reward for reaching the goal.
    reward_step   : float  Reward for each normal (legal) move.
    reward_wall   : float  Penalty for attempting an illegal move.
    max_steps     : int    Maximum steps allowed per episode.
    """

    def __init__(
        self,
        maze_layout: Optional[np.ndarray] = None,
        reward_goal: float  =  100.0,
        reward_step: float  =   -1.0,
        reward_wall: float  =   -5.0,
        max_steps:   int    =  200,
    ):
        self.maze_layout = (
            maze_layout.copy() if maze_layout is not None else DEFAULT_MAZE.copy()
        )
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        self.max_steps   = max_steps

        self.n_rows, self.n_cols = self.maze_layout.shape
        self.n_states = self.n_rows * self.n_cols

        # Locate start and goal
        start_positions = list(zip(*np.where(self.maze_layout == CELL_START)))
        goal_positions  = list(zip(*np.where(self.maze_layout == CELL_GOAL)))

        if not start_positions:
            raise ValueError("Maze must contain exactly one start cell (value=2).")
        if not goal_positions:
            raise ValueError("Maze must contain exactly one goal cell (value=3).")

        self.start_pos: Tuple[int, int] = start_positions[0]
        self.goal_pos:  Tuple[int, int] = goal_positions[0]

        # Verify there is a valid path from start to goal
        if not self._has_valid_path():
            raise ValueError(
                "No valid path exists from the start to the goal in the provided maze."
            )

        # Runtime state
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.steps_taken: int = 0
        self.done: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> int:
        """Reset the environment to the start state. Returns the initial state index."""
        self.agent_pos  = self.start_pos
        self.steps_taken = 0
        self.done        = False
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Apply action and return (next_state, reward, done, info).

        Parameters
        ----------
        action : int   0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict  (contains 'reason' and 'agent_pos')
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset() before stepping again.")

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Check for boundary / wall collision
        if not self._is_valid(nr, nc):
            reward   = self.reward_wall
            info     = {"reason": "wall_or_boundary", "agent_pos": self.agent_pos}
            # Agent stays in place; no state change
        else:
            self.agent_pos = (nr, nc)
            if self.maze_layout[nr, nc] == CELL_GOAL:
                reward      = self.reward_goal
                self.done   = True
                info        = {"reason": "goal_reached", "agent_pos": self.agent_pos}
            else:
                reward = self.reward_step
                info   = {"reason": "normal_step", "agent_pos": self.agent_pos}

        self.steps_taken += 1
        if self.steps_taken >= self.max_steps and not self.done:
            self.done = True
            info["reason"] = "max_steps_exceeded"

        return self._pos_to_state(self.agent_pos), reward, self.done, info

    def render(self) -> List[List[str]]:
        """
        Return the maze as a 2-D list of single characters for display.

        Symbols
        -------
        '#' = wall
        'S' = start
        'G' = goal
        'A' = agent's current position
        '.' = free cell
        """
        grid = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    row.append("A")
                elif self.maze_layout[r, c] == CELL_WALL:
                    row.append("#")
                elif self.maze_layout[r, c] == CELL_START:
                    row.append("S")
                elif self.maze_layout[r, c] == CELL_GOAL:
                    row.append("G")
                else:
                    row.append(".")
            grid.append(row)
        return grid

    # ── State / position helpers ──────────────────────────────────────────────

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to a flat state index."""
        return pos[0] * self.n_cols + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert a flat state index to (row, col)."""
        return divmod(state, self.n_cols)

    def _is_valid(self, r: int, c: int) -> bool:
        """Return True if (r, c) is inside the grid and not a wall."""
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.maze_layout[r, c] != CELL_WALL

    # ── Path validation ───────────────────────────────────────────────────────

    def _has_valid_path(self) -> bool:
        """BFS to check whether the goal is reachable from the start."""
        visited = set()
        queue   = [self.start_pos]
        visited.add(self.start_pos)
        while queue:
            r, c    = queue.pop(0)
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in ACTION_DELTAS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self._is_valid(nr, nc):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_state(self) -> int:
        return self._pos_to_state(self.agent_pos)

    @property
    def goal_state(self) -> int:
        return self._pos_to_state(self.goal_pos)

    @property
    def start_state(self) -> int:
        return self._pos_to_state(self.start_pos)

    @property
    def action_space(self):
        return list(range(N_ACTIONS))


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = MazeEnv()
    state = env.reset()
    print(f"Start state: {state}, position: {env.agent_pos}")
    print(f"Goal  state: {env.goal_state}, position: {env.goal_pos}")
    print(f"Grid size  : {env.n_rows}×{env.n_cols}")
    print(f"Valid path : {env._has_valid_path()}")
    for row in env.render():
        print(" ".join(row))
