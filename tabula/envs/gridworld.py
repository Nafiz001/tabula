"""Grid environments: the smallest worlds where the algorithms are visible.

A grid is worth building first because everything about it can be checked by
hand. The optimal path is countable, the value of each square is arguable on
paper, and when an agent gets it wrong you can see exactly where. Every bug in
a learning algorithm is easier to find here than in anything with pixels.
"""

from __future__ import annotations

import numpy as np

from tabula.core.types import Action, State, Step

# Actions, in a fixed order the renderer and tests both rely on.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_ARROWS = ["^", ">", "v", "<"]
_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class GridWorld:
    """A rectangular grid with walls, a goal, and optional slippery movement.

    The reward structure is the part worth thinking about. A small negative
    reward on every step is what makes the agent prefer short paths: with a
    reward of zero everywhere except the goal, every path that eventually
    arrives is equally good, and the agent has no reason to hurry. This is
    reward shaping in its simplest form, and it is also the easiest place to
    accidentally teach the wrong thing.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        walls: set[tuple[int, int]] | None = None,
        step_reward: float = -0.04,
        goal_reward: float = 1.0,
        slip: float = 0.0,
        max_steps: int = 200,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.walls = walls or set()
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        # slip is the probability the intended move is replaced by a random
        # one. It turns a puzzle into a genuine decision problem: with slip,
        # walking beside a hazard is risky even when it is the shortest route.
        self.slip = slip
        self.max_steps = max_steps

        self._rng = np.random.default_rng()
        self._pos = start
        self._steps = 0

        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("the start or goal is inside a wall")

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return 4

    def state_index(self, pos: tuple[int, int]) -> int:
        """Flatten a (row, col) into the single integer a table is indexed by."""
        return pos[0] * self.cols + pos[1]

    def index_to_pos(self, index: int) -> tuple[int, int]:
        return divmod(index, self.cols)

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self.start
        self._steps = 0
        return self.state_index(self._pos)

    def _move(self, pos: tuple[int, int], action: Action) -> tuple[int, int]:
        """Apply an action, treating walls and edges as a bump that costs a turn."""
        dr, dc = _DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return pos
        if (nr, nc) in self.walls:
            return pos
        return (nr, nc)

    def step(self, action: Action) -> Step:
        if not 0 <= action < 4:
            raise ValueError(f"action {action} is not one of the four directions")

        chosen = action
        if self.slip > 0 and self._rng.random() < self.slip:
            chosen = int(self._rng.integers(4))

        self._pos = self._move(self._pos, chosen)
        self._steps += 1

        if self._pos == self.goal:
            return Step(self.state_index(self._pos), self.goal_reward, terminated=True)

        # Hitting the step limit is truncation, not termination: the agent did
        # not fail, we just stopped watching, and the state it is in still has
        # value.
        truncated = self._steps >= self.max_steps
        return Step(self.state_index(self._pos), self.step_reward, terminated=False, truncated=truncated)

    def render(self, policy: np.ndarray | None = None) -> str:
        """Draw the grid, optionally showing which way a policy points."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.walls:
                    row.append("#")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) == self.start and policy is None:
                    row.append("S")
                elif policy is not None:
                    row.append(ACTION_ARROWS[int(policy[self.state_index((r, c))])])
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)


class CliffWalking:
    """The classic 4x12 cliff, and the clearest demonstration in the subject.

    A row of squares along the bottom sends the agent back to the start with a
    reward of -100. The shortest path runs right alongside it.

    This environment exists in this project for one reason: Q-learning and SARSA
    learn *different policies* here, and the difference is not a bug in either.
    Q-learning is off-policy -- it learns the value of acting greedily, so it
    finds the optimal path hugging the cliff edge. SARSA is on-policy -- it
    learns the value of the policy it is actually following, epsilon-greedy
    exploration included, so it accounts for the chance of a random step
    knocking it off and takes the safer route one row up.

    Neither is wrong. Q-learning answers "what is the best path", SARSA answers
    "what is the best path *for someone who sometimes slips*". Seeing both come
    out of the same training loop is the point.
    """

    def __init__(self, rows: int = 4, cols: int = 12, max_steps: int = 500) -> None:
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        # Everything along the bottom between start and goal.
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}

        self._pos = self.start
        self._steps = 0
        self._rng = np.random.default_rng()

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return 4

    def state_index(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def index_to_pos(self, index: int) -> tuple[int, int]:
        return divmod(index, self.cols)

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self.start
        self._steps = 0
        return self.state_index(self._pos)

    def step(self, action: Action) -> Step:
        dr, dc = _DELTAS[action]
        nr = min(max(self._pos[0] + dr, 0), self.rows - 1)
        nc = min(max(self._pos[1] + dc, 0), self.cols - 1)
        self._pos = (nr, nc)
        self._steps += 1

        if self._pos in self.cliff:
            # Falling costs heavily and returns the agent to the start, without
            # ending the episode -- so the mistake is felt rather than escaped.
            self._pos = self.start
            return Step(self.state_index(self._pos), -100.0, terminated=False,
                        truncated=self._steps >= self.max_steps)

        if self._pos == self.goal:
            return Step(self.state_index(self._pos), -1.0, terminated=True)

        # Every step costs 1, so the agent is pushed toward shorter routes.
        return Step(self.state_index(self._pos), -1.0, terminated=False,
                    truncated=self._steps >= self.max_steps)

    def render(self, policy: np.ndarray | None = None) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.cliff:
                    row.append("C")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) == self.start:
                    row.append("S")
                elif policy is not None:
                    row.append(ACTION_ARROWS[int(policy[self.state_index((r, c))])])
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)


def default_gridworld() -> GridWorld:
    """A 5x5 grid with a wall to walk around, used by the tests and the CLI."""
    return GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        walls={(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)},
    )
