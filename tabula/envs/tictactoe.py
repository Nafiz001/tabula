"""Tic-tac-toe: small enough to solve exactly, which is why it is here.

Every other environment in this project can only be evaluated by comparison --
is the agent better than random, better than before. Tic-tac-toe can be solved
outright: 5,478 reachable positions, so exhaustive minimax gives the true value
of every one. That turns "the agent learned" from an impression into a
measurement. An agent can be checked against perfect play and its policy
compared move by move against the optimal one.

Board convention: +1 and -1 for the two players, 0 for empty, indexed 0-8 in
reading order. Rewards are always from the perspective of the player who just
moved, which is what lets one network play both sides -- flip the board, flip
the sign.
"""

from __future__ import annotations

import numpy as np

from tabula.core.types import Action, Step

# The eight ways to win, as index triples.
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


class TicTacToe:
    """A two-player game with an interface built for self-play."""

    def __init__(self) -> None:
        self.board = np.zeros(9, dtype=np.int8)
        self._player = 1

    @property
    def n_actions(self) -> int:
        return 9

    @property
    def current_player(self) -> int:
        return self._player

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=np.int8)
        self._player = 1
        return self.board.copy()

    def clone(self) -> "TicTacToe":
        """A deep copy, so search can explore without disturbing the real game."""
        other = TicTacToe()
        other.board = self.board.copy()
        other._player = self._player
        return other

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.board == 0)

    def winner(self) -> int:
        """+1, -1, or 0 for no winner yet or a draw."""
        for a, b, c in WIN_LINES:
            total = self.board[a] + self.board[b] + self.board[c]
            if total == 3:
                return 1
            if total == -3:
                return -1
        return 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or not np.any(self.board == 0)

    def step(self, action: Action) -> Step:
        if self.board[action] != 0:
            raise ValueError(f"square {action} is already taken")

        mover = self._player
        self.board[action] = mover
        self._player = -mover

        won = self.winner()
        if won != 0:
            # From the mover's perspective: they just won.
            return Step(self.board.copy(), 1.0, terminated=True)
        if not np.any(self.board == 0):
            return Step(self.board.copy(), 0.0, terminated=True)
        return Step(self.board.copy(), 0.0, terminated=False)

    def canonical(self) -> np.ndarray:
        """The board as the player to move sees it, with their pieces as +1.

        This is what makes one network able to play both sides. Without it the
        network would have to learn "win as X" and "win as O" as separate
        skills, doubling the work for no reason -- the game is symmetric.
        """
        return self.board * self._player

    def key(self) -> tuple:
        """A hashable position, for tables and transposition."""
        return (tuple(int(v) for v in self.board), self._player)

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[int(self.board[r * 3 + c])] for c in range(3)))
        return "\n".join(rows)


class TicTacToeSolver:
    """Exhaustive minimax: the ground truth this project measures against.

    Tic-tac-toe has only a few thousand reachable positions, so every one can be
    evaluated exactly rather than estimated. That is unusual and valuable: it
    means an agent's claim to have learned something can be checked rather than
    believed. The solver answers two questions -- what is this position worth
    under perfect play, and which moves achieve that -- and both are used to
    grade the learned agents.

    Results are memoised by position, so the whole game tree is walked once.
    """

    def __init__(self) -> None:
        self._values: dict[tuple, int] = {}

    def value(self, game: TicTacToe) -> int:
        """The game's value under perfect play, from the mover's perspective.

        +1 the player to move wins, -1 they lose, 0 a draw. From the empty
        board this is 0: tic-tac-toe is a draw with best play, which is itself
        a useful thing for the tests to assert.
        """
        key = game.key()
        if key in self._values:
            return self._values[key]

        winner = game.winner()
        if winner != 0:
            # The player to move is facing a board that is already won, so this
            # position is a loss for them.
            result = -1 if winner == -game.current_player else 1
            self._values[key] = result
            return result

        if not np.any(game.board == 0):
            self._values[key] = 0
            return 0

        # Negamax: the opponent's best outcome is the negation of ours, so one
        # recursion covers both players rather than two mutually recursive
        # maximise/minimise functions.
        best = -2
        for action in game.legal_actions():
            child = game.clone()
            child.step(action)
            best = max(best, -self.value(child))
            if best == 1:
                break  # cannot do better than winning

        self._values[key] = best
        return best

    def best_actions(self, game: TicTacToe) -> list[int]:
        """Every move that achieves the position's true value.

        A set rather than one move, because several may be equally optimal --
        grading an agent against a single arbitrary choice would mark correct
        play wrong.
        """
        best_value = self.value(game)
        out = []
        for action in game.legal_actions():
            child = game.clone()
            child.step(action)
            if -self.value(child) == best_value:
                out.append(int(action))
        return out

    def positions_solved(self) -> int:
        return len(self._values)


class PerfectPlayer:
    """An opponent that never loses. The benchmark every agent is measured on.

    It picks uniformly among the optimal moves rather than the first one, so an
    agent cannot beat it by memorising a single line of play.
    """

    def __init__(self, solver: TicTacToeSolver | None = None, seed: int | None = None) -> None:
        self.solver = solver or TicTacToeSolver()
        self.rng = np.random.default_rng(seed)

    def act(self, game: TicTacToe) -> int:
        return int(self.rng.choice(self.solver.best_actions(game)))


class RandomPlayer:
    """Plays legal moves at random: the floor a trained agent must clear."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: TicTacToe) -> int:
        return int(self.rng.choice(game.legal_actions()))
