"""Self-play: an agent that learns a game with no opponent and no examples.

This is the purest form of the project's premise. There is no dataset, no
teacher, and no hand-written heuristic -- the agent plays itself, and the only
signal is who won. Everything it knows about the game it worked out from that.

Two ideas make it work in a two-player game.

**Canonical boards.** Every position is stored from the point of view of the
player to move, with their own pieces as +1. So "three in a row for me" is one
pattern rather than two, and the same table serves both sides. Without it the
agent would learn to play X and to play O as separate skills, at twice the cost,
for a game that is symmetric.

**Negated bootstrapping.** After the agent moves, the *opponent* moves next, so
the next position is good for them exactly when it is bad for the agent. The
value therefore flips sign as it propagates:

    Q(s,a) <- Q(s,a) + lr * [ r - gamma * max_a' Q(s', a') - Q(s,a) ]

That minus sign is the whole difference from single-agent Q-learning, and
getting it wrong produces an agent that confidently plays into losses -- it has
learned to help its opponent.
"""

from __future__ import annotations

import numpy as np

from tabula.core.schedules import Constant, Schedule
from tabula.envs.tictactoe import TicTacToe


class SelfPlayQLearning:
    """Tabular Q-learning over canonical positions, trained by playing itself.

    The table is a dictionary rather than an array because the reachable
    positions are sparse in the space of all boards: 5,478 of 3^9, and most of
    the rest are unreachable or illegal.
    """

    def __init__(
        self,
        learning_rate: float = 0.3,
        discount: float = 1.0,
        epsilon: Schedule | float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.lr = learning_rate
        # Discount is 1.0 here, deliberately. The game is short and finite, and
        # a win in six moves is worth exactly as much as a win in nine -- there
        # is nothing to discount for.
        self.gamma = discount
        self.epsilon = epsilon if isinstance(epsilon, Schedule) else Constant(epsilon)
        self.rng = np.random.default_rng(seed)

        self.q: dict[tuple, np.ndarray] = {}
        self.steps = 0
        self.games = 0

    def current_epsilon(self) -> float:
        return self.epsilon.value(self.games)

    def _row(self, canonical: np.ndarray) -> np.ndarray:
        """The Q-values for a position, created on first sight."""
        key = tuple(int(v) for v in canonical)
        row = self.q.get(key)
        if row is None:
            row = np.zeros(9, dtype=np.float64)
            self.q[key] = row
        return row

    def act(self, game: TicTacToe, greedy: bool = False) -> int:
        """Choose a move among the legal ones."""
        legal = game.legal_actions()
        if not greedy and self.rng.random() < self.current_epsilon():
            return int(self.rng.choice(legal))

        row = self._row(game.canonical())
        # Only legal moves are considered. Masking rather than penalising
        # illegal moves keeps the table meaning "how good is this move", rather
        # than mixing in "is this move even allowed".
        values = row[legal]
        best = legal[np.flatnonzero(values == values.max())]
        return int(self.rng.choice(best))

    def train(self, episodes: int = 50_000, seed: int | None = None) -> list[float]:
        """Play games against itself, learning from both sides of each one.

        Both players' moves are recorded and updated. Learning from only one
        side would halve the experience and, worse, leave the agent with no idea
        what a strong opponent does.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        game = TicTacToe()
        outcomes: list[float] = []

        for _ in range(episodes):
            game.reset()
            # (canonical position, action) for each move, in order.
            history: list[tuple[np.ndarray, int]] = []

            while not game.is_terminal():
                canonical = game.canonical()
                action = self.act(game)
                history.append((canonical, action))
                game.step(action)

            winner = game.winner()
            outcomes.append(float(winner))
            self.games += 1

            # Walk the game backwards, so the final reward propagates all the
            # way to the opening in a single pass rather than needing many
            # games to seep back one move at a time.
            #
            # The reward is +1 for the player who made the last move if they
            # won, and it alternates sign going back, because consecutive moves
            # belong to opposing players.
            final = 0.0
            if winner != 0:
                final = 1.0  # the player who moved last won

            next_value = None
            for i in range(len(history) - 1, -1, -1):
                canonical, action = history[i]
                row = self._row(canonical)

                if i == len(history) - 1:
                    target = final
                else:
                    # The position after this move belongs to the opponent, so
                    # its value counts against this player.
                    target = -self.gamma * next_value

                row[action] += self.lr * (target - row[action])
                self.steps += 1

                # What the position this move led *from* is worth, for the
                # move before it.
                next_value = float(row[action])

        return outcomes

    def greedy_move(self, game: TicTacToe) -> int:
        return self.act(game, greedy=True)

    @property
    def positions_learned(self) -> int:
        return len(self.q)


def play_match(
    player_x,
    player_o,
    games: int = 100,
    seed: int | None = None,
) -> dict[str, int]:
    """Play a series and count outcomes from X's point of view.

    Both players are called with the game and return a move, so an agent, a
    perfect player and a random player are all interchangeable here.
    """
    rng = np.random.default_rng(seed)
    results = {"x_wins": 0, "o_wins": 0, "draws": 0}
    game = TicTacToe()

    for _ in range(games):
        game.reset()
        while not game.is_terminal():
            mover = player_x if game.current_player == 1 else player_o
            action = _ask(mover, game)
            game.step(action)

        winner = game.winner()
        if winner == 1:
            results["x_wins"] += 1
        elif winner == -1:
            results["o_wins"] += 1
        else:
            results["draws"] += 1

    return results


def _ask(player, game: TicTacToe) -> int:
    """Call a player, whatever shape its interface happens to be."""
    if hasattr(player, "greedy_move"):
        return player.greedy_move(game)
    return player.act(game)


def evaluate_against_perfect(agent, solver, games: int = 200, seed: int | None = None) -> dict:
    """The measurement that matters: can this agent be beaten by a perfect one?

    Tic-tac-toe is a draw with best play, so a perfectly trained agent draws
    every game from both sides. Any loss is a concrete mistake, and the number
    of them is a hard score rather than an impression.
    """
    from tabula.envs.tictactoe import PerfectPlayer

    perfect = PerfectPlayer(solver, seed=seed)

    as_x = play_match(agent, perfect, games=games // 2, seed=seed)
    as_o = play_match(perfect, agent, games=games // 2, seed=None if seed is None else seed + 1)

    losses = as_x["o_wins"] + as_o["x_wins"]
    wins = as_x["x_wins"] + as_o["o_wins"]
    draws = as_x["draws"] + as_o["draws"]

    return {
        "games": games,
        "losses": losses,
        "wins": wins,
        "draws": draws,
        "loss_rate": losses / games,
    }


def policy_agreement(agent, solver, games: int = 500, seed: int | None = None) -> float:
    """How often the agent's move is one that perfect play would choose.

    A finer instrument than the win rate. An agent can draw every game against a
    perfect opponent while still playing loose moves in positions that opponent
    never steers into -- this walks a wide sample of reachable positions and
    grades each choice against the solver.
    """
    rng = np.random.default_rng(seed)
    game = TicTacToe()
    agreed = 0
    total = 0

    for _ in range(games):
        game.reset()
        while not game.is_terminal():
            # A random walk through the game tree, so the sample covers
            # positions a strong opponent would never allow.
            optimal = set(solver.best_actions(game))
            chosen = agent.greedy_move(game) if hasattr(agent, "greedy_move") else agent.act(game)
            agreed += int(chosen in optimal)
            total += 1
            game.step(int(rng.choice(game.legal_actions())))

    return agreed / total if total else 0.0
