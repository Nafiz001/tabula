"""Monte Carlo tree search, guided by a neural network.

Plain minimax evaluates every branch, which is why it solves tic-tac-toe in a
fraction of a second and cannot touch Go. MCTS does the opposite: it spends its
budget where the tree looks promising, building a lopsided tree that is deep
along good lines and shallow elsewhere.

The version here is the AlphaZero one, and it differs from classic MCTS in a way
worth being precise about. Classic MCTS estimates a leaf by playing the game out
at random -- a rollout. That is unbiased but extremely noisy, and it needs
thousands of playouts to say anything useful. AlphaZero replaces the rollout
with a single call to a value network: instead of guessing by playing badly a
thousand times, ask a function that has learned what good positions look like.
The network also supplies a prior over moves, so the search knows where to look
before it has looked anywhere.

Selection uses PUCT:

    score(a) = Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a))

Q is what the search has found so far; the second term is what the network
suggests, damped by how often that move has already been tried. Early on, when
every N is zero, the prior decides. As visits accumulate, measured value takes
over. That crossover is the whole design: prior knowledge steers the search
until search results are worth more than the prior.
"""

from __future__ import annotations

import math

import numpy as np


class Node:
    """One position in the search tree."""

    __slots__ = ("prior", "visits", "value_sum", "children", "player")

    def __init__(self, prior: float, player: int) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        # children maps action -> Node, and its emptiness marks an unexpanded
        # leaf.
        self.children: dict[int, "Node"] = {}
        self.player = player

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """Mean value over visits, from this node's player's perspective."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTS:
    """PUCT search over a two-player game.

    ``evaluate`` is a callback taking a game and returning ``(priors, value)``:
    a probability over the nine actions and a scalar in [-1, 1] for the player
    to move. Keeping it a callback rather than a network reference means the
    search can be tested against a hand-written evaluator, which is how the
    tests check it without training anything.
    """

    def __init__(
        self,
        evaluate,
        simulations: int = 100,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.evaluate = evaluate
        self.simulations = simulations
        # c_puct sets how much the prior is trusted against measured value.
        # Too low and search ignores the network; too high and it never
        # revisits the network's mistakes.
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.rng = np.random.default_rng(seed)

    def search(self, game, add_noise: bool = False) -> np.ndarray:
        """Run the simulations and return a visit-count distribution over moves.

        The visit counts, not the values, are the output. A move that search
        kept returning to is one it found worth examining, and the counts are a
        far more stable signal than the raw values -- which are averages over
        wildly varying numbers of samples.
        """
        root = Node(prior=0.0, player=game.current_player)
        self._expand(root, game, add_noise=add_noise)

        for _ in range(self.simulations):
            node = root
            scratch = game.clone()
            path = [node]

            # Selection: walk down by PUCT until reaching a leaf.
            while node.expanded:
                action, node = self._select_child(node)
                scratch.step(action)
                path.append(node)
                if scratch.is_terminal():
                    break

            # Evaluation: either the game is over, or the network guesses.
            if scratch.is_terminal():
                winner = scratch.winner()
                if winner == 0:
                    value = 0.0
                else:
                    # A terminal position is always a loss for the player to
                    # move -- the opponent just won by moving into it.
                    value = -1.0
            else:
                value = self._expand(node, scratch)

            # Backup: the value alternates sign up the path, because
            # consecutive nodes belong to opposing players.
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = -value

        counts = np.zeros(game.n_actions, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visits
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _select_child(self, node: Node) -> tuple[int, Node]:
        """Pick the child with the highest PUCT score."""
        total_visits = math.sqrt(sum(c.visits for c in node.children.values()))
        best_score = -float("inf")
        best = None

        for action, child in node.children.items():
            # The child's value is from the child's player's perspective, which
            # is the opponent of this node's player -- hence the negation.
            exploit = -child.value
            explore = self.c_puct * child.prior * total_visits / (1 + child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = (action, child)

        return best

    def _expand(self, node: Node, game, add_noise: bool = False) -> float:
        """Create the children of a leaf and return the network's value for it."""
        priors, value = self.evaluate(game)
        legal = game.legal_actions()

        # Illegal moves get no prior. Renormalising over the legal ones matters:
        # a network that has not yet learned the rules will happily put most of
        # its mass on occupied squares, and without this the search would waste
        # its budget on moves it cannot play.
        masked = np.zeros_like(priors)
        masked[legal] = priors[legal]
        total = masked.sum()
        masked = masked / total if total > 0 else _uniform_over(legal, len(priors))

        if add_noise and len(legal) > 0:
            # Dirichlet noise at the root, only during self-play. Without it,
            # a confident network explores nothing and self-play produces the
            # same game forever -- the agent stops learning because it stops
            # seeing anything new. This is the exploration mechanism for a
            # search-based agent, doing the job epsilon does for Q-learning.
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal))
            masked[legal] = (
                (1 - self.dirichlet_weight) * masked[legal]
                + self.dirichlet_weight * noise
            )

        for action in legal:
            node.children[int(action)] = Node(prior=float(masked[action]), player=-node.player)

        return float(value)


def _uniform_over(legal: np.ndarray, n_actions: int) -> np.ndarray:
    out = np.zeros(n_actions)
    if len(legal) > 0:
        out[legal] = 1.0 / len(legal)
    return out
