"""AlphaZero, scaled down to tic-tac-toe.

The loop is the interesting part, and it is short enough to state in full:

1. Play games against yourself, choosing moves by running MCTS with the current
   network.
2. Record, for each position, the search's visit distribution and who
   eventually won.
3. Train the network so its policy matches the search and its value matches the
   outcome.
4. Repeat.

What makes it work is that step 3 trains the network on something *better than
itself*. Search improves on the raw network -- a hundred simulations of
lookahead beat a single forward pass -- so the visit counts are a stronger
policy than the priors that produced them. Training toward them pulls the
network up. Then the improved network makes the next search stronger, which
produces better targets again. That mutual bootstrap is the whole idea, and it
needs no human games, no opening book and no heuristics: the only external fact
supplied is the rules.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tabula.core.mcts import MCTS
from tabula.envs.tictactoe import TicTacToe


class PolicyValueNet(nn.Module):
    """One trunk, two heads: where to move, and how good this is.

    Sharing the trunk is not just an economy. The features that say "this
    position is winning" are largely the features that say "play here", so
    learning them once from both signals is better than learning them twice from
    each. The value head also regularises the policy head, which otherwise
    happily overfits to whatever the search happened to visit.
    """

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        # 18 inputs: two 9-square planes, one for each player's pieces, always
        # from the perspective of the player to move.
        self.trunk = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 9)
        self.value_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        # Logits, not probabilities: the loss needs log-softmax, and softmaxing
        # twice quietly flattens the distribution.
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def encode(game: TicTacToe) -> np.ndarray:
    """Turn a position into network input, from the mover's point of view.

    Two planes rather than one signed plane. A single plane with +1 and -1 asks
    the first layer to learn that the sign means "whose", which it can do but
    need not; separate planes hand it that for free.
    """
    canonical = game.canonical()
    mine = (canonical == 1).astype(np.float32)
    theirs = (canonical == -1).astype(np.float32)
    return np.concatenate([mine, theirs])


class AlphaZeroAgent:
    """Self-play learning with tree search and a policy/value network."""

    def __init__(
        self,
        simulations: int = 50,
        learning_rate: float = 1e-3,
        c_puct: float = 1.4,
        hidden: int = 128,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.net = PolicyValueNet(hidden=hidden).to(self.device)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.simulations = simulations
        self.c_puct = c_puct
        self.iterations = 0
        # A window of recent self-play examples, trained on together rather
        # than one iteration at a time.
        #
        # Discarding each iteration's games after training on them once is the
        # obvious implementation and it does not converge: every batch is a few
        # hundred positions from a single snapshot of the policy, so the network
        # overfits that snapshot and then forgets it. Keeping a window means
        # each update sees positions from many policy versions, which is both
        # more data and more varied data. Measured on tic-tac-toe, this is the
        # difference between plateauing around 10 losses in 40 games against
        # perfect play and reaching zero.
        self.replay: list = []
        self.replay_window = 8000

    def _evaluate(self, game: TicTacToe) -> tuple[np.ndarray, float]:
        """The callback MCTS uses: priors and a value for one position."""
        with torch.no_grad():
            x = torch.as_tensor(encode(game), device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            priors = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            return priors, float(value.item())

    def _mcts(self, seed: int | None = None) -> MCTS:
        return MCTS(
            self._evaluate,
            simulations=self.simulations,
            c_puct=self.c_puct,
            seed=seed,
        )

    def self_play_game(self, temperature_moves: int = 4, seed: int | None = None):
        """Play one game against itself, returning training examples.

        Moves are sampled from the visit distribution for the first few plies
        and taken greedily after. That is a deliberate split: sampling early
        varies the openings so the training set covers the game, while playing
        greedily later means the recorded outcome reflects the agent's real
        strength rather than a blunder it sampled by accident.
        """
        game = TicTacToe()
        game.reset()
        search = self._mcts(seed)

        history: list[tuple[np.ndarray, np.ndarray, int]] = []
        move = 0

        while not game.is_terminal():
            policy = search.search(game, add_noise=True)
            history.append((encode(game), policy.copy(), game.current_player))

            if move < temperature_moves:
                action = int(self.rng.choice(len(policy), p=policy))
            else:
                action = int(np.argmax(policy))

            game.step(action)
            move += 1

        winner = game.winner()
        # The value target is the outcome from each position's own mover's
        # perspective, which is why the player is recorded alongside the state.
        examples = []
        for state, policy, player in history:
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == player else -1.0
            examples.append((state, policy, value))
        return examples

    def train(
        self,
        iterations: int = 20,
        games_per_iteration: int = 20,
        epochs: int = 4,
        batch_size: int = 64,
        seed: int | None = None,
        on_iteration=None,
    ) -> list[dict]:
        """Run the self-play and learning loop."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        log = []
        for iteration in range(iterations):
            examples = []
            for g in range(games_per_iteration):
                examples.extend(
                    self.self_play_game(seed=None if seed is None else seed + iteration * 1000 + g)
                )

            self.replay.extend(examples)
            if len(self.replay) > self.replay_window:
                self.replay = self.replay[-self.replay_window :]

            losses = self._learn(self.replay, epochs=epochs, batch_size=batch_size)
            self.iterations += 1

            entry = {
                "iteration": iteration,
                "examples": len(examples),
                "replay": len(self.replay),
                "policy_loss": losses[0],
                "value_loss": losses[1],
            }
            log.append(entry)
            if on_iteration is not None:
                on_iteration(entry)

        return log

    def _learn(self, examples, epochs: int, batch_size: int) -> tuple[float, float]:
        states = torch.as_tensor(np.array([e[0] for e in examples]), device=self.device)
        policies = torch.as_tensor(np.array([e[1] for e in examples], dtype=np.float32), device=self.device)
        values = torch.as_tensor(np.array([e[2] for e in examples], dtype=np.float32), device=self.device)

        n = len(examples)
        policy_loss_total = 0.0
        value_loss_total = 0.0
        batches = 0

        for _ in range(epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                logits, predicted = self.net(states[idx])

                # Cross-entropy against the search's distribution, not against a
                # single best move. The visit counts carry how close the
                # alternatives were, and collapsing them to an argmax throws
                # that away.
                log_probs = torch.log_softmax(logits, dim=1)
                policy_loss = -(policies[idx] * log_probs).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(predicted, values[idx])

                loss = policy_loss + value_loss
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                batches += 1

        return policy_loss_total / batches, value_loss_total / batches

    def greedy_move(self, game: TicTacToe) -> int:
        """Play the move search likes best, with no exploration noise."""
        policy = self._mcts().search(game, add_noise=False)
        legal = game.legal_actions()
        masked = np.zeros_like(policy)
        masked[legal] = policy[legal]
        return int(np.argmax(masked))

    def act(self, game: TicTacToe) -> int:
        return self.greedy_move(game)

    def raw_policy(self, game: TicTacToe) -> np.ndarray:
        """The network's move preferences with no search at all.

        Comparing this against greedy_move shows how much the tree search is
        contributing -- and, as training proceeds, how much of the search's
        strength the network has absorbed.
        """
        priors, _ = self._evaluate(game)
        legal = game.legal_actions()
        masked = np.zeros_like(priors)
        masked[legal] = priors[legal]
        total = masked.sum()
        return masked / total if total > 0 else masked

    def raw_move(self, game: TicTacToe) -> int:
        return int(np.argmax(self.raw_policy(game)))

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "iterations": self.iterations}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"])
        self.iterations = checkpoint.get("iterations", 0)


class RawNetworkPlayer:
    """The network alone, with search switched off.

    Useful as a control: it separates what the network has learned from what
    the search is doing for it at play time.
    """

    def __init__(self, agent: AlphaZeroAgent) -> None:
        self.agent = agent

    def greedy_move(self, game: TicTacToe) -> int:
        return self.agent.raw_move(game)

    def act(self, game: TicTacToe) -> int:
        return self.agent.raw_move(game)
