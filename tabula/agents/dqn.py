"""Deep Q-Network: Q-learning with a neural network instead of a table.

The algorithm is the same one from the tabular file. What changes is that Q is
now a function the network computes rather than a number looked up, and that
single substitution breaks Q-learning in two specific ways. DQN is really the
name of the two fixes:

1. **Correlated samples.** Consecutive states are nearly identical, so training
   on the stream in order means every gradient step sees the same thing. Fixed
   by the replay buffer, which hands back random batches from a large window of
   past experience.

2. **A moving target.** The target is ``r + gamma * max Q(s')``, computed with
   the same network being updated -- so every step changes the thing it is
   trying to match. The network chases its own tail and the values diverge.
   Fixed by the target network: a frozen copy that supplies the targets, synced
   from the live network every so often. The target sits still long enough to
   be aimed at.

Both fixes exist to buy back an assumption that the table version got for free.
That is the whole story of deep reinforcement learning in miniature.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from tabula.core.replay import ReplayBuffer
from tabula.core.schedules import Constant, Schedule
from tabula.core.types import Action, Step


class QNetwork(nn.Module):
    """A small multilayer perceptron from state to one Q-value per action.

    Outputting every action's value at once, rather than taking the action as
    an input, means a single forward pass gives the whole row -- which is what
    the max in the update needs. Two hidden layers of 128 units is far more than
    CartPole requires and still trains in seconds on a CPU.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN:
    """Deep Q-learning with experience replay and a target network."""

    on_policy = False

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon: Schedule | float = 0.1,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        # learning_starts stops the first updates being computed from a handful
        # of near-identical transitions, which is exactly the correlation the
        # buffer exists to avoid.
        learning_starts: int = 1_000,
        train_every: int = 1,
        # 200 rather than something larger, chosen by measurement: at 1000 the
        # target is too stale and CartPole plateaus around 120 steps, while at
        # 200 the same agent reaches the 500-step ceiling.
        target_sync_every: int = 200,
        double: bool = True,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = discount
        self.epsilon = epsilon if isinstance(epsilon, Schedule) else Constant(epsilon)
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.target_sync_every = target_sync_every
        self.double = double
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        # The target network starts as an exact copy and is only ever updated by
        # copying, never by gradient descent.
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size, state_dim, seed=seed)

        self.steps = 0
        self.updates = 0
        self.last_loss = 0.0

    def current_epsilon(self) -> float:
        return self.epsilon.value(self.steps)

    def act(self, state: np.ndarray, greedy: bool = False) -> Action:
        if not greedy and self.rng.random() < self.current_epsilon():
            return int(self.rng.integers(self.n_actions))

        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(tensor).argmax(dim=1).item())

    def update(self, state, action: Action, step: Step, next_action=None) -> float:
        """Store the transition, then train if there is enough experience."""
        self.buffer.add(np.asarray(state, dtype=np.float32), action, step.reward,
                        np.asarray(step.state, dtype=np.float32), step.terminated)
        self.steps += 1

        if len(self.buffer) < self.learning_starts or self.steps % self.train_every != 0:
            return 0.0
        return self._train_step()

    def _train_step(self) -> float:
        batch = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        terminated = torch.as_tensor(batch.terminated, device=self.device)

        # The value the network currently predicts for the actions actually taken.
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # Double DQN: the live network picks the best next action, the
                # target network scores it. Using one network for both -- as
                # plain DQN does -- means any action whose value is
                # overestimated by noise is both chosen and trusted, so errors
                # compound upward. Splitting the two decisions removes most of
                # that bias for one extra forward pass.
                next_actions = self.q_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            # (1 - terminated) zeroes the bootstrap for genuinely terminal
            # states only. Truncated ones keep their value, which is why the
            # buffer stores terminated rather than done.
            targets = rewards + self.gamma * next_q * (1.0 - terminated)

        # Huber loss rather than squared error. A single wildly wrong target --
        # common early on, when the network is guessing -- produces a gradient
        # proportional to the error under MSE, which can wreck the weights in
        # one step. Huber is quadratic near zero and linear beyond, so outliers
        # are bounded.
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimiser.zero_grad()
        loss.backward()
        # Clipping is a second guard on the same failure: it bounds the update
        # regardless of how bad the batch was.
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self.updates += 1
        if self.updates % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.last_loss = float(loss.item())
        return self.last_loss

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Every action's value in a state, for inspection and tests."""
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(tensor).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "steps": self.steps,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.steps = checkpoint.get("steps", 0)
