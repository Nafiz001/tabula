"""Experience replay: the buffer that makes deep Q-learning work at all.

Two problems make naive neural Q-learning fail, and this file fixes one of them.

Consecutive states in an episode are almost identical -- the pole leans a
fraction further, the cart moves a centimetre. Training a network on that stream
in order means every gradient step sees nearly the same input, so the network
chases the current situation and forgets everything else. It is the opposite of
the independent, identically distributed data every supervised method assumes.

A replay buffer breaks the correlation. Transitions go in as they happen and
come out in random batches, so a single update sees experiences from all over
the episode and from many episodes ago. It also means each transition is learned
from more than once, which matters when generating experience is the expensive
part.

(The second problem, the moving target, is fixed by the target network in the
DQN agent.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    """A batch of transitions, ready for the network."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    # terminated, not done: a truncated episode's final state still has value,
    # and zeroing its bootstrap would teach the agent that the step limit is a
    # failure.
    terminated: np.ndarray

    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly.

    Preallocated arrays rather than a list of tuples: a buffer of a hundred
    thousand transitions is sampled from on every training step, and building a
    batch by indexing into contiguous arrays is far cheaper than gathering
    objects. The ring overwrites the oldest experience, so the buffer holds a
    sliding window of recent behaviour.
    """

    def __init__(self, capacity: int, state_dim: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

        self._position = 0
        self._size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Draw a uniform random batch.

        Sampling with replacement, which is standard: at realistic buffer sizes
        a duplicate within one batch is rare, and avoiding it would cost more
        than it saves.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            terminated=self.terminated[idx],
        )
