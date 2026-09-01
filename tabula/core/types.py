"""The vocabulary every environment and agent shares.

Reinforcement learning has one loop at its centre, and everything else is
detail around it: the agent sees a state, picks an action, and the environment
answers with a reward and a new state. That is the whole interface, and keeping
it this small is what lets a tabular agent and a neural one plug into the same
training loop without either knowing about the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

# A state is whatever the environment says it is. Tabular environments use an
# integer index; continuous ones use a float vector. The agent decides which it
# can handle, not the environment.
State = int | np.ndarray

Action = int


@dataclass(frozen=True)
class Step:
    """What the environment returns after one action.

    ``terminated`` and ``truncated`` are separate on purpose, and the
    distinction is not cosmetic. Terminated means the episode genuinely ended --
    the agent fell off the cliff, the game was won -- so the value of the next
    state is zero. Truncated means we simply stopped watching, usually at a step
    limit, and the next state still has value. Collapsing them into one "done"
    flag teaches the agent that running out of time is as bad as failing, which
    quietly poisons every value estimate near the limit.
    """

    state: State
    reward: float
    terminated: bool
    truncated: bool = False
    info: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        """True when the episode is over for any reason."""
        return self.terminated or self.truncated


@runtime_checkable
class Env(Protocol):
    """The environment interface.

    Deliberately close to the Gymnasium API, so anyone who has used that will
    recognise it -- but implemented here rather than imported, because the
    point of the project is that nothing underneath is borrowed.
    """

    @property
    def n_actions(self) -> int:
        """How many discrete actions are available."""
        ...

    def reset(self, seed: int | None = None) -> State:
        """Start a new episode and return the first state."""
        ...

    def step(self, action: Action) -> Step:
        """Apply an action and return what happened."""
        ...


@runtime_checkable
class TabularEnv(Env, Protocol):
    """An environment whose states can be counted.

    Tabular methods store one value per state, so they need to know how many
    there are. An environment that cannot answer this -- anything continuous --
    needs a function approximator instead, which is the entire reason deep
    reinforcement learning exists.
    """

    @property
    def n_states(self) -> int:
        ...


@runtime_checkable
class TwoPlayerEnv(Protocol):
    """A game two agents play against each other.

    Self-play needs a different shape from the single-agent loop: there is a
    current player, the legal moves change with the position, and a reward
    means "good for whom". Rewards here are always from the perspective of the
    player who just moved, which is what makes the same network usable for both
    sides -- flip the board, flip the sign.
    """

    @property
    def n_actions(self) -> int:
        ...

    def reset(self) -> np.ndarray:
        ...

    def legal_actions(self) -> np.ndarray:
        """Indices of the moves available in the current position."""
        ...

    def step(self, action: Action) -> Step:
        ...

    def clone(self) -> "TwoPlayerEnv":
        """A deep copy, so search can explore without disturbing the real game."""
        ...

    @property
    def current_player(self) -> int:
        """+1 or -1."""
        ...
