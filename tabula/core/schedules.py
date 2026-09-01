"""Schedules for values that change as training proceeds.

The one that matters is epsilon, the probability of acting randomly. It encodes
the exploration/exploitation trade-off, which is the problem reinforcement
learning is really about: an agent that always takes its current best action
never discovers a better one, and an agent that always explores never uses what
it knows.

The resolution is to explore a lot early, when the estimates are worthless, and
less later, when they are not. That is what decaying epsilon does, and getting
the decay wrong is the most common reason a correct algorithm fails to learn --
too fast and it commits to a bad policy it never questions, too slow and it
spends the whole run flailing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Schedule:
    """A value that changes with the step count."""

    def value(self, step: int) -> float:
        raise NotImplementedError


@dataclass
class Constant(Schedule):
    """A value that never changes. Useful as a baseline and for evaluation."""

    c: float

    def value(self, step: int) -> float:
        return self.c


@dataclass
class LinearDecay(Schedule):
    """Falls from ``start`` to ``end`` over ``steps``, then holds.

    Linear is the right default because it is predictable: after half the decay
    window you are exactly halfway. Exponential decay spends most of its budget
    near the floor, which is usually not what you meant.
    """

    start: float
    end: float
    steps: int

    def value(self, step: int) -> float:
        if self.steps <= 0:
            return self.end
        fraction = min(1.0, step / self.steps)
        return self.start + fraction * (self.end - self.start)


@dataclass
class ExponentialDecay(Schedule):
    """Decays by a fixed ratio per step, with a floor.

    The floor is not optional. Without it epsilon reaches zero and the agent
    stops exploring entirely -- which looks fine until the environment changes,
    or until you realise it locked onto a policy it had no evidence for.
    """

    start: float
    end: float
    rate: float

    def value(self, step: int) -> float:
        return max(self.end, self.start * math.exp(-self.rate * step))
