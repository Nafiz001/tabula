"""CartPole: balance a pole on a cart by pushing it left or right.

The state is four real numbers -- position, velocity, angle, angular velocity --
so there is no finite set of states to keep a table of. The obvious workaround
is to chop each dimension into bins and use a table anyway, which
DiscretisedCartPole below does.

It is worth being accurate about how that turns out, because the tempting claim
is wrong. Discretisation does *not* fail here: six bins per dimension reaches
405 of a possible 500 steps. CartPole is only four-dimensional and its dynamics
are smooth, so coarse bins genuinely capture it. What the neural agent buys is
sample efficiency and the last stretch of performance -- 500 steps in 400
episodes against 405 in 3000.

Where binning actually breaks down is dimensionality and generalisation. The
table has bins^dims entries, so four dimensions at six bins is 1,296 states but
ten dimensions is sixty million, most never visited twice. And a table learns
nothing about a bin from its neighbours, however similar they are, while a
network shares weights across the whole space. Those are the reasons to reach
for function approximation -- not a failure to balance this particular pole.

The physics is the standard cart-pole model, integrated with semi-implicit
Euler. It is written out rather than imported so that nothing about the task is
opaque: every constant below is a physical quantity with a unit.
"""

from __future__ import annotations

import math

import numpy as np

from tabula.core.types import Action, Step


class CartPole:
    """A pole hinged to a cart that slides along a frictionless track.

    The agent pushes left or right at every step. The episode ends when the
    pole falls past 12 degrees or the cart leaves the track, and each surviving
    step is worth 1 -- so the return is simply how long the agent kept it up.
    """

    # Physical constants, in SI units.
    GRAVITY = 9.8
    CART_MASS = 1.0
    POLE_MASS = 0.1
    POLE_HALF_LENGTH = 0.5  # actual length is twice this
    FORCE = 10.0
    TIMESTEP = 0.02  # 50 Hz

    # Failure thresholds.
    ANGLE_LIMIT = 12 * math.pi / 180  # 12 degrees, in radians
    POSITION_LIMIT = 2.4  # metres from centre

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self._rng = np.random.default_rng()
        self._state = np.zeros(4, dtype=np.float64)
        self._steps = 0

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def state_dim(self) -> int:
        return 4

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # A small random perturbation, so the agent cannot memorise one
        # trajectory. Starting exactly upright every time would let a policy
        # that ignores the state entirely look competent.
        self._state = self._rng.uniform(-0.05, 0.05, size=4)
        self._steps = 0
        return self._state.copy()

    def step(self, action: Action) -> Step:
        if action not in (0, 1):
            raise ValueError(f"action must be 0 (left) or 1 (right), got {action}")

        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE if action == 1 else -self.FORCE

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        total_mass = self.CART_MASS + self.POLE_MASS
        pole_moment = self.POLE_MASS * self.POLE_HALF_LENGTH

        # The standard cart-pole equations of motion. The pole's angular
        # acceleration depends on the cart's, and vice versa, so the shared
        # term is computed once.
        temp = (force + pole_moment * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.POLE_HALF_LENGTH
            * (4.0 / 3.0 - self.POLE_MASS * cos_theta**2 / total_mass)
        )
        x_acc = temp - pole_moment * theta_acc * cos_theta / total_mass

        # Semi-implicit Euler: update velocity first, then use the *new*
        # velocity to update position. It is barely more work than the explicit
        # form and is far more stable for oscillating systems -- explicit Euler
        # steadily adds energy, so an untouched pole would swing wider forever.
        x_dot += self.TIMESTEP * x_acc
        x += self.TIMESTEP * x_dot
        theta_dot += self.TIMESTEP * theta_acc
        theta += self.TIMESTEP * theta_dot

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._steps += 1

        failed = (
            abs(x) > self.POSITION_LIMIT
            or abs(theta) > self.ANGLE_LIMIT
        )
        # Every surviving step is worth 1, including the one that fails. The
        # return is therefore the number of steps balanced, which is the
        # quantity anyone actually cares about.
        truncated = self._steps >= self.max_steps and not failed
        return Step(self._state.copy(), 1.0, terminated=failed, truncated=truncated)

    def render(self) -> str:
        """A one-line view of the cart and pole, for the CLI."""
        x, _, theta, _ = self._state
        width = 41
        centre = width // 2
        pos = int(centre + x / self.POSITION_LIMIT * centre)
        pos = max(0, min(width - 1, pos))

        track = ["-"] * width
        track[pos] = "#"
        degrees = theta * 180 / math.pi
        lean = "|" if abs(degrees) < 2 else ("/" if degrees < 0 else "\\")
        return f"{''.join(track)}  {lean} {degrees:+5.1f} deg  x={x:+5.2f}"


class DiscretisedCartPole:
    """CartPole chopped into bins, so a table can be used on it.

    This exists to be beaten. Discretisation is the obvious way to reuse
    tabular methods on a continuous problem, and trying it shows why function
    approximation is worth the trouble: the bins throw away exactly the
    distinctions the agent needs, and the number of states explodes as soon as
    you make them finer. Four dimensions at six bins each is already 1,296
    states, most of which are never visited.
    """

    def __init__(self, bins: int = 6, max_steps: int = 500) -> None:
        self.env = CartPole(max_steps=max_steps)
        self.bins = bins

        # Velocity is unbounded in principle, so the edges are chosen from what
        # the system actually reaches rather than from the failure thresholds.
        self.edges = [
            np.linspace(-2.4, 2.4, bins - 1),
            np.linspace(-3.0, 3.0, bins - 1),
            np.linspace(-CartPole.ANGLE_LIMIT, CartPole.ANGLE_LIMIT, bins - 1),
            np.linspace(-3.5, 3.5, bins - 1),
        ]

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def n_states(self) -> int:
        return self.bins**4

    def _discretise(self, state: np.ndarray) -> int:
        index = 0
        for i, value in enumerate(state):
            index = index * self.bins + int(np.digitize(value, self.edges[i]))
        return index

    def reset(self, seed: int | None = None) -> int:
        return self._discretise(self.env.reset(seed))

    def step(self, action: Action) -> Step:
        step = self.env.step(action)
        return Step(
            self._discretise(step.state),
            step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
        )
