"""Tabular agents: one number per state-action pair, learned from experience.

These are built before the neural versions on purpose. A table has no
approximation error, no optimiser, and no hyperparameters beyond the learning
rate and the discount -- so when a tabular agent fails to learn, the problem is
the algorithm or the environment, never the network. Every bug found here is a
bug that would have been ten times harder to find in a deep agent.

Both agents below differ in exactly one line, and that line changes what they
learn. It is worth reading them side by side.
"""

from __future__ import annotations

import numpy as np

from tabula.core.schedules import Constant, Schedule
from tabula.core.types import Action, State, Step


class TabularAgent:
    """Shared machinery: a Q-table, epsilon-greedy action selection, bookkeeping."""

    # on_policy tells the training loop whether this agent needs to know the
    # action it will take next. Declaring it here keeps the loop from having to
    # know which concrete agents exist.
    on_policy: bool = False

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: Schedule | float = 0.1,
        seed: int | None = None,
        optimistic_init: float = 0.0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        # The discount decides how far ahead the agent cares. At 0 it is purely
        # greedy about the next reward; at 1 a distant reward counts as much as
        # an immediate one, which stops values converging in a task that never
        # ends. 0.99 is the usual compromise.
        self.gamma = discount
        self.epsilon = epsilon if isinstance(epsilon, Schedule) else Constant(epsilon)

        # Initialising above the true values makes the agent optimistic: an
        # unvisited action looks better than a visited one, so it explores
        # systematically rather than at random. It is a genuinely different
        # exploration strategy from epsilon-greedy, and often a better one.
        self.q = np.full((n_states, n_actions), optimistic_init, dtype=np.float64)

        self.rng = np.random.default_rng(seed)
        self.steps = 0
        # visits backs the diagnostics: an agent that never visited a state has
        # not learned about it, however confident its table looks.
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)

    def current_epsilon(self) -> float:
        return self.epsilon.value(self.steps)

    def act(self, state: State, greedy: bool = False) -> Action:
        """Choose an action: epsilon-greedy while training, greedy to evaluate."""
        if not greedy and self.rng.random() < self.current_epsilon():
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(int(state))

    def greedy_action(self, state: int) -> Action:
        """The best action, with ties broken at random.

        Breaking ties randomly matters more than it looks. ``argmax`` always
        returns the lowest index, so an agent whose table is still all zeros
        would walk in one direction forever and explore nothing -- the very
        first episodes would be wasted.
        """
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def greedy_policy(self) -> np.ndarray:
        """The best action in every state, for rendering and comparison."""
        return np.argmax(self.q, axis=1)

    def state_values(self) -> np.ndarray:
        """V(s) = max_a Q(s, a): how good each state is under greedy play."""
        return np.max(self.q, axis=1)

    def update(self, state: State, action: Action, step: Step, next_action: Action | None = None) -> float:
        raise NotImplementedError


class QLearning(TabularAgent):
    """Q-learning: off-policy temporal-difference control.

    The update uses the *best* next action, whatever the agent actually does
    next:

        Q(s,a) <- Q(s,a) + lr * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    That ``max`` is what makes it off-policy. The agent can wander around
    exploring, and still learn the value of playing perfectly, because the
    target never mentions the exploratory action it is about to take. It is why
    Q-learning converges to the optimal policy regardless of how badly it
    behaves along the way -- and why, on the cliff, it learns the dangerous
    shortest path: it evaluates the route as if it will never slip.
    """

    def update(self, state: State, action: Action, step: Step, next_action: Action | None = None) -> float:
        s, a = int(state), int(action)
        self.visits[s, a] += 1
        self.steps += 1

        # A terminated state has no future, so its bootstrap is zero. A
        # truncated one does -- the episode was cut short, not lost -- so its
        # value still counts. Treating them alike would teach the agent that
        # running out of time is a failure.
        if step.terminated:
            target = step.reward
        else:
            target = step.reward + self.gamma * float(np.max(self.q[int(step.state)]))

        error = target - self.q[s, a]
        self.q[s, a] += self.lr * error
        return float(error)


class SARSA(TabularAgent):
    """SARSA: on-policy temporal-difference control.

    One symbol different from Q-learning, and a different answer:

        Q(s,a) <- Q(s,a) + lr * [ r + gamma * Q(s',a') - Q(s,a) ]

    Where Q-learning takes the max, SARSA takes the value of the action it is
    *actually going to take* -- which, being epsilon-greedy, is sometimes
    random. So SARSA learns the value of its own imperfect behaviour, and on the
    cliff it walks a row further from the edge: it has priced in the chance of
    an exploratory step pushing it off.

    The name is the tuple it needs: state, action, reward, next state, next
    action.
    """

    on_policy = True

    def update(self, state: State, action: Action, step: Step, next_action: Action | None = None) -> float:
        if next_action is None and not step.terminated:
            raise ValueError("SARSA needs the next action, since it is on-policy")

        s, a = int(state), int(action)
        self.visits[s, a] += 1
        self.steps += 1

        if step.terminated:
            target = step.reward
        else:
            target = step.reward + self.gamma * self.q[int(step.state), int(next_action)]

        error = target - self.q[s, a]
        self.q[s, a] += self.lr * error
        return float(error)


class ExpectedSARSA(TabularAgent):
    """Expected SARSA: on-policy, but averaged over the policy instead of sampled.

    Rather than using the action it happens to draw, it uses the expected value
    under its own epsilon-greedy policy:

        Q(s,a) <- Q(s,a) + lr * [ r + gamma * E_a'[Q(s',a')] - Q(s,a) ]

    That removes the variance SARSA gets from sampling the next action, usually
    at no cost beyond a few more multiplications. It sits between the two: safer
    than Q-learning, steadier than SARSA.

    It is not on-policy in the loop's sense: it averages over the policy rather
    than sampling from it, so it never needs to be told the next action.
    """

    def update(self, state: State, action: Action, step: Step, next_action: Action | None = None) -> float:
        s, a = int(state), int(action)
        self.visits[s, a] += 1
        self.steps += 1

        if step.terminated:
            target = step.reward
        else:
            next_q = self.q[int(step.state)]
            eps = self.current_epsilon()
            # Every action has probability eps/n from exploring; the greedy one
            # additionally has (1 - eps).
            probs = np.full(self.n_actions, eps / self.n_actions)
            probs[int(np.argmax(next_q))] += 1.0 - eps
            target = step.reward + self.gamma * float(np.dot(probs, next_q))

        error = target - self.q[s, a]
        self.q[s, a] += self.lr * error
        return float(error)


class RandomAgent:
    """Acts uniformly at random. The baseline every result is measured against.

    Without it, "the agent reached the goal" means nothing -- a random walk
    reaches the goal too, eventually. The interesting number is always the gap.
    """

    on_policy = False

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def act(self, state: State, greedy: bool = False) -> Action:
        return int(self.rng.integers(self.n_actions))

    def update(self, state: State, action: Action, step: Step, next_action: Action | None = None) -> float:
        return 0.0

    def current_epsilon(self) -> float:
        return 1.0
