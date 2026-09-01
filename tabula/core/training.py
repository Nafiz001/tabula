"""The training loop, and the measurement that says whether it worked.

Everything here is deliberately boring. The loop is short because the
interesting decisions live in the agents; what this file has to get right is
the bookkeeping -- which is where results quietly become wrong. Reporting the
return of the exploring policy, for instance, understates a good agent, because
it is still taking random actions on purpose. Evaluation therefore always runs
greedily, and separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingHistory:
    """What happened during training, kept for plotting and for assertions."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    # td_errors is the average absolute temporal-difference error per episode:
    # the size of the correction the agent is still making. It falling toward
    # zero is what "converged" actually looks like from the inside.
    td_errors: list[float] = field(default_factory=list)

    def moving_average(self, window: int = 100) -> np.ndarray:
        """Smoothed returns.

        Raw episode returns in reinforcement learning are extremely noisy --
        a single unlucky exploratory step can dominate an episode -- so an
        unsmoothed curve says almost nothing about whether learning is
        happening.
        """
        if not self.returns:
            return np.array([])
        arr = np.asarray(self.returns, dtype=np.float64)
        if len(arr) < window:
            return np.array([arr.mean()])
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    def summary(self, last: int = 100) -> dict[str, float]:
        if not self.returns:
            return {}
        recent = np.asarray(self.returns[-last:], dtype=np.float64)
        return {
            "episodes": float(len(self.returns)),
            "mean_return_last": float(recent.mean()),
            "std_return_last": float(recent.std()),
            "mean_length_last": float(np.mean(self.lengths[-last:])),
            "final_epsilon": float(self.epsilons[-1]) if self.epsilons else 0.0,
        }


def train(
    env,
    agent,
    episodes: int = 1000,
    max_steps: int = 1000,
    seed: int | None = None,
    on_episode=None,
) -> TrainingHistory:
    """Run the agent in the environment, letting it learn as it goes.

    The SARSA branch is the only untidy part, and it is unavoidable: an
    on-policy agent has to know the action it will take next before it can
    update the current one, so the loop has to choose the next action early and
    carry it forward.
    """
    history = TrainingHistory()
    # The loop asks the agent what it needs rather than checking its class. An
    # on-policy agent has to know the action it will take next before it can
    # update the current one, so the loop chooses that action early and carries
    # it forward -- but which agents need that is the agent's business.
    on_policy = getattr(agent, "on_policy", False)

    for episode in range(episodes):
        # Seeding per episode rather than once keeps runs reproducible while
        # still giving every episode a different start when the environment
        # randomises.
        state = env.reset(seed=None if seed is None else seed + episode)
        action = agent.act(state) if on_policy else None

        total_reward = 0.0
        errors: list[float] = []

        for _ in range(max_steps):
            if not on_policy:
                action = agent.act(state)

            step = env.step(action)
            total_reward += step.reward

            next_action = None
            if on_policy and not step.terminated:
                next_action = agent.act(step.state)

            error = agent.update(state, action, step, next_action)
            errors.append(abs(error))

            state = step.state
            if on_policy:
                action = next_action

            if step.done:
                break

        history.returns.append(total_reward)
        history.lengths.append(len(errors))
        history.epsilons.append(agent.current_epsilon())
        history.td_errors.append(float(np.mean(errors)) if errors else 0.0)

        if on_episode is not None:
            on_episode(episode, history)

    return history


@dataclass
class EvalResult:
    """What a greedy evaluation found."""

    mean_return: float
    std_return: float
    mean_length: float
    success_rate: float
    episodes: int

    def __str__(self) -> str:
        return (
            f"return {self.mean_return:+.2f} +/- {self.std_return:.2f}  "
            f"length {self.mean_length:.1f}  success {self.success_rate:.0%}"
        )


def evaluate(
    env,
    agent,
    episodes: int = 100,
    max_steps: int = 1000,
    seed: int | None = None,
    success_reward: float | None = None,
) -> EvalResult:
    """Measure the agent's actual policy, with exploration switched off.

    This is the number that means something. During training the agent is
    deliberately acting randomly some of the time, so its training return is a
    measure of the exploring policy, not of what it has learned. Reporting the
    training curve as the result is the most common way to overstate an agent.
    """
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0

    for episode in range(episodes):
        state = env.reset(seed=None if seed is None else seed + episode)
        total = 0.0
        steps = 0

        for _ in range(max_steps):
            action = agent.act(state, greedy=True)
            step = env.step(action)
            total += step.reward
            steps += 1
            state = step.state
            if step.done:
                # "Success" means the episode ended on purpose, not by running
                # out of time -- unless the caller defines it by reward.
                if success_reward is not None:
                    successes += int(total >= success_reward)
                else:
                    successes += int(step.terminated)
                break

        returns.append(total)
        lengths.append(steps)

    arr = np.asarray(returns, dtype=np.float64)
    return EvalResult(
        mean_return=float(arr.mean()),
        std_return=float(arr.std()),
        mean_length=float(np.mean(lengths)),
        success_rate=successes / episodes,
        episodes=episodes,
    )
