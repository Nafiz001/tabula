"""Tests for the tabular agents and the grid environments.

Reinforcement learning is unusually easy to test badly. An agent that learns
something is not the same as an agent that learns the right thing, and a run
that improves once is not the same as one that improves reliably. So these
tests do three things: seed everything so results are reproducible, compare
against a random baseline rather than an absolute number, and check the learned
*policy* where the optimal one is known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabula.agents import ExpectedSARSA, QLearning, RandomAgent, SARSA
from tabula.core import Constant, LinearDecay, evaluate, train
from tabula.envs import CliffWalking, GridWorld, default_gridworld
from tabula.envs.gridworld import DOWN, LEFT, RIGHT, UP


# --- environments ---


def test_gridworld_movement():
    env = GridWorld(rows=3, cols=3, start=(1, 1), goal=(2, 2), step_reward=-1.0)
    env.reset()

    # Each action moves one square in its own direction.
    assert env.index_to_pos(env.step(UP).state) == (0, 1)
    assert env.index_to_pos(env.step(RIGHT).state) == (0, 2)
    assert env.index_to_pos(env.step(DOWN).state) == (1, 2)
    assert env.index_to_pos(env.step(LEFT).state) == (1, 1)


def test_gridworld_walls_and_edges_block():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), walls={(0, 1)})
    env.reset()

    # Walking into the edge leaves the agent where it was, but still costs a turn.
    step = env.step(UP)
    assert env.index_to_pos(step.state) == (0, 0)
    assert step.reward == pytest.approx(env.step_reward)

    # A wall blocks the same way an edge does.
    assert env.index_to_pos(env.step(RIGHT).state) == (0, 0)


def test_gridworld_goal_terminates():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(0, 1))
    env.reset()
    step = env.step(RIGHT)
    assert step.terminated
    assert not step.truncated
    assert step.reward == pytest.approx(env.goal_reward)


def test_step_limit_truncates_rather_than_terminates():
    """The distinction matters: truncation must not look like failure.

    A truncated state still has value -- the agent did not fail, we stopped
    watching -- and an agent that bootstraps zero from it learns that running
    out of time is as bad as dying.
    """
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), max_steps=3)
    env.reset()
    env.step(UP)  # into the wall, going nowhere
    env.step(UP)
    step = env.step(UP)

    assert step.truncated
    assert not step.terminated
    assert step.done


def test_gridworld_slip_is_random_but_seeded():
    env = GridWorld(rows=5, cols=5, slip=1.0)  # every move is replaced

    env.reset(seed=42)
    first = [env.step(RIGHT).state for _ in range(10)]
    env.reset(seed=42)
    second = [env.step(RIGHT).state for _ in range(10)]

    assert first == second, "the same seed must produce the same trajectory"


def test_gridworld_rejects_start_inside_wall():
    with pytest.raises(ValueError):
        GridWorld(rows=3, cols=3, start=(1, 1), walls={(1, 1)})


def test_cliff_sends_the_agent_home():
    env = CliffWalking()
    env.reset()
    step = env.step(RIGHT)  # straight into the cliff from the start

    assert step.reward == pytest.approx(-100.0)
    assert not step.terminated, "falling costs dearly but does not end the episode"
    assert env.index_to_pos(step.state) == env.start


# --- agents ---


def test_greedy_action_breaks_ties_randomly():
    """With an all-zero table every action ties, and argmax would always pick 0.

    That is not a cosmetic preference: an agent that always chooses action 0
    when it knows nothing walks in a straight line and explores almost nothing.
    """
    agent = QLearning(n_states=1, n_actions=4, seed=0)
    chosen = {agent.greedy_action(0) for _ in range(200)}
    assert len(chosen) > 1, "ties were broken deterministically"


def test_epsilon_zero_is_fully_greedy():
    agent = QLearning(n_states=2, n_actions=3, epsilon=Constant(0.0), seed=0)
    agent.q[0] = [0.0, 5.0, 0.0]
    assert all(agent.act(0) == 1 for _ in range(50))


def test_qlearning_update_arithmetic():
    """The update is checked by hand, because everything downstream trusts it."""
    agent = QLearning(n_states=2, n_actions=2, learning_rate=0.5, discount=0.9, seed=0)
    agent.q[1] = [3.0, 7.0]

    from tabula.core.types import Step

    # target = 1 + 0.9 * max(3, 7) = 7.3; error = 7.3 - 0 = 7.3
    # Q(0,0) <- 0 + 0.5 * 7.3 = 3.65
    agent.update(0, 0, Step(state=1, reward=1.0, terminated=False))
    assert agent.q[0, 0] == pytest.approx(3.65)


def test_terminal_state_does_not_bootstrap():
    """A terminated state has no future, so its target is the reward alone."""
    agent = QLearning(n_states=2, n_actions=2, learning_rate=1.0, discount=0.9, seed=0)
    agent.q[1] = [100.0, 100.0]

    from tabula.core.types import Step

    agent.update(0, 0, Step(state=1, reward=2.0, terminated=True))
    assert agent.q[0, 0] == pytest.approx(2.0), "a terminal state must not contribute value"


def test_sarsa_requires_the_next_action():
    """SARSA is on-policy, so it cannot update without knowing what comes next."""
    agent = SARSA(n_states=2, n_actions=2, seed=0)
    from tabula.core.types import Step

    with pytest.raises(ValueError):
        agent.update(0, 0, Step(state=1, reward=0.0, terminated=False), next_action=None)


# --- learning ---


@pytest.mark.parametrize("agent_cls", [QLearning, SARSA, ExpectedSARSA])
def test_agents_learn_the_gridworld(agent_cls):
    """Every agent should beat a random walk by a wide margin, reproducibly."""
    env = default_gridworld()
    agent = agent_cls(
        env.n_states, env.n_actions,
        learning_rate=0.2, epsilon=LinearDecay(1.0, 0.05, 3000), seed=0,
    )
    train(env, agent, episodes=2000, seed=0)

    learned = evaluate(env, agent, episodes=50, seed=1000)
    baseline = evaluate(env, RandomAgent(env.n_actions, seed=0), episodes=50, seed=1000)

    assert learned.success_rate == 1.0, f"{agent_cls.__name__} does not reach the goal reliably"
    assert learned.mean_return > baseline.mean_return + 2.0
    # The shortest path around the walls in this grid is 8 steps.
    assert learned.mean_length <= 10, f"path length {learned.mean_length} is far from optimal"


def test_qlearning_finds_the_optimal_path_length():
    """On a grid with no walls the optimal path is exactly computable."""
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), step_reward=-0.1)
    agent = QLearning(
        env.n_states, env.n_actions,
        learning_rate=0.3, epsilon=LinearDecay(1.0, 0.01, 2000), seed=0,
    )
    train(env, agent, episodes=1500, seed=0)

    result = evaluate(env, agent, episodes=20, seed=500)
    # Manhattan distance from (0,0) to (4,4) is 8 moves.
    assert result.mean_length == pytest.approx(8.0), "the agent did not find the shortest path"


def test_learning_is_reproducible():
    """The same seed must give the same table, or no result here is checkable."""

    def run():
        env = default_gridworld()
        agent = QLearning(env.n_states, env.n_actions, epsilon=LinearDecay(1.0, 0.05, 1000), seed=7)
        train(env, agent, episodes=300, seed=7)
        return agent.q.copy()

    np.testing.assert_array_equal(run(), run())


def test_training_reduces_td_error():
    """Convergence, seen from the inside: the corrections get smaller."""
    env = default_gridworld()
    agent = QLearning(
        env.n_states, env.n_actions,
        learning_rate=0.2, epsilon=LinearDecay(1.0, 0.05, 2000), seed=0,
    )
    history = train(env, agent, episodes=2000, seed=0)

    early = float(np.mean(history.td_errors[:200]))
    late = float(np.mean(history.td_errors[-200:]))
    assert late < early * 0.5, f"temporal-difference error barely moved: {early:.3f} -> {late:.3f}"


def test_optimistic_initialisation_explores_without_epsilon():
    """Optimism is a real alternative to random exploration, not a trick.

    With every unvisited action looking better than any visited one, a purely
    greedy agent still sweeps the state space -- so this runs with epsilon
    pinned to zero, which would otherwise leave it stuck on its first choice.
    """
    env = default_gridworld()
    agent = QLearning(
        env.n_states, env.n_actions,
        learning_rate=0.2, epsilon=Constant(0.0), optimistic_init=1.0, seed=0,
    )
    train(env, agent, episodes=1500, seed=0)

    result = evaluate(env, agent, episodes=20, seed=1000)
    assert result.success_rate == 1.0
    visited = int((agent.visits > 0).sum())
    assert visited > env.n_states, "optimism did not drive exploration"


# --- the cliff: where the two algorithms disagree ---


def _train_on_cliff(agent_cls, episodes=3000):
    env = CliffWalking()
    agent = agent_cls(
        env.n_states, env.n_actions,
        learning_rate=0.5, discount=1.0,
        epsilon=LinearDecay(1.0, 0.1, 2000), seed=0,
    )
    history = train(env, agent, episodes=episodes, seed=0)
    return env, agent, history


def test_qlearning_learns_the_optimal_cliff_path():
    """Off-policy learning evaluates greedy play, so it takes the short route.

    The optimal path is 13 steps at -1 each: along the row above the cliff and
    down at the end.
    """
    env, agent, _ = _train_on_cliff(QLearning)
    result = evaluate(env, agent, episodes=10, seed=99)
    assert result.mean_return == pytest.approx(-13.0), (
        f"Q-learning should find the 13-step optimal path, got {result.mean_return}"
    )


def test_sarsa_learns_a_safer_cliff_path_than_qlearning():
    """The textbook result, and the reason both algorithms are here.

    SARSA is on-policy: it learns the value of the epsilon-greedy behaviour it
    is actually following, so it prices in the chance that a random step pushes
    it off the cliff and keeps its distance. Q-learning learns the value of
    perfect play and walks the edge.

    So SARSA's *greedy* policy is longer -- worse, if you never explore -- while
    its *online* return during training is better, because it falls off less.
    Neither is a bug. They answer different questions.
    """
    _, q_agent, q_history = _train_on_cliff(QLearning)
    _, s_agent, s_history = _train_on_cliff(SARSA)

    cliff = CliffWalking()
    q_result = evaluate(cliff, q_agent, episodes=10, seed=99)
    s_result = evaluate(cliff, s_agent, episodes=10, seed=99)

    # Greedy: Q-learning's path is shorter, so its return is higher.
    assert q_result.mean_return > s_result.mean_return, (
        "Q-learning should find the shorter greedy path"
    )

    # Online, while still exploring: SARSA does better, because it falls less.
    q_online = float(np.mean(q_history.returns[-500:]))
    s_online = float(np.mean(s_history.returns[-500:]))
    assert s_online > q_online, (
        f"SARSA should suffer less while exploring: SARSA {s_online:.1f} vs Q {q_online:.1f}"
    )


def test_sarsa_keeps_away_from_the_cliff_edge():
    """Checked structurally, not just by score: where does the policy actually go?

    Starting from the corner, SARSA's first move should be up and away, rather
    than right along the edge.
    """
    env, agent, _ = _train_on_cliff(SARSA)
    start_state = env.state_index(env.start)
    assert agent.greedy_policy()[start_state] == UP, (
        "SARSA should step away from the cliff rather than along it"
    )
