"""Tests for the replay buffer, DQN, MCTS and the AlphaZero loop.

Deep reinforcement learning is slow to test properly, so these are split. The
mechanical parts -- the buffer, the search, the shapes and the sign conventions
-- are checked exactly and cheaply. The learning claims are checked with short
runs that assert improvement over a baseline, and the full convergence runs are
marked slow so a normal test run stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabula.agents import DQN, AlphaZeroAgent, RawNetworkPlayer
from tabula.agents.alphazero import encode
from tabula.agents.selfplay import evaluate_against_perfect, policy_agreement
from tabula.core import LinearDecay, MCTS, ReplayBuffer, evaluate, train
from tabula.envs import CartPole, DiscretisedCartPole, TicTacToe, TicTacToeSolver
from tabula.envs.tictactoe import RandomPlayer


# --- replay buffer ---


def test_replay_buffer_stores_and_samples():
    buf = ReplayBuffer(capacity=100, state_dim=4, seed=0)
    assert len(buf) == 0

    for i in range(10):
        buf.add(np.full(4, i, dtype=np.float32), i % 2, float(i), np.full(4, i + 1), i == 9)
    assert len(buf) == 10

    batch = buf.sample(5)
    assert len(batch) == 5
    assert batch.states.shape == (5, 4)
    assert batch.actions.shape == (5,)


def test_replay_buffer_overwrites_oldest():
    """A ring buffer holds a sliding window, not a growing history."""
    buf = ReplayBuffer(capacity=3, state_dim=1, seed=0)
    for i in range(5):
        buf.add(np.array([i], dtype=np.float32), 0, 0.0, np.array([0.0]), False)

    assert len(buf) == 3
    assert buf.is_full
    # The first two should have been overwritten by the last two.
    stored = sorted(buf.states.flatten().tolist())
    assert stored == [2.0, 3.0, 4.0]


def test_replay_buffer_records_terminated_not_done():
    """The distinction has to survive into the buffer, or the target is wrong.

    Zeroing the bootstrap for a truncated state teaches the agent that hitting
    the step limit is a failure.
    """
    buf = ReplayBuffer(capacity=4, state_dim=1, seed=0)
    buf.add(np.array([0.0]), 0, 1.0, np.array([1.0]), terminated=False)
    buf.add(np.array([1.0]), 0, 1.0, np.array([2.0]), terminated=True)

    assert buf.terminated[0] == 0.0
    assert buf.terminated[1] == 1.0


def test_empty_buffer_cannot_be_sampled():
    buf = ReplayBuffer(capacity=10, state_dim=2, seed=0)
    with pytest.raises(ValueError):
        buf.sample(1)


# --- CartPole physics ---


def test_cartpole_pole_falls_when_pushed_one_way():
    """A physics check that does not depend on any agent.

    Pushing consistently in one direction must eventually topple the pole; if
    it does not, the dynamics are wrong and no learning result would mean
    anything.
    """
    env = CartPole()
    env.reset(seed=0)
    for _ in range(500):
        step = env.step(1)
        if step.terminated:
            break
    assert step.terminated, "the pole never fell despite constant one-way pushes"


def test_cartpole_rejects_invalid_actions():
    env = CartPole()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(2)


def test_cartpole_is_deterministic_given_a_seed():
    env = CartPole()
    env.reset(seed=42)
    first = [env.step(i % 2).state.copy() for i in range(20)]
    env.reset(seed=42)
    second = [env.step(i % 2).state.copy() for i in range(20)]
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b)


def test_cartpole_truncates_at_the_step_limit_without_terminating():
    """Surviving to the limit is a success, and must not look like a failure."""
    env = CartPole(max_steps=5)
    env.reset(seed=0)
    # The pole barely moves in five steps from near-upright, so this ends by
    # truncation rather than by falling.
    for _ in range(5):
        step = env.step(0)
    assert step.truncated
    assert not step.terminated


def test_discretisation_produces_valid_state_indices():
    env = DiscretisedCartPole(bins=6)
    state = env.reset(seed=0)
    assert 0 <= state < env.n_states
    for _ in range(50):
        step = env.step(int(np.random.default_rng(0).integers(2)))
        assert 0 <= step.state < env.n_states
        if step.done:
            break


# --- DQN ---


def test_dqn_produces_valid_actions_and_shapes():
    agent = DQN(state_dim=4, n_actions=2, seed=0)
    state = np.zeros(4, dtype=np.float32)

    assert agent.act(state) in (0, 1)
    assert agent.q_values(state).shape == (2,)


def test_dqn_target_network_starts_identical_and_lags_behind():
    """The target network is the fix for the moving-target problem.

    It has to start as an exact copy, and it must not follow the live network
    on every step -- if it did, it would not be a fixed target at all.
    """
    agent = DQN(state_dim=4, n_actions=2, seed=0, learning_starts=1, target_sync_every=10_000)

    for p, t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p, t), "the target network did not start as a copy"

    from tabula.core.types import Step

    for i in range(200):
        agent.update(np.random.randn(4).astype(np.float32), i % 2,
                     Step(np.random.randn(4).astype(np.float32), 1.0, terminated=False))

    diverged = any(
        not torch.allclose(p, t)
        for p, t in zip(agent.q_net.parameters(), agent.target_net.parameters())
    )
    assert diverged, "the target network tracked the live one, so it is not a fixed target"


def test_dqn_target_syncs_when_it_should():
    agent = DQN(state_dim=4, n_actions=2, seed=0, learning_starts=1, target_sync_every=5)
    from tabula.core.types import Step

    for i in range(60):
        agent.update(np.random.randn(4).astype(np.float32), i % 2,
                     Step(np.random.randn(4).astype(np.float32), 1.0, terminated=False))

    for p, t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p, t), "a sync should have made the networks identical again"


def test_dqn_does_not_train_before_the_buffer_fills():
    """Training on the first few correlated transitions is what the wait avoids."""
    agent = DQN(state_dim=4, n_actions=2, seed=0, learning_starts=100)
    from tabula.core.types import Step

    for i in range(50):
        loss = agent.update(np.zeros(4, dtype=np.float32), 0,
                            Step(np.zeros(4, dtype=np.float32), 1.0, terminated=False))
        assert loss == 0.0
    assert agent.updates == 0


def test_dqn_learns_cartpole():
    """The claim: a network that starts random learns to balance the pole.

    Kept short enough to run in a normal test pass; the full 500-step result is
    in the slow test below.
    """
    env = CartPole()
    agent = DQN(env.state_dim, env.n_actions, learning_rate=1e-3,
                epsilon=LinearDecay(1.0, 0.02, 5_000), seed=0)
    train(env, agent, episodes=150, seed=0)

    result = evaluate(env, agent, episodes=10, seed=500)
    # A random policy manages about 20 steps.
    assert result.mean_return > 80, f"only balanced for {result.mean_return:.0f} steps"


@pytest.mark.slow
def test_dqn_solves_cartpole():
    """The full result: the agent reaches the 500-step ceiling."""
    env = CartPole()
    agent = DQN(env.state_dim, env.n_actions, learning_rate=1e-3,
                target_sync_every=200, epsilon=LinearDecay(1.0, 0.02, 10_000), seed=0)
    train(env, agent, episodes=400, seed=0)

    result = evaluate(env, agent, episodes=20, seed=500)
    assert result.mean_return > 450, f"balanced for {result.mean_return:.0f} of 500 steps"


# --- MCTS ---


def test_mcts_finds_an_immediate_win_with_a_blind_evaluator():
    """Search alone should see a win in one, with no network guidance at all.

    The evaluator here returns a uniform prior and a value of zero, so anything
    the search finds it found by looking ahead.
    """

    def blind(game):
        return np.ones(9) / 9, 0.0

    search = MCTS(blind, simulations=200, seed=0)
    game = TicTacToe()
    game.reset()
    #  X X .        X to move; square 2 wins
    #  O O .
    #  . . .
    game.board[:] = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    game._player = 1

    policy = search.search(game)
    assert int(np.argmax(policy)) == 2, f"search missed a win in one: {policy}"


def test_mcts_blocks_an_immediate_threat():
    def blind(game):
        return np.ones(9) / 9, 0.0

    search = MCTS(blind, simulations=300, seed=0)
    game = TicTacToe()
    game.reset()
    #  X X .        O to move and must block square 2
    #  O . .
    #  . . .
    game.board[:] = [1, 1, 0, -1, 0, 0, 0, 0, 0]
    game._player = -1

    policy = search.search(game)
    assert int(np.argmax(policy)) == 2, f"search failed to block: {policy}"


def test_mcts_never_suggests_an_illegal_move():
    def blind(game):
        # Deliberately hostile: all the prior mass sits on an occupied square.
        priors = np.zeros(9)
        priors[4] = 1.0
        return priors, 0.0

    search = MCTS(blind, simulations=50, seed=0)
    game = TicTacToe()
    game.reset()
    game.step(4)  # the centre is now taken

    policy = search.search(game)
    assert policy[4] == 0.0, "search put weight on an occupied square"
    assert policy.sum() == pytest.approx(1.0)


def test_mcts_visit_counts_form_a_distribution():
    def blind(game):
        return np.ones(9) / 9, 0.0

    search = MCTS(blind, simulations=100, seed=0)
    game = TicTacToe()
    game.reset()

    policy = search.search(game)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()


def test_more_simulations_do_not_make_search_worse():
    """A sanity check on the backup signs.

    A sign error in the negation makes search actively choose losing moves, and
    more simulations then make it worse rather than better -- which is the
    clearest symptom of that bug.
    """

    def blind(game):
        return np.ones(9) / 9, 0.0

    game = TicTacToe()
    game.reset()
    #  X X .        a win in one is available at square 2
    #  O O .
    #  . . .
    game.board[:] = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    game._player = 1

    weak = MCTS(blind, simulations=30, seed=0).search(game.clone())
    strong = MCTS(blind, simulations=400, seed=0).search(game.clone())
    assert strong[2] >= weak[2], "more search made the winning move less attractive"


# --- AlphaZero ---


def test_encoding_shape_and_perspective():
    game = TicTacToe()
    game.reset()
    encoded = encode(game)
    assert encoded.shape == (18,)
    assert encoded.sum() == 0, "an empty board should encode as all zeros"

    game.step(0)  # X takes a corner; now O to move
    encoded = encode(game)
    # From O's view: O has nothing, X's corner is in the opponent plane.
    assert encoded[0] == 0.0
    assert encoded[9] == 1.0


def test_alphazero_network_output_shapes():
    agent = AlphaZeroAgent(simulations=5, seed=0)
    game = TicTacToe()
    game.reset()

    priors, value = agent._evaluate(game)
    assert priors.shape == (9,)
    assert priors.sum() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0, "the value head must be bounded by its tanh"


def test_alphazero_plays_only_legal_moves_untrained():
    """Even before training, the search must respect the rules."""
    agent = AlphaZeroAgent(simulations=10, seed=0)
    game = TicTacToe()

    for _ in range(20):
        game.reset()
        while not game.is_terminal():
            move = agent.greedy_move(game)
            assert move in game.legal_actions()
            game.step(move)


def test_alphazero_self_play_produces_consistent_examples():
    """Each example must carry the outcome from its own mover's perspective."""
    agent = AlphaZeroAgent(simulations=10, seed=0)
    examples = agent.self_play_game(seed=0)

    assert len(examples) > 0
    for state, policy, value in examples:
        assert state.shape == (18,)
        assert policy.shape == (9,)
        assert policy.sum() == pytest.approx(1.0)
        assert value in (-1.0, 0.0, 1.0)

    # Consecutive positions belong to opposing players, so in a decisive game
    # their values must alternate rather than repeat.
    values = [v for _, _, v in examples]
    if any(v != 0 for v in values):
        for a, b in zip(values, values[1:]):
            assert a == -b, f"values did not alternate between players: {values}"


def test_alphazero_training_reduces_loss():
    agent = AlphaZeroAgent(simulations=15, seed=0)
    log = agent.train(iterations=6, games_per_iteration=6, epochs=2, seed=0)

    assert log[-1]["value_loss"] < log[0]["value_loss"], "the value head did not improve"
    assert log[-1]["replay"] > log[0]["replay"], "the replay window did not grow"


def test_alphazero_improves_over_an_untrained_network():
    """A short run has to move the needle, or the loop is not learning.

    The raw network is compared rather than the searched agent, because search
    masks a weak network -- the point here is what the *network* learned.
    """
    solver = TicTacToeSolver()
    agent = AlphaZeroAgent(simulations=25, seed=0)

    before = policy_agreement(RawNetworkPlayer(agent), solver, games=100, seed=2)
    agent.train(iterations=12, games_per_iteration=12, epochs=3, seed=0)
    after = policy_agreement(RawNetworkPlayer(agent), solver, games=100, seed=2)

    assert after > before + 0.05, (
        f"the network barely improved: {before:.1%} -> {after:.1%}"
    )


@pytest.mark.slow
def test_alphazero_reaches_perfect_play():
    """The full claim: self-play from zero reaches an unbeatable policy.

    Slow because it is a real training run -- roughly three minutes on a CPU.
    """
    solver = TicTacToeSolver()
    agent = AlphaZeroAgent(simulations=60, learning_rate=1e-3, seed=0)
    agent.train(iterations=36, games_per_iteration=25, epochs=3, seed=0)

    result = evaluate_against_perfect(agent, solver, games=100, seed=7)
    assert result["losses"] == 0, (
        f"lost {result['losses']} of 100 games to a perfect opponent"
    )


def test_random_baseline_for_context():
    """Without this number, none of the above means anything."""
    solver = TicTacToeSolver()
    agreement = policy_agreement(RandomPlayer(seed=0), solver, games=200, seed=2)
    # Random play agrees with an optimal move fairly often by luck, because
    # many positions have several optimal moves. That is exactly why the
    # threshold for a trained agent has to be well above it.
    assert 0.4 < agreement < 0.75
