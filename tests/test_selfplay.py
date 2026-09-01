"""Tests for tic-tac-toe, the exhaustive solver, and self-play learning.

This file is where the project's central claim gets checked. Everywhere else an
agent can only be compared against a baseline; here the game is small enough to
solve outright, so "the agent learned" becomes a number: how many games it loses
to an opponent that provably cannot be beaten, and how often its moves agree
with ones minimax certifies as optimal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabula.agents.selfplay import (
    SelfPlayQLearning,
    evaluate_against_perfect,
    play_match,
    policy_agreement,
)
from tabula.core import LinearDecay
from tabula.envs.tictactoe import (
    PerfectPlayer,
    RandomPlayer,
    TicTacToe,
    TicTacToeSolver,
)


# --- the game ---


def test_win_detection_on_every_line():
    """All eight winning lines, checked individually."""
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for line in lines:
        game = TicTacToe()
        game.reset()
        game.board[list(line)] = 1
        assert game.winner() == 1, f"line {line} was not recognised as a win"
        assert game.is_terminal()


def test_full_board_without_a_line_is_a_draw():
    game = TicTacToe()
    game.reset()
    #  X O X
    #  X O O
    #  O X X
    game.board[:] = [1, -1, 1, 1, -1, -1, -1, 1, 1]
    assert game.winner() == 0
    assert game.is_terminal()


def test_playing_an_occupied_square_is_rejected():
    game = TicTacToe()
    game.reset()
    game.step(4)
    with pytest.raises(ValueError):
        game.step(4)


def test_players_alternate_and_reward_is_from_the_movers_view():
    game = TicTacToe()
    game.reset()
    assert game.current_player == 1
    step = game.step(0)
    assert game.current_player == -1
    assert step.reward == 0.0

    # Set up X to win with one more move.
    game.reset()
    game.board[:] = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    game._player = 1
    step = game.step(2)
    assert step.terminated
    assert step.reward == 1.0, "the reward must be positive for the player who won"


def test_canonical_board_flips_with_the_player():
    """One network plays both sides only if the board is normalised for it."""
    game = TicTacToe()
    game.reset()
    game.step(0)  # X takes a corner; it is now O to move

    canonical = game.canonical()
    # From O's perspective, O's own pieces are +1 -- and O has none yet, while
    # X's corner shows as -1.
    assert canonical[0] == -1
    assert game.current_player == -1


def test_clone_is_independent():
    """Search explores on a copy, so the copy must not touch the real game."""
    game = TicTacToe()
    game.reset()
    game.step(4)

    copy = game.clone()
    copy.step(0)
    assert game.board[0] == 0, "mutating a clone changed the original"
    assert copy.board[0] == -1


# --- the solver: this is the ground truth everything else is graded against ---


def test_tictactoe_is_a_draw_with_perfect_play():
    """The known result, and a check that the search itself is right.

    If this were wrong, every measurement in this file would be measuring
    against a broken oracle.
    """
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    assert solver.value(game) == 0


def test_every_opening_move_draws():
    """Also known: no first move wins or loses tic-tac-toe."""
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    assert sorted(solver.best_actions(game)) == list(range(9))


def test_only_the_centre_answers_a_corner_opening():
    """A concrete tactical fact, so the solver is checked on more than totals.

    Against a corner opening, every reply except the centre loses.
    """
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    game.step(0)  # X takes the top-left corner

    assert solver.best_actions(game) == [4], "the centre is the only drawing reply"

    # And confirming the other side of it: an edge reply loses.
    game.step(1)
    assert solver.value(game) == 1, "X should be winning after O replies on an edge"


def test_solver_finds_an_immediate_win():
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    #  X X .        X to move, so square 2 wins immediately
    #  O O .
    #  . . .
    game.board[:] = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    game._player = 1
    assert solver.value(game) == 1
    assert 2 in solver.best_actions(game)


def test_solver_reports_every_move_as_optimal_when_all_of_them_lose():
    """A subtlety worth pinning down, because it looks like a bug.

    In this position O appears obliged to block square 2 -- but O is already
    lost. Blocking loses to X taking the centre, which creates two threats at
    once, and every other move loses immediately. Since every move has the same
    value, minimax is indifferent and reports all of them as optimal.

    That is correct, and it is also a real limitation of plain minimax: it will
    not prefer to lose slowly, or to pick the line that gives a fallible
    opponent the most chances to go wrong.
    """
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    #  X X .        O to move, and already lost
    #  O . .
    #  . . .
    game.board[:] = [1, 1, 0, -1, 0, 0, 0, 0, 0]
    game._player = -1

    assert solver.value(game) == -1, "O should be losing here"
    assert sorted(solver.best_actions(game)) == sorted(game.legal_actions().tolist())


def test_solver_finds_the_only_saving_move():
    """Where a position *is* still drawable, the forced move must be unique.

    Against a corner opening the centre is the only reply that draws; every
    other loses. This is the sharpest single fact in the game, and it is the
    strongest check that the search is right.
    """
    solver = TicTacToeSolver()
    game = TicTacToe()
    game.reset()
    game.step(0)  # X takes a corner

    assert solver.value(game) == 0, "O can still draw here"
    assert solver.best_actions(game) == [4], "only the centre saves the game"


def test_a_perfect_player_never_loses():
    """Two perfect players draw every time, from both sides."""
    solver = TicTacToeSolver()
    a = PerfectPlayer(solver, seed=0)
    b = PerfectPlayer(solver, seed=1)

    result = play_match(a, b, games=50, seed=0)
    assert result["x_wins"] == 0
    assert result["o_wins"] == 0
    assert result["draws"] == 50


def test_perfect_play_beats_random_and_never_loses_to_it():
    solver = TicTacToeSolver()
    perfect = PerfectPlayer(solver, seed=0)
    random_player = RandomPlayer(seed=1)

    as_x = play_match(perfect, random_player, games=100, seed=0)
    assert as_x["o_wins"] == 0, "a perfect player lost to a random one"
    assert as_x["x_wins"] > 80, "perfect play should punish random play most of the time"

    as_o = play_match(random_player, perfect, games=100, seed=1)
    assert as_o["x_wins"] == 0, "a perfect player lost to a random one as O"


# --- self-play learning ---


def test_random_play_loses_heavily_to_perfect_play():
    """The baseline, so the trained numbers below mean something."""
    solver = TicTacToeSolver()
    result = evaluate_against_perfect(RandomPlayer(seed=0), solver, games=100, seed=1)
    assert result["losses"] > 60, "a random player should lose most games to perfect play"
    assert result["wins"] == 0, "a perfect player cannot be beaten"


def test_self_play_reaches_perfect_play():
    """The headline claim, verified rather than asserted.

    The agent is given the rules and nothing else -- no examples, no opponent,
    no heuristics -- and plays itself. Afterwards it must not lose a single game
    to an opponent that is provably unbeatable.
    """
    solver = TicTacToeSolver()
    agent = SelfPlayQLearning(
        learning_rate=0.3, epsilon=LinearDecay(0.9, 0.05, 40_000), seed=0
    )
    agent.train(episodes=50_000, seed=0)

    result = evaluate_against_perfect(agent, solver, games=200, seed=1)
    assert result["losses"] == 0, (
        f"the agent lost {result['losses']} of 200 games to a perfect opponent"
    )
    assert result["draws"] == 200


def test_self_play_agrees_with_optimal_play():
    """A finer instrument than the win rate.

    An agent can draw every game against a perfect opponent while still playing
    loose moves in positions that opponent never steers into. This walks a
    random sample of the game tree and grades each choice against the solver.
    """
    solver = TicTacToeSolver()
    agent = SelfPlayQLearning(
        learning_rate=0.3, epsilon=LinearDecay(0.9, 0.05, 40_000), seed=0
    )
    agent.train(episodes=50_000, seed=0)

    agreement = policy_agreement(agent, solver, games=300, seed=2)
    baseline = policy_agreement(RandomPlayer(seed=0), solver, games=300, seed=2)

    assert agreement > 0.9, f"only {agreement:.1%} of moves were optimal"
    assert agreement > baseline + 0.25


def test_self_play_improves_monotonically_with_experience():
    """More games should mean fewer losses. A flat curve would mean no learning."""
    solver = TicTacToeSolver()
    losses = []
    for episodes in (2_000, 10_000, 40_000):
        agent = SelfPlayQLearning(
            learning_rate=0.3, epsilon=LinearDecay(0.9, 0.05, episodes * 0.8), seed=0
        )
        agent.train(episodes=episodes, seed=0)
        losses.append(evaluate_against_perfect(agent, solver, games=100, seed=1)["losses"])

    assert losses[0] > losses[-1], f"no improvement across training: {losses}"
    assert losses[-1] == 0, f"the longest run still lost {losses[-1]} games"


def test_self_play_learns_both_sides():
    """Training on both players' moves is what makes one table serve both.

    An agent that only learned as X would be helpless as O, so this checks the
    two halves separately rather than trusting the combined total.
    """
    solver = TicTacToeSolver()
    agent = SelfPlayQLearning(
        learning_rate=0.3, epsilon=LinearDecay(0.9, 0.05, 40_000), seed=0
    )
    agent.train(episodes=50_000, seed=0)

    perfect = PerfectPlayer(solver, seed=0)
    as_x = play_match(agent, perfect, games=50, seed=0)
    as_o = play_match(perfect, agent, games=50, seed=1)

    assert as_x["o_wins"] == 0, "the agent loses when playing X"
    assert as_o["x_wins"] == 0, "the agent loses when playing O"


def test_self_play_is_reproducible():
    def run():
        agent = SelfPlayQLearning(epsilon=LinearDecay(0.9, 0.05, 2000), seed=3)
        agent.train(episodes=2_000, seed=3)
        return {k: v.copy() for k, v in agent.q.items()}

    a, b = run(), run()
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_agent_only_ever_plays_legal_moves():
    """Illegal moves are masked, not penalised, so this should hold from move one."""
    agent = SelfPlayQLearning(seed=0)
    agent.train(episodes=500, seed=0)

    game = TicTacToe()
    for _ in range(50):
        game.reset()
        while not game.is_terminal():
            move = agent.greedy_move(game)
            assert move in game.legal_actions(), f"chose occupied square {move}"
            game.step(move)
