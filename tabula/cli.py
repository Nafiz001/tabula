"""Command line interface: train agents, watch them, and play against them.

    tabula train grid          tabular Q-learning on a grid
    tabula cliff               Q-learning against SARSA on the cliff
    tabula train cartpole      DQN on the pole-balancing task
    tabula train tictactoe     self-play, then a score against perfect play
    tabula train alphazero     search-guided self-play with a network
    tabula play                play tic-tac-toe against a trained agent
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _bar(value: float, low: float, high: float, width: int = 30) -> str:
    """A crude horizontal bar, so learning curves are visible in a terminal."""
    if high <= low:
        return ""
    fraction = max(0.0, min(1.0, (value - low) / (high - low)))
    filled = int(fraction * width)
    return "#" * filled + "." * (width - filled)


def cmd_grid(args) -> int:
    from tabula.agents import QLearning, RandomAgent
    from tabula.core import LinearDecay, evaluate, train
    from tabula.envs import default_gridworld

    env = default_gridworld()
    print("GridWorld: reach G, avoiding the walls. Each step costs a little,")
    print("so the agent is pushed toward the shortest route.\n")
    print(env.render())
    print()

    baseline = evaluate(env, RandomAgent(env.n_actions, seed=0), episodes=50, seed=1000)
    print(f"random walk : {baseline}")

    agent = QLearning(
        env.n_states, env.n_actions,
        learning_rate=0.2, epsilon=LinearDecay(1.0, 0.05, args.episodes // 2), seed=args.seed,
    )
    history = train(env, agent, episodes=args.episodes, seed=args.seed)
    result = evaluate(env, agent, episodes=50, seed=1000)
    print(f"after {args.episodes:5d}: {result}\n")

    print("learning curve (mean return per block of episodes):")
    block = max(1, args.episodes // 12)
    returns = history.returns
    lo, hi = min(returns), max(returns)
    for i in range(0, len(returns), block):
        chunk = returns[i : i + block]
        mean = sum(chunk) / len(chunk)
        print(f"  {i:5d} {mean:+7.2f} {_bar(mean, lo, hi)}")

    print("\nlearned policy:")
    print(env.render(policy=agent.greedy_policy()))
    return 0


def cmd_cliff(args) -> int:
    """The comparison that motivates having both algorithms."""
    from tabula.agents import QLearning, SARSA
    from tabula.core import LinearDecay, evaluate, train
    from tabula.envs import CliffWalking

    print("CliffWalking: S to G along the bottom. The C squares are a cliff --")
    print("stepping on one costs -100 and sends the agent back to S.\n")
    print("Q-learning is off-policy: it learns the value of playing perfectly,")
    print("so it takes the shortest path along the edge. SARSA is on-policy: it")
    print("learns the value of its own exploring behaviour, so it keeps clear.\n")

    for cls in (QLearning, SARSA):
        env = CliffWalking()
        agent = cls(
            env.n_states, env.n_actions,
            learning_rate=0.5, discount=1.0,
            epsilon=LinearDecay(1.0, 0.1, args.episodes // 2), seed=args.seed,
        )
        history = train(env, agent, episodes=args.episodes, seed=args.seed)
        result = evaluate(env, agent, episodes=20, seed=99)
        online = float(np.mean(history.returns[-500:]))

        print(f"--- {cls.__name__} ---")
        print(f"  greedy return : {result.mean_return:7.1f}   (the policy it learned)")
        print(f"  online return : {online:7.1f}   (what it scored while still exploring)")
        print(env.render(policy=agent.greedy_policy()))
        print()

    print("Q-learning's greedy path is shorter; SARSA scores better while exploring.")
    print("Neither is wrong -- they answer different questions.")
    return 0


def cmd_cartpole(args) -> int:
    from tabula.agents import DQN, RandomAgent
    from tabula.core import LinearDecay, evaluate, train
    from tabula.envs import CartPole

    env = CartPole()
    print("CartPole: keep the pole upright by pushing the cart left or right.")
    print("The return is simply how many steps it stayed up, out of 500.\n")

    baseline = evaluate(env, RandomAgent(2, seed=0), episodes=20, seed=500)
    print(f"random pushes : {baseline.mean_return:.0f} steps\n")

    agent = DQN(
        env.state_dim, env.n_actions,
        learning_rate=1e-3, epsilon=LinearDecay(1.0, 0.02, 10_000), seed=args.seed,
    )

    start = time.time()
    block = max(1, args.episodes // 12)

    def report(episode, history):
        if (episode + 1) % block == 0:
            recent = float(np.mean(history.returns[-block:]))
            print(f"  episode {episode + 1:4d}: {recent:6.1f} steps  {_bar(recent, 0, 500)}")

    print(f"training DQN for {args.episodes} episodes:")
    train(env, agent, episodes=args.episodes, seed=args.seed, on_episode=report)

    result = evaluate(env, agent, episodes=20, seed=500)
    print(f"\nafter training: {result.mean_return:.0f} steps  ({time.time() - start:.0f}s)")

    if args.watch:
        print("\none episode, greedy:")
        state = env.reset(seed=7)
        for i in range(200):
            step = env.step(agent.act(state, greedy=True))
            state = step.state
            if i % 20 == 0:
                print("  " + env.render())
            if step.done:
                break
    return 0


def cmd_tictactoe(args) -> int:
    from tabula.agents.selfplay import (
        SelfPlayQLearning,
        evaluate_against_perfect,
        policy_agreement,
    )
    from tabula.core import LinearDecay
    from tabula.envs import RandomPlayer, TicTacToeSolver

    print("Tic-tac-toe by self-play. The agent is given the rules and nothing")
    print("else: no examples, no opponent, no heuristics. It plays itself.\n")

    solver = TicTacToeSolver()
    print(f"solving the game exhaustively for ground truth...", end=" ", flush=True)
    start = time.time()
    from tabula.envs import TicTacToe

    game = TicTacToe()
    game.reset()
    value = solver.value(game)
    print(f"{solver.positions_solved()} positions in {time.time() - start:.2f}s")
    print(f"value of the empty board: {value}  (0 means a draw with perfect play)\n")

    baseline = evaluate_against_perfect(RandomPlayer(seed=0), solver, games=100, seed=1)
    print(f"random play vs perfect : {baseline['losses']:3d} losses, {baseline['draws']:3d} draws of 100")

    agent = SelfPlayQLearning(
        learning_rate=0.3,
        epsilon=LinearDecay(0.9, 0.05, args.episodes * 0.8),
        seed=args.seed,
    )

    checkpoints = [args.episodes // 8, args.episodes // 4, args.episodes // 2, args.episodes]
    played = 0
    print()
    for target in checkpoints:
        agent.train(episodes=target - played, seed=args.seed + played)
        played = target
        result = evaluate_against_perfect(agent, solver, games=100, seed=1)
        agreement = policy_agreement(agent, solver, games=200, seed=2)
        print(
            f"after {played:6d} self-play games: "
            f"{result['losses']:3d} losses, {result['draws']:3d} draws  |  "
            f"{agreement:5.1%} of moves optimal"
        )

    print(f"\npositions learned: {agent.positions_learned}")
    if args.save:
        import pickle

        with open(args.save, "wb") as f:
            pickle.dump(agent.q, f)
        print(f"saved to {args.save}")
    return 0


def cmd_alphazero(args) -> int:
    from tabula.agents import AlphaZeroAgent, RawNetworkPlayer
    from tabula.agents.selfplay import evaluate_against_perfect, policy_agreement
    from tabula.envs import TicTacToeSolver

    print("AlphaZero-style self-play: tree search guided by a neural network,")
    print("with the network trained on what the search discovers.\n")
    print("The search improves on the raw network, so its move distribution is a")
    print("better policy than the network's own. Training toward it pulls the")
    print("network up, which makes the next search stronger again.\n")

    solver = TicTacToeSolver()
    agent = AlphaZeroAgent(simulations=args.simulations, learning_rate=1e-3, seed=args.seed)

    before = evaluate_against_perfect(agent, solver, games=40, seed=1)
    print(f"untrained: {before['losses']:2d} losses of 40 vs perfect play\n")

    start = time.time()

    def report(entry):
        if (entry["iteration"] + 1) % max(1, args.iterations // 6) == 0:
            result = evaluate_against_perfect(agent, solver, games=40, seed=1)
            agreement = policy_agreement(RawNetworkPlayer(agent), solver, games=60, seed=2)
            print(
                f"  iteration {entry['iteration'] + 1:3d}  "
                f"replay {entry['replay']:5d}  "
                f"{result['losses']:2d} losses of 40  |  "
                f"network alone {agreement:5.1%} optimal  |  "
                f"value loss {entry['value_loss']:.3f}"
            )

    agent.train(
        iterations=args.iterations,
        games_per_iteration=args.games,
        epochs=3,
        seed=args.seed,
        on_iteration=report,
    )

    result = evaluate_against_perfect(agent, solver, games=100, seed=7)
    print(f"\nfinal: {result['losses']} losses, {result['draws']} draws of 100 "
          f"({time.time() - start:.0f}s)")
    if args.save:
        agent.save(args.save)
        print(f"saved to {args.save}")
    return 0


def cmd_play(args) -> int:
    """Play tic-tac-toe against an agent trained on the spot."""
    from tabula.agents.selfplay import SelfPlayQLearning
    from tabula.core import LinearDecay
    from tabula.envs import TicTacToe

    print(f"Training an opponent ({args.episodes} self-play games)...", end=" ", flush=True)
    agent = SelfPlayQLearning(
        learning_rate=0.3, epsilon=LinearDecay(0.9, 0.05, args.episodes * 0.8), seed=args.seed
    )
    agent.train(episodes=args.episodes, seed=args.seed)
    print("done.\n")
    print("You are X, the agent is O. Squares are numbered:\n")
    print("  0 1 2\n  3 4 5\n  6 7 8\n")

    game = TicTacToe()
    game.reset()

    while not game.is_terminal():
        print(game.render())
        if game.current_player == 1:
            legal = game.legal_actions().tolist()
            try:
                raw = input(f"your move {legal}: ").strip()
            except EOFError:
                print()
                return 0
            if not raw.isdigit() or int(raw) not in legal:
                print("  not a legal square\n")
                continue
            game.step(int(raw))
        else:
            move = agent.greedy_move(game)
            print(f"agent plays {move}\n")
            game.step(move)

    print(game.render())
    winner = game.winner()
    if winner == 1:
        print("\nYou win. That should not happen against a fully trained agent.")
    elif winner == -1:
        print("\nThe agent wins.")
    else:
        print("\nA draw, which is the correct result with best play from both sides.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tabula",
        description="Reinforcement learning from scratch: agents that start from nothing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="train an agent")
    train_sub = train_parser.add_subparsers(dest="task", required=True)

    p = train_sub.add_parser("grid", help="tabular Q-learning on a grid")
    p.add_argument("--episodes", type=int, default=2000)
    p.set_defaults(func=cmd_grid)

    p = train_sub.add_parser("cartpole", help="DQN on pole balancing")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--watch", action="store_true", help="print an episode afterwards")
    p.set_defaults(func=cmd_cartpole)

    p = train_sub.add_parser("tictactoe", help="tabular self-play")
    p.add_argument("--episodes", type=int, default=60_000)
    p.add_argument("--save", help="write the learned table here")
    p.set_defaults(func=cmd_tictactoe)

    p = train_sub.add_parser("alphazero", help="search-guided self-play with a network")
    p.add_argument("--iterations", type=int, default=36)
    p.add_argument("--games", type=int, default=25, help="self-play games per iteration")
    p.add_argument("--simulations", type=int, default=60, help="MCTS simulations per move")
    p.add_argument("--save", help="write the trained network here")
    p.set_defaults(func=cmd_alphazero)

    p = sub.add_parser("cliff", help="Q-learning against SARSA on the cliff")
    p.add_argument("--episodes", type=int, default=3000)
    p.set_defaults(func=cmd_cliff)

    p = sub.add_parser("play", help="play tic-tac-toe against a trained agent")
    p.add_argument("--episodes", type=int, default=60_000)
    p.set_defaults(func=cmd_play)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
