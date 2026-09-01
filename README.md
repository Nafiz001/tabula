# tabula

Reinforcement learning from scratch in Python. Agents that begin knowing
nothing and learn by acting — no gym, no stable-baselines, no pretrained
anything. The environments, the algorithms, the replay buffer and the tree
search are all implemented here.

```console
$ tabula train tictactoe

Tic-tac-toe by self-play. The agent is given the rules and nothing
else: no examples, no opponent, no heuristics. It plays itself.

solving the game exhaustively for ground truth... 4804 positions in 0.15s
value of the empty board: 0  (0 means a draw with perfect play)

random play vs perfect :  90 losses,  10 draws of 100

after   7500 self-play games:  63 losses,  37 draws  |  78.3% of moves optimal
after  15000 self-play games:  78 losses,  22 draws  |  84.0% of moves optimal
after  30000 self-play games:  14 losses,  86 draws  |  94.7% of moves optimal
after  60000 self-play games:   0 losses, 100 draws  |  97.2% of moves optimal
```

That last line is the point of the project. The agent was given the rules and
nothing else, played itself sixty thousand times, and ended up unable to lose
to an opponent that is **provably** unbeatable.

Note that the middle of that run gets *worse* before it gets better — 63 losses,
then 78. That is real and it is expected: epsilon is still high there, so the
agent is deliberately playing badly a good deal of the time, and the table is
being rewritten faster than it is being refined. The output is left as it came
out rather than smoothed, because a curve that only ever goes up is usually a
sign the run was cherry-picked.

## Why measurement, not vibes

Most reinforcement learning demos show a curve going up. A curve going up is
weak evidence: it can mean the agent learned the task, or that it learned to
exploit a bug in the reward, or that the baseline was bad.

Tic-tac-toe has 5,478 reachable positions, so it can be solved outright by
exhaustive minimax. That gives ground truth for every position in the game, and
turns vague claims into numbers:

- **Losses against perfect play.** Tic-tac-toe is a draw with best play, so a
  fully trained agent draws every game from both sides. Any loss is a concrete
  mistake.
- **Policy agreement.** A random walk through the game tree, grading every
  choice against the moves minimax certifies as optimal. This is the finer
  instrument: an agent can draw every game against a perfect opponent while
  still playing loosely in positions that opponent never steers into.

| | vs perfect play | moves optimal |
|---|---|---|
| Random | 90 losses / 100 | 61% |
| Self-play Q-learning | **0 losses / 100** | 97% |
| AlphaZero-style | **0 losses / 100** | 98% |

## What is in it

| Algorithm | | |
|---|---|---|
| Q-learning | off-policy TD control | tabular |
| SARSA | on-policy TD control | tabular |
| Expected SARSA | averaged over the policy | tabular |
| DQN | replay buffer + target network + Double DQN | PyTorch |
| Self-play Q-learning | two-player, canonical boards | tabular |
| MCTS | PUCT selection, network-guided | — |
| AlphaZero | search + policy/value net + self-play | PyTorch |

| Environment | | |
|---|---|---|
| GridWorld | walls, step costs, optional slippery moves | tabular |
| CliffWalking | the classic, and the reason both TD algorithms are here | tabular |
| CartPole | cart-pole physics, semi-implicit Euler | continuous |
| Tic-tac-toe | plus an exhaustive minimax solver | two-player |

## The cliff: one line of code, two different answers

Q-learning and SARSA differ in a single symbol of their update, and
`CliffWalking` exists to make that difference visible.

```
Q-learning:  Q(s,a) <- Q(s,a) + lr * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
SARSA:       Q(s,a) <- Q(s,a) + lr * [ r + gamma *        Q(s',a') - Q(s,a) ]
```

Q-learning takes the **max** — the value of playing perfectly — so it is
off-policy and finds the optimal path hugging the cliff edge. SARSA uses the
action it is **actually about to take**, which under epsilon-greedy is sometimes
random, so it prices in the chance of slipping and detours a row higher.

```console
$ tabula cliff

--- QLearning ---
  greedy return :   -13.0        <- the shortest path
  online return :   -56.1        <- but it falls off while exploring
^ < > > > v > v > > v v
v v > > > > > > > > > v
> > > > > > > > > > > v
S C C C C C C C C C C G          <- walks the edge

--- SARSA ---
  greedy return :   -17.0        <- a longer path
  online return :   -28.6        <- but it survives exploration
> > > > > > > > > > > v
^ ^ ^ > ^ > ^ ^ ^ ^ ^ v
^ ^ ^ ^ ^ < ^ ^ ^ ^ > v
S C C C C C C C C C C G          <- keeps its distance
```

Neither is wrong. Q-learning answers "what is the best path". SARSA answers
"what is the best path *for someone who sometimes slips*". Both come out of the
same training loop, and the tests assert both halves.

## Deep RL: what actually breaks, and what fixes it

DQN is Q-learning with a network instead of a table. That single substitution
breaks it in two specific ways, and DQN is really the name of the two fixes:

**Correlated samples.** Consecutive states are nearly identical — the pole leans
a fraction further — so training on the stream in order means every gradient
step sees the same thing. The replay buffer hands back random batches from a
large window of past experience instead.

**A moving target.** The target is `r + gamma * max Q(s')`, computed with the
same network being updated, so every step changes the thing it is trying to
match. A frozen target network supplies the targets and is synced periodically,
so the target sits still long enough to be aimed at.

Both fixes buy back an assumption the table version got for free. That is the
story of deep RL in miniature.

**A tuning result worth recording.** With the target synced every 1000 updates,
CartPole plateaus at 120 steps. At 200, the same agent reaches the 500-step
ceiling. The target was simply too stale — nothing else changed.

```
random pushes    :  20 steps
DQN, 400 episodes: 500 steps  (the maximum)
```

### An honest note on discretisation

The tempting claim is that CartPole is where tables stop working. Measured, that
is false: binning each of the four dimensions into six buckets reaches **405 of
500 steps**. CartPole is low-dimensional and smooth, so coarse bins genuinely
capture it.

What the network actually buys is sample efficiency and the last stretch of
performance — 500 steps in 400 episodes against 405 in 3,000. Where binning
*does* break down is dimensionality (bins^dims: four dimensions at six bins is
1,296 states, ten dimensions is sixty million) and generalisation (a table
learns nothing about a bin from its neighbours; a network shares weights across
the whole space).

## AlphaZero, scaled down

The loop, in full:

1. Play games against yourself, choosing moves by running MCTS with the current
   network.
2. Record each position's search visit distribution and who eventually won.
3. Train the network so its policy matches the search and its value matches the
   outcome.
4. Repeat.

It works because step 3 trains the network on something **better than itself**.
A hundred simulations of lookahead beat a single forward pass, so the visit
counts are a stronger policy than the priors that produced them. Training toward
them pulls the network up, which makes the next search stronger, which produces
better targets again.

Classic MCTS estimates a leaf by playing the game out at random. AlphaZero
replaces that rollout with one call to a value network: instead of guessing by
playing badly a thousand times, ask a function that has learned what good
positions look like. Selection uses PUCT, where the network's prior decides
early and measured value takes over as visits accumulate.

```console
$ tabula train alphazero
untrained:  12 losses of 40 vs perfect play

  iteration   6  replay  1074   6 losses of 40  |  network alone 93.1% optimal
  iteration  12  replay  2183   7 losses of 40  |  network alone 93.6% optimal
  iteration  18  replay  3318   2 losses of 40  |  network alone 97.6% optimal
  iteration  24  replay  4545   0 losses of 40  |  network alone 97.6% optimal
  iteration  36  replay  7050   0 losses of 40  |  network alone 98.2% optimal

final: 0 losses, 100 draws of 100 (170s)
```

**The bug that mattered most.** The first version discarded each iteration's
games after training on them once. It plateaued at 10–12 losses in 40 and would
not improve. Every batch was a few hundred positions from a single snapshot of
the policy, so the network overfit that snapshot and then forgot it. Keeping a
window of the last 8,000 examples — so each update sees positions from many
policy versions — was the difference between a permanent plateau and reaching
zero losses.

## Running it

```console
$ pip install -e ".[dev]"

$ tabula train grid              # tabular Q-learning, with a learning curve
$ tabula cliff                   # Q-learning vs SARSA, side by side
$ tabula train cartpole          # DQN, with --watch to see an episode
$ tabula train tictactoe         # self-play, scored against perfect play
$ tabula train alphazero         # search-guided self-play with a network
$ tabula play                    # play tic-tac-toe against a trained agent
```

## Testing

```console
$ pytest
69 passed
```

Reinforcement learning is unusually easy to test badly: an agent that learns
*something* is not one that learned the right thing, and a run that improves
once is not one that improves reliably. So:

- **Everything is seeded.** Identical runs produce identical tables, asserted
  directly. A flaky RL test is worse than no test.
- **Claims are relative to a baseline**, never to an absolute number — a random
  walk reaches the goal too, eventually.
- **Policies are checked structurally**, not just by score: the optimal path
  across an empty 5×5 grid is exactly 8 steps, and SARSA's first move from the
  corner must be *away* from the cliff.
- **Search is checked with a blind evaluator** — uniform priors, zero value — so
  anything MCTS finds, it found by looking ahead rather than by being told.
- **The solver itself is tested**, because every other measurement is graded
  against it: the empty board is a draw, all nine openings draw, and the centre
  is the *only* reply that saves a corner opening.

Full training runs are marked `slow` and excluded from the default pass.

### A test that was wrong, and what it revealed

I asserted that in this position O must block square 2:

```
X X .
O . .
. . .
```

The solver disagreed, reporting *every* legal move as optimal. It was right and
I was wrong: O is already lost. Blocking at 2 loses to X taking the centre,
which creates two threats at once, and every other move loses immediately. Since
all moves share the same value, minimax is indifferent.

That is also a genuine limitation of plain minimax, now documented in the test:
it will not prefer to lose slowly, or pick the line that gives a fallible
opponent the most chances to go wrong.

## Known limitations

- **The environments are small.** Everything here is chosen so results can be
  verified, which rules out anything with pixels. Scaling to Atari would need
  convolutions, frame stacking, and far more compute than a CPU.
- **No policy-gradient methods.** No REINFORCE, A2C or PPO, so nothing here
  handles a continuous *action* space — only continuous states.
- **DQN uses uniform replay.** Prioritised replay samples surprising
  transitions more often and usually learns faster; it is not implemented.
- **MCTS is single-threaded** with no tree reuse between moves. Each move
  rebuilds the tree from scratch, which is fine at 60 simulations and wasteful
  at 60,000.
- **AlphaZero has no arena gate.** Real AlphaZero only adopts a new network if
  it beats the old one in a match; here every iteration is accepted, which is
  fine for a game this small but would eventually let a bad update through.
- **Tic-tac-toe is the only two-player game.** Connect Four would be the honest
  next step — still verifiable, but too large to solve exhaustively on a laptop,
  so the ground-truth benchmark would have to go.

## Layout

```
tabula/core/       types, schedules, replay buffer, MCTS, training loop
tabula/envs/       GridWorld, CliffWalking, CartPole, tic-tac-toe + solver
tabula/agents/     Q-learning, SARSA, Expected SARSA, DQN, self-play, AlphaZero
tabula/cli.py      train, compare, and play
tests/             69 tests, seeded and baseline-relative
```
