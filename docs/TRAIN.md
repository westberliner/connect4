# How train.py Works

This document explains how the Connect 4 training script works: its flow, components, and how to interpret and tune it.

---

## Overview

`train.py` uses **reinforcement learning** to teach an AI to play Connect 4. The AI (the “agent”) plays many games against a fixed opponent (random, greedy, or minimax). It gets rewards for winning and penalties for losing, and a neural network is updated so it gradually learns better moves.

```
┌─────────────────────────────────────────────────────────────────┐
│                        train.py flow                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Parse arguments (algorithm, opponent, timesteps, etc.)       │
│  2. Create run directories (logs/, models/)                      │
│  3. Build vectorized environment (N parallel Connect4 games)      │
│  4. Create PPO or DQN model                                     │
│  5. Register callbacks (checkpoints, evaluation, TensorBoard)     │
│  6. model.learn(total_timesteps)  ← main training loop            │
│  7. Save final model                                             │
│  8. Run final evaluation vs random, greedy, minimax               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Main Components

### 1. `make_env(opponent_type, render_mode, minimax_depth)`

**Purpose:** Builds a single Connect 4 environment that the RL library can use.

- **Opponent:** Chooses who the agent plays against:
  - `random` – random valid moves
  - `greedy` – wins if possible, blocks your wins, else random
  - `minimax` – looks several moves ahead (depth set by `minimax_depth`)
- **Render mode:** Usually `None` during training (no window). Can be `"human"` or `"rgb_array"` for debugging.
- **Monitor:** Wraps the env in `Monitor` so episode rewards and lengths are logged for TensorBoard.

Returns a **factory function** (a function that returns a new env when called). Stable-Baselines3 uses this to create multiple independent environments.

---

### 2. `evaluate_against_opponent(model, opponent_type, n_games)`

**Purpose:** Measures how good the current model is against a given opponent.

- Loads the right opponent (random, greedy, or minimax).
- Runs `n_games` full games.
- For each game:
  - Gets observation from env.
  - Asks the model for an action with `model.predict(obs, deterministic=True)`.
  - If the action is invalid (full column), falls back to first valid move.
  - Steps the env until the game ends.
- Counts wins, draws, losses.

**Returns:** `(win_rate, draw_rate, loss_rate)` as fractions in [0, 1].

This is used both **during** training (via `EvalCallback`) and **after** training for the final report.

---

### 3. `train(args)`

**Purpose:** Runs the full training pipeline.

#### Step A: Directories

- Builds a run name: `{algorithm}_{opponent}_{timestamp}` (e.g. `PPO_minimax_20250129_143022`).
- Creates:
  - `logs/<run_name>/` – TensorBoard logs and evaluation logs.
  - `models/<run_name>/` – checkpoints and final model.

#### Step B: Vectorized environment

- **Why vectorize?** Multiple envs run in parallel so the agent sees more games per second.
- **`n_envs > 1`:** Uses `SubprocVecEnv` – each env in a separate process. Faster, but a bit more resource-heavy.
- **`n_envs == 1`:** Uses `DummyVecEnv` – single process. Good for debugging.

Each env is created by `make_env(...)`, so they all use the same opponent type and (for minimax) the same depth.

#### Step C: Algorithm (PPO or DQN)

**PPO (Proximal Policy Optimization)** – default:

- **Policy:** `MlpPolicy` – a small MLP that maps board state → action (or action probabilities).
- **n_steps=2048:** Before each update, the agent plays 2048 steps **in total across all envs** (e.g. 2048/4 ≈ 512 steps per env if `n_envs=4`).
- **batch_size=64:** Each gradient update uses 64 of those steps.
- **n_epochs=10:** Each batch of 2048 steps is used for 10 passes (epochs) when updating the policy.
- **gamma=0.99:** Discount for future rewards (how much to value later wins vs immediate).

**DQN (Deep Q-Network):**

- Learns a Q-value for each (state, action).
- **buffer_size=100000:** Replay buffer of past transitions.
- **learning_starts=1000:** Collect 1000 steps before the first update.
- **exploration_fraction / exploration_final_eps:** Control epsilon-greedy exploration (random moves early, fewer later).

Both log to TensorBoard via `tensorboard_log=log_dir`.

#### Step D: Callbacks

- **CheckpointCallback:** Every `checkpoint_freq` timesteps, saves the current model to `models/<run_name>/connect4_<n>_steps.zip`. Lets you stop early and still keep a model.
- **EvalCallback:** Every `eval_freq` timesteps, runs 20 evaluation episodes on `eval_env`, logs mean reward to TensorBoard, and can save the best model so far.

Callbacks are combined in a `CallbackList` and passed to `model.learn(...)`.

#### Step E: Training loop

- `model.learn(total_timesteps=..., callback=..., progress_bar=True)` runs the main loop:
  - The agent acts in all envs, collects transitions.
  - When enough steps are collected (e.g. 2048 for PPO), it updates the neural network.
  - Callbacks run at their frequencies (checkpoint, eval).
  - Progress bar shows `current_step / total_timesteps`.

Training stops when `total_timesteps` is reached (or slightly over, due to batch boundaries).

#### Step F: Save and evaluate

- Saves the final policy to `models/<run_name>/final_model`.
- Runs `evaluate_against_opponent` for 100 games vs **random**, **greedy**, and **minimax**.
- Prints a small table of win/draw/loss rates for each opponent.

---

## Command-line arguments

| Argument | Default | Meaning |
|----------|---------|--------|
| `--algorithm` | PPO | `PPO` or `DQN`. |
| `--opponent` | random | Opponent: `random`, `greedy`, or `minimax`. |
| `--minimax-depth` | 4 | Search depth for minimax (stronger/slower if higher). |
| `--total-timesteps` | 100000 | How many env steps to train for. |
| `--learning-rate` | 3e-4 | Learning rate for the optimizer. |
| `--n-envs` | 4 | Number of parallel environments. |
| `--checkpoint-freq` | 10000 | Save a checkpoint every N steps. |
| `--eval-freq` | 5000 | Run evaluation and log every N steps. |

Example:

```bash
python train.py --opponent minimax --minimax-depth 4 --total-timesteps 1000000
```

---

## Full games vs step-based updates

**Episodes are full games.** One episode = one complete game (from empty board to win/draw). Each *step* is a single move (one piece dropped). So one game might be 20–40 steps.

**Training updates are on batches of steps**, not “one game at a time.” For PPO, the agent collects e.g. 2048 steps *in total across all envs*. Those steps can span many games (e.g. 50+ games across 4 envs). Then the network is updated using that batch of transitions.

The agent still **learns from full-game outcomes** because:
- The final reward (+1 win, -1 loss, 0 draw) is given on the last step of each game.
- The algorithm (e.g. PPO) uses **discounting** (gamma) and **advantage estimation** to credit that outcome back to earlier moves in the same game. So winning a game makes the moves that led to it more likely; losing makes them less likely.

So: **games are full**, but **learning happens on mixed batches of steps** from many games, with win/loss/draw flowing backward to the moves in each episode.

---

## Data flow during training

1. **Observation:** Each env returns a 6×7 board from the **agent’s** view: `1` = agent, `-1` = opponent, `0` = empty.
2. **Action:** The model outputs a column 0–6. The env only accepts valid columns (non-full); invalid choices are corrected in evaluation, and the env can terminate with a penalty for invalid moves during training.
3. **Reward:** Win +1, loss -1, draw 0, invalid move -0.5, small step penalty -0.01 (see `Connect4Env`).
4. **Update:** The algorithm uses many (observation, action, reward, next observation) tuples to update the policy so that good moves get higher probability (PPO) or higher Q-value (DQN).

---

## Why the progress bar can show more than 1,000,000

Training proceeds in **batches**. For PPO with `n_steps=2048` and `n_envs=4`, one “rollout” is 2048 steps. The loop will finish the current rollout even if that pushes the total over `total_timesteps`. So you might see e.g. 1,006,040/1,000,000; that’s normal. After that, the script saves the model and runs the final evaluation (which can take a while vs minimax).

---

## Summary

- **train.py** sets up parallel Connect 4 envs, a PPO or DQN model, and callbacks for checkpoints and evaluation.
- **Training** = many games against the chosen opponent, with the neural network updated from rewards.
- **When it reaches (or slightly exceeds) total_timesteps**, it saves the final model and reports win/draw/loss rates vs random, greedy, and minimax.

For more on the environment (rewards, observation, actions), see `Connect4Env.py`. For playing or evaluating trained models, see `README.md` and `evaluate.py`.

---

## Making the agent as strong as possible (“undefeatable”)

Connect 4 is **solved**: with perfect play, the first player can always force a win. RL can get very strong but usually does not reach mathematically perfect play. To push your trained agent as far as possible with the current setup:

### Parameters that matter most

| Parameter | For maximum strength | Why |
|-----------|----------------------|-----|
| **Opponent** | `minimax` | Only minimax teaches the agent to think ahead; random/greedy are too weak. |
| **minimax-depth** | `5` or `6` | Deeper = stronger opponent = harder training signal. Depth 6 is very strong; 7+ is slow. |
| **total-timesteps** | `2_000_000` or more | More games = more learning. Diminishing returns after a while. |
| **algorithm** | `PPO` | Tends to work better than DQN for this kind of game. |
| **n-envs** | `8` | More parallel envs = more games per second = faster training. |
| **learning-rate** | `3e-4` (default) or lower `1e-4` for long runs | Lower can help stability when training long. |

### Recommended training strategy (curriculum)

Train in stages against stronger and stronger opponents, then finish against strong minimax:

```bash
# 1. Basics (optional, if starting from scratch)
python train.py --opponent random --total-timesteps 200000

# 2. Tactics (wins, blocks)
python train.py --opponent greedy --total-timesteps 300000

# 3. Strong opponent (load best so far if you have it, or start fresh)
python train.py --opponent minimax --minimax-depth 4 --total-timesteps 500000

# 4. Very strong opponent
python train.py --opponent minimax --minimax-depth 5 --total-timesteps 1000000

# 5. As strong as current opponent gets (slow per game)
python train.py --opponent minimax --minimax-depth 6 --total-timesteps 2000000 --n-envs 8
```

To **continue from a saved model** (e.g. from stage 3 to 4), you’d need to add a `--load` option to `train.py` and pass the path; the script doesn’t support that by default yet.

### What “undefeatable” means here

- **Vs random/greedy:** With enough training vs minimax, the agent can get to near 100% win rate.
- **Vs minimax depth 4–6:** The agent can get strong (e.g. 50–70% win or better depending on depth and who goes first) but likely won’t be perfect.
- **True perfect play:** Would require a solver (e.g. minimax/alpha-beta to full depth or a solved opening book), not RL. For a “human-level undefeatable” bot, the settings above are a practical target.
