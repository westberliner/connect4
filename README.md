# Connect 4 AI Training Environment

A visual Connect 4 game with reinforcement learning training capabilities. Train AI agents using PPO/DQN algorithms and watch them learn to play.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-green)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0%2B-orange)

## Features

- Visual game with Pygame rendering
- Human vs Human, Human vs AI, and AI vs AI modes
- Gymnasium-compatible environment for RL training
- PPO and DQN algorithm support via Stable-Baselines3
- TensorBoard integration for training visualization
- Built-in opponents: Random and Greedy (blocks/wins)

---

## Installation

### Using pipenv (recommended)

```bash
# Clone or navigate to the project
cd connect4

# Install pipenv if you don't have it
pip install pipenv

# Create virtual environment and install dependencies
pipenv install

# Activate the environment
pipenv shell
```

### Create Pipfile

If you don't have a Pipfile, create one:

```bash
pipenv install pygame numpy gymnasium "stable-baselines3" torch tensorboard tqdm rich
```

### Alternative: Using venv

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Playing the Game

### Human vs Human

```bash
python connect4.py
```

### Play Against Random AI

```bash
python connect4.py --mode random
```

### Play Against Greedy AI

The greedy AI will block your winning moves and take its own wins:

```bash
python connect4.py --mode greedy
```

### Play Against Trained Model

```bash
python connect4.py --mode ai --model models/<run_name>/final_model
```

### Let AI Go First

Add `--ai-first` to any AI mode:

```bash
python connect4.py --mode greedy --ai-first
```

### Controls

| Key | Action |
|-----|--------|
| Click | Drop piece in column |
| R | Restart game |
| Q | Quit |

---

## Training the AI

### Basic Training

Train against the random opponent (easiest):

```bash
python train.py
```

This will:
- Train for 100,000 timesteps using PPO
- Save checkpoints to `models/PPO_random_<timestamp>/`
- Log training data to `logs/PPO_random_<timestamp>/`

### Training Options

```bash
# Train against greedy opponent (harder)
python train.py --opponent greedy

# Train for longer
python train.py --total-timesteps 500000

# Use DQN instead of PPO
python train.py --algorithm DQN

# Adjust learning rate
python train.py --learning-rate 0.0001

# Use more parallel environments (faster training)
python train.py --n-envs 8
```

### All Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` | PPO | RL algorithm (PPO or DQN) |
| `--opponent` | random | Opponent type (random or greedy) |
| `--total-timesteps` | 100000 | Training duration |
| `--learning-rate` | 0.0003 | Learning rate |
| `--n-envs` | 4 | Parallel environments |
| `--checkpoint-freq` | 10000 | Save model every N steps |
| `--eval-freq` | 5000 | Evaluate every N steps |

### Training Strategy

1. **Start with random opponent** to learn basic gameplay:
   ```bash
   python train.py --opponent random --total-timesteps 200000
   ```

2. **Graduate to greedy opponent** for strategic play:
   ```bash
   python train.py --opponent greedy --total-timesteps 500000
   ```

---

## Monitoring with TensorBoard

### Start TensorBoard

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Key Metrics to Watch

| Metric | Description | Good Sign |
|--------|-------------|-----------|
| `rollout/ep_rew_mean` | Average episode reward | Increasing toward +1 |
| `rollout/ep_len_mean` | Average episode length | Stable or decreasing |
| `train/loss` | Training loss | Decreasing, then stable |
| `train/entropy_loss` | Exploration metric | Slowly decreasing |

### Tips

- **Reward approaching +1**: AI is winning most games
- **Reward around 0**: AI is drawing or 50/50 win/loss
- **Reward approaching -1**: AI is losing (needs more training)

---

## Evaluation

### Watch AI Play

```bash
# Watch AI vs random opponent
python evaluate.py watch models/<run>/final_model --opponent random

# Watch AI vs greedy with slower playback
python evaluate.py watch models/<run>/final_model --opponent greedy --delay 1.0
```

### Play Against Your Trained AI

```bash
python evaluate.py play models/<run>/final_model

# Or use connect4.py
python connect4.py --mode ai --model models/<run>/final_model
```

### Benchmark Performance

Run 1000 games to get accurate win rates:

```bash
python evaluate.py benchmark models/<run>/final_model
```

### Watch AI vs AI

```bash
# Same model plays itself
python evaluate.py versus models/<run>/final_model

# Two different models
python evaluate.py versus models/run1/final_model --model2 models/run2/final_model
```

---

## Project Structure

```
connect4/
├── connect4.py       # Main game - play with visual interface
├── Board.py          # Game logic and rules
├── GfxEngine.py      # Pygame rendering
├── Connect4Env.py    # Gymnasium environment for RL
├── train.py          # Training script
├── evaluate.py       # Evaluation and playback
├── requirements.txt  # Dependencies
├── PRD.md           # Product requirements document
├── models/          # Saved model checkpoints
└── logs/            # TensorBoard logs
```

---

## Quick Start Example

```bash
# 1. Install
pipenv install
pipenv shell

# 2. Play the game first
python connect4.py

# 3. Train an AI (takes a few minutes)
python train.py --total-timesteps 100000

# 4. Watch TensorBoard (in another terminal)
tensorboard --logdir logs/

# 5. Play against your trained AI
python connect4.py --mode ai --model models/PPO_random_*/final_model
```

---

## Troubleshooting

### Pygame font error on Python 3.14

The game handles this automatically with a visual fallback. For full text support, use Python 3.11 or 3.12.

### SSL certificate errors during pip install

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Model won't load

Make sure you're pointing to the model file, not the directory:
```bash
# Correct
python connect4.py --mode ai --model models/PPO_random_20240115/final_model

# Wrong (missing final_model)
python connect4.py --mode ai --model models/PPO_random_20240115/
```

---

## License

MIT
