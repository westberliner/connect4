# Connect 4 AI Training Environment - Product Requirements Document

## Overview

Build a visual Connect 4 game environment designed for AI training. The system will provide a graphical representation of the game state that can be used both for human observation during training and as input for vision-based AI models.

---

## Goals

1. **Visual Game Environment**: A clear, real-time graphical display of the Connect 4 board
2. **AI-Compatible Interface**: Standardized environment API compatible with reinforcement learning frameworks
3. **Training Observability**: Watch AI agents learn and play in real-time

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Game Controller                     │
│                      (connect4.py)                       │
├─────────────────────────────────────────────────────────┤
│                           │                              │
│              ┌────────────┴────────────┐                │
│              ▼                         ▼                │
│     ┌─────────────────┐      ┌─────────────────┐       │
│     │      Board      │      │    GfxEngine    │       │
│     │   (Board.py)    │◄────►│ (GfxEngine.py)  │       │
│     └─────────────────┘      └─────────────────┘       │
│              │                         │                │
│              ▼                         ▼                │
│     ┌─────────────────┐      ┌─────────────────┐       │
│     │   Game Logic    │      │  Pygame/Visual  │       │
│     │   Win Detection │      │    Rendering    │       │
│     └─────────────────┘      └─────────────────┘       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    AI Environment Layer                  │
│                    (Connect4Env.py)                      │
│         Gym-compatible wrapper for RL training          │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Game Logic (`Board.py`) ✅ COMPLETE

### Requirements

- [x] **Board Representation**
  - 7 columns × 6 rows grid
  - Cell states: Empty (0), Player 1 (1), Player 2 (2)
  - Store as 2D numpy array for AI compatibility

- [x] **Game Actions**
  - `drop_piece(column, player)` - Drop piece in specified column
  - `get_valid_moves()` - Return list of columns that aren't full
  - `is_valid_move(column)` - Check if a column can accept a piece

- [x] **Win Detection**
  - `check_winner()` - Return winner (1, 2) or None
  - Check horizontal, vertical, and both diagonal directions
  - `is_terminal()` - Return True if game is won or board is full

- [x] **State Management**
  - `reset()` - Clear board for new game
  - `copy()` - Return deep copy of board state
  - `get_state()` - Return numpy array representation

---

## Phase 2: Graphics Engine (`GfxEngine.py`) ✅ COMPLETE

### Requirements

- [x] **Display Setup**
  - Initialize Pygame window (700×700)
  - Define color scheme:
    - Board: Blue (#0066CC)
    - Empty slots: Dark gray
    - Player 1: Red (#E74C3C)
    - Player 2: Yellow (#F1C40F)
    - Background: Dark gray

- [x] **Rendering**
  - `render()` - Draw current board state
  - Draw circular slots for each cell
  - Visual distinction between players
  - Column hover indicator for human play

- [ ] **Animation** (Optional - Deferred)
  - Piece drop animation
  - Win celebration/highlight

- [x] **Frame Capture**
  - `get_frame()` - Return current frame as numpy array (RGB)
  - `get_frame_resized()` - Resized frame for neural network input
  - Used for vision-based AI training

---

## Phase 3: Game Controller (`connect4.py`) ✅ COMPLETE

### Requirements

- [x] **Game Loop**
  - Initialize board and graphics engine
  - Alternate turns between players
  - Handle game termination (win/draw)
  - Support game reset

- [x] **Play Modes**
  - Human vs Human (mouse input) ✅
  - Human vs AI (pending AI implementation)
  - AI vs AI (pending AI implementation)

- [x] **Input Handling**
  - Mouse click to select column (human players)
  - Keyboard shortcuts (R = reset, Q = quit)

---

## Phase 4: AI Environment (`Connect4Env.py`) ✅ COMPLETE

### Requirements

- [x] **Gymnasium Interface**
  - Inherit from `gymnasium.Env`
  - Implement required methods:
    - `reset()` - Reset game, return initial observation
    - `step(action)` - Execute action, return (obs, reward, done, truncated, info)
    - `render()` - Display current state

- [x] **Observation Space**
  - Option A: Board state as 6×7 numpy array (simple) ✅
  - Option B: RGB frame from graphics engine (vision-based) ✅
  - Option C: Both (configurable) ✅

- [x] **Action Space**
  - Discrete(7) - One action per column
  - `action_masks()` for invalid action masking

- [x] **Reward Structure**
  - Win: +1.0
  - Loss: -1.0
  - Draw: 0.0
  - Invalid move: -0.5 (terminates episode)
  - Step penalty: -0.01 (encourages faster wins)

- [x] **Self-Play Support**
  - Environment manages both players via opponent parameter
  - Includes `RandomOpponent` and `GreedyOpponent` classes
  - Supports training against fixed or custom opponents

---

## Phase 5: Training Pipeline (`train.py`) ✅ COMPLETE

### Requirements

- [x] **Algorithm Selection**
  - PPO and DQN implemented
  - Uses Stable-Baselines3

- [x] **Training Configuration**
  - Configurable timesteps, learning rate, batch size
  - Checkpoint saving frequency configurable
  - Parallel environments for faster training

- [x] **Opponent Strategies**
  - Random agent (baseline) ✅
  - Greedy agent (medium difficulty) ✅
  - Self-play (future enhancement)

- [x] **Logging & Monitoring**
  - TensorBoard integration ✅
  - Checkpoint callbacks ✅
  - Evaluation callbacks ✅
  - Final win rate evaluation ✅

---

## Phase 6: Evaluation & Visualization (`evaluate.py`) ✅ COMPLETE

### Requirements

- [x] **Performance Metrics**
  - Win rate vs random agent ✅
  - Win rate vs greedy agent ✅
  - Benchmark command with detailed stats ✅

- [x] **Visual Playback**
  - Load trained model ✅
  - Watch AI play in real-time with graphics ✅
  - Adjustable playback speed (--delay) ✅

- [x] **Play Modes**
  - Watch AI vs opponent (watch command)
  - Play against AI yourself (play command)
  - Watch AI vs AI (versus command)
  - Benchmark performance (benchmark command)

---

## File Structure

```
connect4/
├── connect4.py          # Main game controller
├── Board.py             # Game logic and state
├── GfxEngine.py         # Pygame visualization
├── Connect4Env.py       # Gymnasium environment wrapper
├── train.py             # Training script
├── evaluate.py          # Evaluation and visualization
├── agents/
│   ├── random_agent.py  # Random baseline
│   └── rule_agent.py    # Heuristic-based agent
├── models/              # Saved model checkpoints
├── logs/                # Training logs
├── requirements.txt     # Dependencies
└── PRD.md               # This document
```

---

## Dependencies

```
pygame>=2.5.0           # Graphics rendering
numpy>=1.24.0           # Array operations
gymnasium>=0.29.0       # RL environment interface
stable-baselines3>=2.0  # RL algorithms
torch>=2.0.0            # Neural network backend
tensorboard>=2.14.0     # Training visualization
```

---

## Implementation Order

| Step | Task | Status |
|------|------|--------|
| 1 | Complete `Board.py` with full game logic | ✅ Done |
| 2 | Complete `GfxEngine.py` with Pygame rendering | ✅ Done |
| 3 | Integrate board + engine in `connect4.py` | ✅ Done |
| 4 | Add human playable mode (mouse input) | ✅ Done |
| 5 | Create `Connect4Env.py` Gymnasium wrapper | ✅ Done |
| 6 | Implement baseline agents (random, rule-based) | ✅ Done |
| 7 | Set up training pipeline with Stable-Baselines3 | ✅ Done |
| 8 | Add TensorBoard logging and checkpoints | ✅ Done |
| 9 | Create evaluation scripts and visual playback | ✅ Done |
| 10 | Optimize and tune training hyperparameters | ⬜ Next |

---

## Success Criteria

1. ✅ Visual game renders correctly with both players' moves
2. ✅ Human can play against AI using mouse input
3. ✅ AI can be trained using the Gymnasium environment
4. ✅ Trained AI consistently beats random agent (>90% win rate)
5. ✅ Training progress is visible via TensorBoard
6. ✅ Can watch trained AI play in real-time visualization

---

## Future Enhancements (Out of Scope)

- Distributed training
- AlphaZero-style MCTS implementation
- Transfer learning to other board games
- Tournament mode with multiple AI versions
