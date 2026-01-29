"""Gymnasium environment wrapper for Connect 4."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Board import Board


class Connect4Env(gym.Env):
    """
    Connect 4 environment compatible with Gymnasium.
    
    Observation:
        - Board state as 6x7 numpy array (or RGB frame if render_mode='rgb_array')
        - Values: 0 (empty), 1 (current player), -1 (opponent)
        
    Actions:
        - Discrete(7): Column to drop piece (0-6)
        
    Rewards:
        - Win: +1.0
        - Loss: -1.0
        - Draw: 0.0
        - Invalid move: -0.5 (terminates episode)
        - Step: -0.01 (small penalty to encourage faster wins)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, opponent=None):
        """
        Initialize the Connect 4 environment.
        
        Args:
            render_mode: 'human' for Pygame window, 'rgb_array' for frame capture
            opponent: Optional opponent agent (callable that takes board state, returns action)
                     If None, environment expects external control of both players
        """
        super().__init__()
        
        self.board = Board()
        self.render_mode = render_mode
        self.opponent = opponent
        self.current_player = 1  # 1 = agent, 2 = opponent
        
        # Action space: 7 columns
        self.action_space = spaces.Discrete(7)
        
        # Observation space: 6x7 board with values -1, 0, 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6, 7), dtype=np.int8
        )
        
        # Graphics engine (lazy initialization)
        self._engine = None
    
    @property
    def engine(self):
        """Lazy initialization of graphics engine."""
        if self._engine is None and self.render_mode is not None:
            from GfxEngine import Engine
            self._engine = Engine(self.board, window_title="Connect 4 - AI Training")
        return self._engine
    
    def _get_obs(self):
        """
        Get observation from current player's perspective.
        
        Returns:
            Board state where 1 = current player's pieces, -1 = opponent's pieces
        """
        state = self.board.get_state().astype(np.int8)
        # Convert to current player's perspective
        obs = np.zeros_like(state)
        obs[state == self.current_player] = 1
        obs[state == (3 - self.current_player)] = -1
        return obs
    
    def _get_info(self):
        """Get additional info about the game state."""
        return {
            "valid_moves": self.board.get_valid_moves(),
            "current_player": self.current_player,
            "board_state": self.board.get_state().copy()
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial board state
            info: Additional information
        """
        super().reset(seed=seed)
        
        self.board.reset()
        self.current_player = 1
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Column to drop piece (0-6)
            
        Returns:
            observation: New board state
            reward: Reward for this step
            terminated: True if game is over
            truncated: Always False (no time limit)
            info: Additional information
        """
        # Check for invalid move
        if not self.board.is_valid_move(action):
            return self._get_obs(), -0.5, True, False, self._get_info()
        
        # Make the agent's move
        self.board.drop_piece(action, self.current_player)
        
        # Check if agent won
        winner = self.board.check_winner()
        if winner == self.current_player:
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), 1.0, True, False, self._get_info()
        
        # Check for draw
        if self.board.is_full():
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        # Opponent's turn
        if self.opponent is not None:
            opponent_player = 3 - self.current_player
            
            # Get opponent's action
            opponent_obs = self._get_obs_for_player(opponent_player)
            opponent_action = self.opponent(opponent_obs, self._get_info())
            
            # Make opponent's move (if valid)
            if self.board.is_valid_move(opponent_action):
                self.board.drop_piece(opponent_action, opponent_player)
                
                # Check if opponent won
                winner = self.board.check_winner()
                if winner == opponent_player:
                    if self.render_mode == "human":
                        self.render()
                    return self._get_obs(), -1.0, True, False, self._get_info()
                
                # Check for draw after opponent's move
                if self.board.is_full():
                    if self.render_mode == "human":
                        self.render()
                    return self._get_obs(), 0.0, True, False, self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        # Small step penalty to encourage faster wins
        return self._get_obs(), -0.01, False, False, self._get_info()
    
    def _get_obs_for_player(self, player):
        """Get observation from a specific player's perspective."""
        state = self.board.get_state().astype(np.int8)
        obs = np.zeros_like(state)
        obs[state == player] = 1
        obs[state == (3 - player)] = -1
        return obs
    
    def render(self):
        """Render the current game state."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == "human":
            if self.engine:
                self.engine.set_current_player(self.current_player)
                self.engine.render()
                self.engine.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            if self.engine:
                self.engine.render()
                return self.engine.get_frame()
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
    
    def action_masks(self):
        """
        Return a mask of valid actions.
        
        Returns:
            numpy array of shape (7,) with True for valid actions
        """
        return np.array([self.board.is_valid_move(i) for i in range(7)], dtype=bool)


class RandomOpponent:
    """Random agent that picks a valid move uniformly at random."""
    
    def __call__(self, obs, info):
        valid_moves = info["valid_moves"]
        if valid_moves:
            return np.random.choice(valid_moves)
        return 0


class GreedyOpponent:
    """
    Greedy agent that:
    1. Wins if possible
    2. Blocks opponent's winning move
    3. Otherwise picks randomly
    """
    
    def __call__(self, obs, info):
        valid_moves = info["valid_moves"]
        board_state = info["board_state"]
        
        if not valid_moves:
            return 0
        
        # Create a temporary board for simulation
        temp_board = Board()
        temp_board.grid = board_state.copy()
        
        # Find which player we are (the one with fewer or equal pieces)
        p1_count = np.sum(board_state == 1)
        p2_count = np.sum(board_state == 2)
        our_player = 2 if p2_count <= p1_count else 1
        opp_player = 3 - our_player
        
        # Check if we can win
        for col in valid_moves:
            test_board = Board()
            test_board.grid = board_state.copy()
            test_board.drop_piece(col, our_player)
            if test_board.check_winner() == our_player:
                return col
        
        # Check if we need to block
        for col in valid_moves:
            test_board = Board()
            test_board.grid = board_state.copy()
            test_board.drop_piece(col, opp_player)
            if test_board.check_winner() == opp_player:
                return col
        
        # Otherwise random
        return np.random.choice(valid_moves)


class MinimaxOpponent:
    """
    Minimax agent with alpha-beta pruning.
    Looks ahead multiple moves for stronger play.
    """
    
    def __init__(self, depth=4):
        """
        Args:
            depth: How many moves to look ahead (higher = stronger but slower)
                   Recommended: 4-6 for reasonable speed
        """
        self.depth = depth
    
    def __call__(self, obs, info):
        valid_moves = info["valid_moves"]
        board_state = info["board_state"]
        
        if not valid_moves:
            return 0
        
        # Find which player we are
        p1_count = np.sum(board_state == 1)
        p2_count = np.sum(board_state == 2)
        our_player = 2 if p2_count <= p1_count else 1
        
        best_score = float('-inf')
        best_col = valid_moves[0]
        
        # Evaluate each possible move
        for col in valid_moves:
            test_board = Board()
            test_board.grid = board_state.copy()
            test_board.drop_piece(col, our_player)
            
            score = self._minimax(test_board, self.depth - 1, float('-inf'), 
                                  float('inf'), False, our_player)
            
            if score > best_score:
                best_score = score
                best_col = col
        
        return best_col
    
    def _minimax(self, board, depth, alpha, beta, is_maximizing, our_player):
        """Minimax with alpha-beta pruning."""
        opp_player = 3 - our_player
        
        # Check terminal states
        winner = board.check_winner()
        if winner == our_player:
            return 100 + depth  # Prefer faster wins
        elif winner == opp_player:
            return -100 - depth  # Prefer slower losses
        elif board.is_full() or depth == 0:
            return self._evaluate_position(board, our_player)
        
        valid_moves = board.get_valid_moves()
        
        # Prefer center columns (better strategy)
        valid_moves = sorted(valid_moves, key=lambda x: abs(x - 3))
        
        if is_maximizing:
            max_eval = float('-inf')
            for col in valid_moves:
                new_board = board.copy()
                new_board.drop_piece(col, our_player)
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, 
                                           False, our_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for col in valid_moves:
                new_board = board.copy()
                new_board.drop_piece(col, opp_player)
                eval_score = self._minimax(new_board, depth - 1, alpha, beta, 
                                           True, our_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_position(self, board, our_player):
        """
        Evaluate board position heuristically.
        Counts potential winning lines and center control.
        """
        opp_player = 3 - our_player
        score = 0
        
        # Center column preference
        center_col = 3
        center_count = np.sum(board.grid[:, center_col] == our_player)
        score += center_count * 3
        
        # Evaluate all windows of 4
        # Horizontal
        for row in range(board.ROWS):
            for col in range(board.COLS - 3):
                window = board.grid[row, col:col+4]
                score += self._evaluate_window(window, our_player, opp_player)
        
        # Vertical
        for row in range(board.ROWS - 3):
            for col in range(board.COLS):
                window = board.grid[row:row+4, col]
                score += self._evaluate_window(window, our_player, opp_player)
        
        # Diagonal (positive slope)
        for row in range(3, board.ROWS):
            for col in range(board.COLS - 3):
                window = [board.grid[row-i, col+i] for i in range(4)]
                score += self._evaluate_window(window, our_player, opp_player)
        
        # Diagonal (negative slope)
        for row in range(board.ROWS - 3):
            for col in range(board.COLS - 3):
                window = [board.grid[row+i, col+i] for i in range(4)]
                score += self._evaluate_window(window, our_player, opp_player)
        
        return score
    
    def _evaluate_window(self, window, our_player, opp_player):
        """Evaluate a window of 4 cells."""
        window = list(window)
        score = 0
        
        our_count = window.count(our_player)
        opp_count = window.count(opp_player)
        empty_count = window.count(0)
        
        if our_count == 4:
            score += 100
        elif our_count == 3 and empty_count == 1:
            score += 5
        elif our_count == 2 and empty_count == 2:
            score += 2
        
        if opp_count == 3 and empty_count == 1:
            score -= 4  # Block opponent's threats
        
        return score


# Register the environment with Gymnasium
def register_env():
    """Register Connect4 environment with Gymnasium."""
    gym.register(
        id="Connect4-v0",
        entry_point="Connect4Env:Connect4Env",
    )
