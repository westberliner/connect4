"""Main controller for the Connect 4 game."""
import argparse
import time

import numpy as np

from GfxEngine import Engine
from Board import Board


class Game:
    """Main game controller that orchestrates board logic and rendering."""
    
    def __init__(self, opponent=None, ai_player=2):
        """
        Initialize the game components.
        
        Args:
            opponent: Optional AI opponent (callable that takes board state, returns action)
            ai_player: Which player the AI controls (1 or 2)
        """
        self.board = Board()
        self.engine = Engine(self.board)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.opponent = opponent
        self.ai_player = ai_player
    
    def reset(self):
        """Reset the game to initial state."""
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None
    
    def switch_player(self):
        """Switch to the other player."""
        self.current_player = 3 - self.current_player  # Toggles between 1 and 2
    
    def make_move(self, column):
        """
        Attempt to make a move in the specified column.
        
        Args:
            column: The column to drop a piece in (0-6)
            
        Returns:
            True if move was successful, False otherwise
        """
        if self.game_over:
            return False
        
        if not self.board.is_valid_move(column):
            return False
        
        # Make the move
        self.board.drop_piece(column, self.current_player)
        
        # Check for game over
        winner = self.board.check_winner()
        if winner is not None:
            self.game_over = True
            self.winner = winner
        elif self.board.is_full():
            self.game_over = True
            self.winner = None  # Draw
        else:
            self.switch_player()
        
        return True
    
    def get_ai_move(self):
        """Get move from AI opponent."""
        if self.opponent is None:
            return None
        
        # Create observation from AI's perspective
        state = self.board.get_state()
        obs = np.zeros_like(state, dtype=np.int8)
        obs[state == self.ai_player] = 1
        obs[state == (3 - self.ai_player)] = -1
        
        info = {
            "valid_moves": self.board.get_valid_moves(),
            "board_state": state.copy()
        }
        
        return self.opponent(obs, info)
    
    def run(self):
        """Main game loop."""
        running = True
        human_player = 3 - self.ai_player if self.opponent else None
        
        while running:
            # Handle input events
            events = self.engine.handle_events()
            
            if events['quit']:
                running = False
                continue
            
            if events['reset']:
                self.reset()
            
            # Handle click
            if events['click_col'] >= 0:
                if self.game_over:
                    # Click to restart after game over
                    self.reset()
                elif self.opponent is None or self.current_player == human_player:
                    # Human's turn (or no AI opponent)
                    self.make_move(events['click_col'])
            
            # AI's turn
            if (self.opponent is not None and 
                not self.game_over and 
                self.current_player == self.ai_player):
                action = self.get_ai_move()
                if action is not None and self.board.is_valid_move(action):
                    self.make_move(action)
                time.sleep(0.3)  # Brief pause so human can see AI move
            
            # Update engine state for hover preview
            self.engine.set_current_player(self.current_player)
            
            # Render
            self.engine.render()
            
            # Show game over screen if needed
            if self.game_over:
                self.engine.render_game_over(self.winner)
            
            # Limit frame rate
            self.engine.tick(60)
        
        self.engine.quit()


def load_model_opponent(model_path):
    """Load a trained model and return it as an opponent callable."""
    from stable_baselines3 import PPO, DQN
    
    # Try PPO first, then DQN
    model = None
    try:
        model = PPO.load(model_path)
    except Exception:
        try:
            model = DQN.load(model_path)
        except Exception:
            raise ValueError(f"Could not load model from {model_path}")
    
    def opponent(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        # Ensure valid action
        valid_moves = info["valid_moves"]
        if action not in valid_moves and valid_moves:
            action = valid_moves[0]
        return action
    
    return opponent


def main():
    """Entry point for the game."""
    parser = argparse.ArgumentParser(description="Play Connect 4")
    
    parser.add_argument(
        "--mode", type=str, default="human",
        choices=["human", "random", "greedy", "minimax", "ai"],
        help="Game mode: human (2 player), random/greedy/minimax (vs rule-based AI), ai (vs trained model)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model (required for --mode ai)"
    )
    parser.add_argument(
        "--depth", type=int, default=4,
        help="Minimax search depth (higher = stronger, default: 4)"
    )
    parser.add_argument(
        "--ai-first", action="store_true",
        help="Let AI play first (as red)"
    )
    
    args = parser.parse_args()
    
    opponent = None
    ai_player = 1 if args.ai_first else 2
    
    if args.mode == "random":
        from Connect4Env import RandomOpponent
        opponent = RandomOpponent()
        print("Playing against Random AI")
    elif args.mode == "greedy":
        from Connect4Env import GreedyOpponent
        opponent = GreedyOpponent()
        print("Playing against Greedy AI")
    elif args.mode == "minimax":
        from Connect4Env import MinimaxOpponent
        opponent = MinimaxOpponent(depth=args.depth)
        print(f"Playing against Minimax AI (depth={args.depth})")
    elif args.mode == "ai":
        if args.model is None:
            print("Error: --model required for --mode ai")
            print("Example: python connect4.py --mode ai --model models/<run>/final_model")
            return
        opponent = load_model_opponent(args.model)
        print(f"Playing against trained model: {args.model}")
    else:
        print("Human vs Human mode")
    
    if opponent:
        print(f"You are {'YELLOW (second)' if args.ai_first else 'RED (first)'}")
    print("R = restart, Q = quit\n")
    
    game = Game(opponent=opponent, ai_player=ai_player)
    game.run()


if __name__ == "__main__":
    main()