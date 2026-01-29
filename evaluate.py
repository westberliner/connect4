"""Evaluation and visual playback for trained Connect 4 models."""
import argparse
import time
import os

import numpy as np
from stable_baselines3 import PPO, DQN

from Connect4Env import Connect4Env, RandomOpponent, GreedyOpponent, MinimaxOpponent
from Board import Board


def load_model(model_path):
    """Load a trained model from file."""
    # Try PPO first, then DQN
    try:
        return PPO.load(model_path)
    except Exception:
        pass
    
    try:
        return DQN.load(model_path)
    except Exception:
        pass
    
    raise ValueError(f"Could not load model from {model_path}")


def get_ai_action(model, obs, env):
    """Get action from model, ensuring it's valid."""
    action_masks = env.action_masks()
    action, _ = model.predict(obs, deterministic=True)
    
    # Ensure action is valid
    if not action_masks[action]:
        valid_actions = np.where(action_masks)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
    
    return action


def watch_ai_vs_opponent(model_path, opponent_type="random", n_games=5, delay=0.5, depth=4):
    """
    Watch the AI play against an opponent with visual display.
    
    Args:
        model_path: Path to the trained model
        opponent_type: 'random', 'greedy', or 'minimax'
        n_games: Number of games to play
        delay: Delay between moves (seconds)
        depth: Minimax search depth (if using minimax)
    """
    model = load_model(model_path)
    
    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "greedy":
        opponent = GreedyOpponent()
    elif opponent_type == "minimax":
        opponent = MinimaxOpponent(depth=depth)
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")
    
    env = Connect4Env(render_mode="human", opponent=opponent)
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nWatching AI vs {opponent_type} opponent")
    print(f"Playing {n_games} games with {delay}s delay between moves")
    print("Press Q to quit, R to restart current game\n")
    
    for game_num in range(n_games):
        print(f"Game {game_num + 1}/{n_games}")
        obs, info = env.reset()
        done = False
        move_count = 0
        
        while not done:
            # Get AI's move
            action = get_ai_action(model, obs, env)
            obs, reward, done, truncated, info = env.step(action)
            move_count += 1
            
            # Small delay to watch the game
            time.sleep(delay)
            
            # Check for quit events
            if env.engine:
                events = env.engine.handle_events()
                if events['quit']:
                    env.close()
                    print(f"\nFinal: {wins}W / {draws}D / {losses}L")
                    return
        
        # Record result
        if reward > 0:
            wins += 1
            result = "WIN"
        elif reward < 0:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        print(f"  Result: {result} in {move_count} moves")
        
        # Pause at end of game
        time.sleep(1.0)
    
    env.close()
    
    print(f"\nFinal Results vs {opponent_type}:")
    print(f"  Wins:   {wins}/{n_games} ({wins/n_games*100:.1f}%)")
    print(f"  Draws:  {draws}/{n_games} ({draws/n_games*100:.1f}%)")
    print(f"  Losses: {losses}/{n_games} ({losses/n_games*100:.1f}%)")


def watch_ai_vs_ai(model1_path, model2_path=None, n_games=5, delay=0.5):
    """
    Watch two AI models play against each other.
    
    Args:
        model1_path: Path to first model (plays as Player 1)
        model2_path: Path to second model (or None to use same model)
        n_games: Number of games to play
        delay: Delay between moves (seconds)
    """
    model1 = load_model(model1_path)
    model2 = load_model(model2_path) if model2_path else model1
    
    # Create environment without opponent (we'll control both)
    env = Connect4Env(render_mode="human", opponent=None)
    
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    print(f"\nWatching AI vs AI")
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path or 'same as Model 1'}")
    print(f"Playing {n_games} games\n")
    
    for game_num in range(n_games):
        print(f"Game {game_num + 1}/{n_games}")
        obs, info = env.reset()
        done = False
        current_player = 1
        
        while not done:
            # Get current model's action
            if current_player == 1:
                action = get_ai_action(model1, obs, env)
            else:
                # Flip observation for model2's perspective
                action = get_ai_action(model2, -obs, env)
            
            obs, reward, done, truncated, info = env.step(action)
            current_player = 3 - current_player
            
            time.sleep(delay)
            
            if env.engine:
                events = env.engine.handle_events()
                if events['quit']:
                    env.close()
                    return
        
        # Determine winner
        winner = env.board.check_winner()
        if winner == 1:
            model1_wins += 1
            print("  Model 1 wins!")
        elif winner == 2:
            model2_wins += 1
            print("  Model 2 wins!")
        else:
            draws += 1
            print("  Draw!")
        
        time.sleep(1.0)
    
    env.close()
    
    print(f"\nResults:")
    print(f"  Model 1: {model1_wins} wins")
    print(f"  Model 2: {model2_wins} wins")
    print(f"  Draws:   {draws}")


def play_vs_ai(model_path, human_first=True):
    """
    Play against the trained AI.
    
    Args:
        model_path: Path to the trained model
        human_first: If True, human plays first (red)
    """
    model = load_model(model_path)
    
    # Import here to avoid circular imports
    from GfxEngine import Engine
    
    board = Board()
    engine = Engine(board, window_title="Connect 4 - You vs AI")
    
    human_player = 1 if human_first else 2
    ai_player = 2 if human_first else 1
    current_player = 1
    game_over = False
    winner = None
    
    print(f"\nPlaying against AI")
    print(f"You are {'RED (first)' if human_first else 'YELLOW (second)'}")
    print("Click to drop a piece, R to restart, Q to quit\n")
    
    running = True
    while running:
        events = engine.handle_events()
        
        if events['quit']:
            running = False
            continue
        
        if events['reset']:
            board.reset()
            current_player = 1
            game_over = False
            winner = None
        
        if not game_over:
            if current_player == human_player:
                # Human's turn
                if events['click_col'] >= 0:
                    if board.is_valid_move(events['click_col']):
                        board.drop_piece(events['click_col'], human_player)
                        winner = board.check_winner()
                        if winner or board.is_full():
                            game_over = True
                        else:
                            current_player = ai_player
            else:
                # AI's turn
                # Create observation from AI's perspective
                state = board.get_state().astype(np.int8)
                obs = np.zeros_like(state)
                obs[state == ai_player] = 1
                obs[state == human_player] = -1
                
                # Get AI action
                action_masks = np.array([board.is_valid_move(i) for i in range(7)])
                action, _ = model.predict(obs, deterministic=True)
                
                if not action_masks[action]:
                    valid = np.where(action_masks)[0]
                    action = valid[0] if len(valid) > 0 else 0
                
                board.drop_piece(action, ai_player)
                winner = board.check_winner()
                if winner or board.is_full():
                    game_over = True
                else:
                    current_player = human_player
                
                time.sleep(0.3)  # Brief pause for AI move
        
        engine.set_current_player(current_player)
        engine.render()
        
        if game_over:
            engine.render_game_over(winner)
        
        engine.tick(60)
    
    engine.quit()
    
    if winner == human_player:
        print("You win!")
    elif winner == ai_player:
        print("AI wins!")
    else:
        print("Draw!")


def benchmark(model_path, n_games=1000):
    """
    Run a comprehensive benchmark of the model.
    
    Args:
        model_path: Path to the trained model
        n_games: Number of games per opponent
    """
    model = load_model(model_path)
    
    print(f"\nBenchmarking model: {model_path}")
    print(f"Games per opponent: {n_games}")
    print("-" * 50)
    
    for opponent_type in ["random", "greedy", "minimax"]:
        if opponent_type == "random":
            opponent = RandomOpponent()
        elif opponent_type == "greedy":
            opponent = GreedyOpponent()
        else:
            opponent = MinimaxOpponent(depth=4)
        
        env = Connect4Env(opponent=opponent)
        
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        
        for _ in range(n_games):
            obs, info = env.reset()
            done = False
            moves = 0
            
            while not done:
                action = get_ai_action(model, obs, env)
                obs, reward, done, truncated, info = env.step(action)
                moves += 1
            
            total_moves += moves
            
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
        
        env.close()
        
        print(f"\nvs {opponent_type}:")
        print(f"  Win rate:  {wins/n_games*100:6.2f}%")
        print(f"  Draw rate: {draws/n_games*100:6.2f}%")
        print(f"  Loss rate: {losses/n_games*100:6.2f}%")
        print(f"  Avg moves: {total_moves/n_games:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and play Connect 4 AI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Watch AI vs opponent
    watch_parser = subparsers.add_parser("watch", help="Watch AI play against opponent")
    watch_parser.add_argument("model", help="Path to trained model")
    watch_parser.add_argument("--opponent", type=str, default="random",
                              choices=["random", "greedy", "minimax"])
    watch_parser.add_argument("--games", type=int, default=5)
    watch_parser.add_argument("--delay", type=float, default=0.5)
    watch_parser.add_argument("--depth", type=int, default=4,
                              help="Minimax search depth")
    
    # Play against AI
    play_parser = subparsers.add_parser("play", help="Play against the AI")
    play_parser.add_argument("model", help="Path to trained model")
    play_parser.add_argument("--ai-first", action="store_true",
                             help="Let AI play first")
    
    # AI vs AI
    versus_parser = subparsers.add_parser("versus", help="Watch two AIs play")
    versus_parser.add_argument("model1", help="Path to first model")
    versus_parser.add_argument("--model2", help="Path to second model (optional)")
    versus_parser.add_argument("--games", type=int, default=5)
    versus_parser.add_argument("--delay", type=float, default=0.5)
    
    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    bench_parser.add_argument("model", help="Path to trained model")
    bench_parser.add_argument("--games", type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.command == "watch":
        depth = getattr(args, 'depth', 4)
        watch_ai_vs_opponent(args.model, args.opponent, args.games, args.delay, depth)
    elif args.command == "play":
        play_vs_ai(args.model, human_first=not args.ai_first)
    elif args.command == "versus":
        watch_ai_vs_ai(args.model1, args.model2, args.games, args.delay)
    elif args.command == "benchmark":
        benchmark(args.model, args.games)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
