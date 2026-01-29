"""Training script for Connect 4 AI using Stable-Baselines3."""
import os
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from Connect4Env import Connect4Env, RandomOpponent, GreedyOpponent, MinimaxOpponent


def make_env(opponent_type="random", render_mode=None, minimax_depth=4):
    """Factory function to create environment instances."""
    def _init():
        if opponent_type == "random":
            opponent = RandomOpponent()
        elif opponent_type == "greedy":
            opponent = GreedyOpponent()
        elif opponent_type == "minimax":
            opponent = MinimaxOpponent(depth=minimax_depth)
        else:
            opponent = None
        
        env = Connect4Env(render_mode=render_mode, opponent=opponent)
        env = Monitor(env)
        return env
    return _init


def evaluate_against_opponent(model, opponent_type, n_games=100):
    """
    Evaluate the model against a specific opponent.
    
    Returns:
        win_rate: Percentage of games won
        draw_rate: Percentage of games drawn
        loss_rate: Percentage of games lost
    """
    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "greedy":
        opponent = GreedyOpponent()
    elif opponent_type == "minimax":
        opponent = MinimaxOpponent(depth=4)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    env = Connect4Env(opponent=opponent)
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(n_games):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Use action masking for valid moves
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True)
            
            # Ensure action is valid
            if not action_masks[action]:
                valid_actions = np.where(action_masks)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
            
            obs, reward, done, truncated, info = env.step(action)
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    
    env.close()
    
    return wins / n_games, draws / n_games, losses / n_games


def train(args):
    """Main training function."""
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algorithm}_{args.opponent}_{timestamp}"
    
    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training {args.algorithm} against {args.opponent} opponent")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    print("-" * 50)
    
    # Create vectorized environment for parallel training
    minimax_depth = getattr(args, 'minimax_depth', 4)
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(args.opponent, minimax_depth=minimax_depth) 
                            for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(args.opponent, minimax_depth=minimax_depth)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.opponent, minimax_depth=minimax_depth)])
    
    # Select algorithm
    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir,
        )
    elif args.algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_dir,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix="connect4"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    for opp_type in ["random", "greedy", "minimax"]:
        win_rate, draw_rate, loss_rate = evaluate_against_opponent(
            model, opp_type, n_games=100
        )
        print(f"vs {opp_type:8s}: Win {win_rate*100:5.1f}% | "
              f"Draw {draw_rate*100:5.1f}% | Loss {loss_rate*100:5.1f}%")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Connect 4 AI")
    
    parser.add_argument(
        "--algorithm", type=str, default="PPO",
        choices=["PPO", "DQN"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--opponent", type=str, default="random",
        choices=["random", "greedy", "minimax"],
        help="Opponent type to train against"
    )
    parser.add_argument(
        "--minimax-depth", type=int, default=4,
        help="Minimax search depth (higher = stronger, slower)"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=100000,
        help="Total timesteps to train"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5000,
        help="Evaluate every N timesteps"
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
