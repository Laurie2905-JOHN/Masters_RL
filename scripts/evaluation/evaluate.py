#!/usr/bin/env python
import gymnasium as gym
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os
import argparse
from utils.process_data import get_data

def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model."""
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    mean_reward = sum(episode_rewards) / num_episodes
    print(f"Mean reward over {num_episodes} episodes: {mean_reward[0]}")
    return mean_reward

def main(args):
    # Setup environment and other configurations
    ingredient_df = get_data()

    # Create vectorized environments using make_vec_env
    env = make_vec_env(args.env_name, n_envs=args.num_envs, env_kwargs={
        'ingredient_df': ingredient_df,
        'render_mode': args.render_mode,
        'num_people': args.num_people,
        'target_calories': args.target_calories
    })
    env = VecNormalize.load(os.path.join(args.save_dir, f"{args.norm_prefix}"), env)

    # Load the trained model
    if args.algo == 'A2C':
        model = A2C.load(os.path.join(args.save_dir, f"{args.save_prefix}"), env=env)
    elif args.algo == 'PPO':
        model = PPO.load(os.path.join(args.save_dir, f"{args.save_prefix}"), env=env)

    # Evaluate the loaded model
    evaluate_model(model, env, num_episodes=args.eval_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on an environment")
    parser.add_argument("--env_name", type=str, default='CalorieOnlyEnv-v1', help="Name of the environment")
    parser.add_argument("--algo", type=str, choices=['A2C', 'PPO'], default='A2C', help="RL algorithm to use (A2C or PPO)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode for the environment")
    parser.add_argument("--num_people", type=int, default=50, help="Number of people in the environment")
    parser.add_argument("--target_calories", type=int, default=530, help="Target calories for the environment")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total number of timesteps for training")
    parser.add_argument("--save_dir", type=str, default="saved_models/best_models", help="Directory to load models and checkpoints from")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for saved model files")
    parser.add_argument("--norm_prefix", type=str, default="vec_normalize.pkl", help="Prefix for normalization statistics")
    parser.add_argument("--eval_episodes", type=int, default=1, help="Number of episodes for evaluation")

    args = parser.parse_args()

    # Set default log and save directories and prefix if not provided
    if args.save_prefix is None:
        args.save_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env"
        
    main(args)
