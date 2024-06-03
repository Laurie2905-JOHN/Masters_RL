#!/usr/bin/env python
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os
import argparse
import random
import torch
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback
from utils.train_utils import get_unique_directory
from utils.train_utils import generate_random_seeds

# Function to select the appropriate device (CPU or GPU)
def select_device(args):
    if args.device == 'auto':
        if args.algo == 'A2C':
            return "cpu"
        else:
            return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return args.device

# Function to set random seeds for reproducibility
def set_seed(seed, device):
    if seed is not None:
        print(f"Using seed: {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

# Function to set up the environment
def setup_environment(args, seed, ingredient_df):
    env = make_vec_env(args.env_name, n_envs=args.num_envs, env_kwargs={
        'ingredient_df': ingredient_df,
        'render_mode': args.render_mode,
        'num_people': args.num_people,
        'target_calories': args.target_calories
    }, seed=seed)
    
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Main training function
def main(args, seed):
    device = select_device(args)
    set_seed(seed, device)

    print(f"Using device: {device}")

    # Load data required for environment setup
    ingredient_df = get_data()

    # Create and normalize vectorized environments
    env = setup_environment(args, seed, ingredient_df)

    # Set up TensorBoard logging directory
    tensorboard_log_dir = os.path.join(args.log_dir, f"{args.log_prefix}_seed{seed}")

    # Create unique directories for saving models
    save_dir, save_prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed{seed}")
    best_dir, best_prefix = get_unique_directory(args.best_dir, f"{args.best_prefix}_seed{seed}")
    best_model_path = os.path.join(best_dir, best_prefix)
    
    reward_dir, reward_prefix = get_unique_directory(args.reward_dir, f"{args.reward_dir}_seed{seed}")
    
    # Configure logger for TensorBoard and stdout
    new_logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])

    # Set up callbacks for saving checkpoints, evaluating models, and logging additional information
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=save_dir, name_prefix=save_prefix)
    eval_callback = EvalCallback(env, best_model_save_path=best_model_path, eval_freq=args.eval_freq, log_path=tensorboard_log_dir, deterministic=True, render=False)
    info_logger_callback = InfoLoggerCallback()
    callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

    # Choose the RL algorithm (A2C or PPO)
    if args.algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
    elif args.algo == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # Set the logger for the model and start training
    model.set_logger(new_logger)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Save the final model and VecNormalize statistics
    final_save = os.path.join(save_dir, f"{save_prefix}_final.zip")
    model.save(final_save)
    final_vec_normalize = os.path.join(save_dir, f"{save_prefix}_vec_normalize_best.pkl")
    env.save(final_vec_normalize)
    
    # Save the reward distribution to a JSON file
    reward_distribution_path = os.path.join(reward_dir, f"{reward_prefix}_json")
    env.save_reward_distribution(reward_distribution_path)

    del model

    # Load the model and VecNormalize to verify they have saved correctly
    try:
        env = setup_environment(args, seed, ingredient_df)
        env = VecNormalize.load(final_vec_normalize, env)

        if args.algo == 'A2C':
            model = A2C.load(final_save, env=env)
        elif args.algo == 'PPO':
            model = PPO.load(final_save, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--env_name", type=str, default='CalorieOnlyEnv-v1', help="Name of the environment")
    parser.add_argument("--algo", type=str, choices=['A2C', 'PPO'], default='A2C', help="RL algorithm to use (A2C or PPO)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode for the environment")
    parser.add_argument("--num_people", type=int, default=50, help="Number of people in the environment")
    parser.add_argument("--target_calories", type=int, default=530, help="Target calories for the environment")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total number of timesteps for training")
    parser.add_argument("--log_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'tensorboard')), help="Directory for tensorboard logs")
    parser.add_argument("--log_prefix", type=str, default=None, help="Filename for tensorboard logs")
    parser.add_argument("--save_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'checkpoints')), help="Directory to save models and checkpoints")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for saved model files")
    parser.add_argument("--reward_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'reward')), help="Directory to save reward data")
    parser.add_argument("--reward_prefix", type=str, default=None, help="Prefix for saved reward data")
    parser.add_argument("--best_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'best_models')), help="Directory for best model")
    parser.add_argument("--best_prefix", type=str, default=None, help="Prefix for saving best model")
    parser.add_argument("--save_freq", type=int, default=1000, help="Frequency of saving checkpoints")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency of evaluations")
    parser.add_argument("--seed", type=int, nargs='+', default=generate_random_seeds(3), help="Random seed for the environment (use -1 for random, or multiple values for multiple seeds)")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device to use for training (cpu, cuda, or auto)")

    args = parser.parse_args()

    # Set default log and save directories and prefix if not provided
    if args.log_prefix is None:
        args.log_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env"
    if args.save_prefix is None:
        args.save_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env"
    if args.reward_prefix is None:
        args.reward_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}_reward_data"
    if args.best_prefix is None:
        args.best_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env_best"
    
    for seed in args.seed:
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        main(args, seed)
