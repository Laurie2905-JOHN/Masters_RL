import os
import gymnasium as gym
import numpy as np
import random
from typing import Union, Callable
from stable_baselines3.common.env_util import make_vec_env
import torch
from models.envs.env import *
import os
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import psutil
import time
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from models.wrappers.common import RewardTrackingWrapper
from sb3_contrib.common.wrappers import ActionMasker
from models.action_masks.masks import mask_fn
from torch import nn
import json
import yaml
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import A2C, PPO

# Function to set up the environment
def setup_environment(args, reward_save_path=None, eval=False):
    
    env_kwargs = {
                "ingredient_df": args.ingredient_df,
                "max_ingredients": args.max_ingredients,
                "action_scaling_factor": args.action_scaling_factor,
                "render_mode": args.render_mode,
                "seed": args.seed,
                "verbose": args.verbose,
                "initialization_strategy": args.initialization_strategy,
                "reward_type": args.reward_type,
                "algo": args.algo,
                "gamma": args.gamma,
                "max_episode_steps": args.max_episode_steps
                }
        
    def make_env():
        
        env = gym.make(args.env_name, **env_kwargs)
        
        # Apply the RewardTrackingWrapper if needed
        if args.plot_reward_history:
            if reward_save_path is None:
                raise ValueError("reward_save_path must be specified when plot_reward_history is True")
            env = RewardTrackingWrapper(
                env,
                args.reward_save_interval,
                reward_save_path,
                )
            
        if args.algo == "MASKED_PPO":
            env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        
        # # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode. Need to repeat this so if i want to experiment with different steps.
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = Monitor(env)  # wrap it with monitor env again to explicitely take the change into account
  
        return env

    env = make_vec_env(make_env,
                       n_envs=args.num_envs,
                       seed=args.seed,
                       )

    if eval:
        pass
    else:
        env = VecNormalize(
            env, 
            norm_obs=args.vecnorm_norm_obs, 
            norm_reward=args.vecnorm_norm_reward, 
            clip_obs=args.vecnorm_clip_obs, 
            clip_reward=args.vecnorm_clip_reward, 
            gamma=args.gamma, 
            epsilon=args.vecnorm_epsilon, 
            norm_obs_keys=args.vecnorm_norm_obs_keys
        )
    
    return env





def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

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

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
# Ensure the specified directory exists, creating it if necessary
def ensure_dir_exists(directory, verbose=0):
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose > 1:
            print(f"Warning: Directory {directory} did not exist and was created.")

# Load a pretrained model if a checkpoint is specified
def load_model(args, env, tensorboard_log_dir, seed):
    checkpoint_path = os.path.abspath(args.pretrained_checkpoint_path)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    steps = checkpoint_name.split("_")[-2]
    pkl_name = f"{'_'.join(checkpoint_name.split('_')[:-2])}_vecnormalize_{steps}_steps.pkl"
    pkl_path = os.path.join(checkpoint_dir, pkl_name)

    if os.path.exists(pkl_path):
        print(f"Loading VecNormalize from: {pkl_path}")
        env = VecNormalize.load(pkl_path, env)
    else:
        print(f"VecNormalize file does not exist: {pkl_path}")
        return None

    model = {
        'A2C': A2C.load,
        'PPO': PPO.load
    }.get(args.algo, lambda *args, **kwargs: None)(checkpoint_path, env=env, verbose=args.verbose, tensorboard_log=tensorboard_log_dir, device=args.device, seed=seed)

    if model is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    return model

# Create a new model with the specified algorithm and hyperparameters
def create_model(args, env, tensorboard_log_dir, seed):
    common_hyperparams = {
        'verbose': args.verbose,
        'tensorboard_log': tensorboard_log_dir,
        'device': args.device,
        'seed': seed,
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'normalize_advantage': args.normalize_advantage,
        'gae_lambda': args.gae_lambda,
        'policy_kwargs': args.policy_kwargs,
    }

    if args.algo == 'A2C' :
        if args.env_name == 'SchoolMealSelection-v0':
            algo_hyperparams = {
                'n_steps': args.n_steps,
                'rms_prop_eps': args.rms_prop_eps,
                'use_rms_prop': args.use_rms_prop,
                'use_sde': args.use_sde,
                'sde_sample_freq': args.sde_sample_freq,
            }
        else:
            algo_hyperparams = {
                'n_steps': args.n_steps,
                'rms_prop_eps': args.rms_prop_eps,
                'use_rms_prop': args.use_rms_prop,
            }
    elif args.algo == 'PPO':
        if args.env_name == 'SchoolMealSelection-v0':
            algo_hyperparams = {
                'n_steps': args.n_steps,
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'clip_range': args.clip_range,
                'clip_range_vf': args.clip_range_vf,
                'target_kl': args.target_kl,
                'use_sde': args.use_sde,
                'sde_sample_freq': args.sde_sample_freq,
            }
        else:
            algo_hyperparams = {
                'n_steps': args.n_steps,
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'clip_range': args.clip_range,
                'clip_range_vf': args.clip_range_vf,
                'target_kl': args.target_kl,
            }
            
    elif args.algo == 'MASKED_PPO':
        algo_hyperparams = {
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'clip_range': args.clip_range,
            'clip_range_vf': args.clip_range_vf,
            'target_kl': args.target_kl,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    model_class = {
        'A2C': A2C,
        'PPO': PPO,
        'MASKED_PPO': MaskablePPO
    }.get(args.algo, None)

    if model_class is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    return model_class('MultiInputPolicy', env, **common_hyperparams, **algo_hyperparams)

# Save hyperparameters to a JSON file
def save_hyperparams(hyperparams, args, seed):
    hyperparams_dir, hyperparams_prefix = get_unique_directory(args.hyperparams_dir, f"{args.hyperparams_prefix}_seed_{seed}_hyperparameters", ".json")
    hyperparams_path = os.path.join(hyperparams_dir, f"{hyperparams_prefix}")
    with open(hyperparams_path, 'w') as f:
        json.dump({k: str(v) for k, v in hyperparams.items()}, f, indent=4)

# Convert a comma-separated string into a list
def str_to_list(value):
    return value.split(',')

# Load hyperparameters from a YAML file
def load_hyperparams(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
def set_default_prefixes(args):
    # Function to set the default prefixes if not provided
    no_name = f"{args.env_name}_{args.algo}_reward_type_{args.reward_type}_{args.total_timesteps}_{args.num_envs}env_NewMASK".replace('-', '_')

    if args.log_prefix is None:
        args.log_prefix = no_name
    if args.reward_prefix is None:
        args.reward_prefix = f"{no_name}_reward_data"
    if args.save_prefix is None:
        args.save_prefix = no_name
    if args.best_prefix is None:
        args.best_prefix = f"{no_name}_env_best"
    if args.hyperparams_prefix is None:
        args.hyperparams_prefix = no_name
    return args

def get_activation_fn(name):
    return {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }.get(name.lower(), nn.ReLU)

    
def generate_random_seeds(n):
    return [random.randint(0, 2**32 - 1) for _ in range(n)]


def get_unique_directory(directory, base_name, file_extension):
    """
    Generate a unique filename with a given extension in the given directory by appending a suffix if necessary.
    """
    base_name = base_name.removesuffix('.zip')  # Ensure base_name does not already end with any specified extensions
    base_name = base_name.removesuffix('.json')
    base_name = base_name.removesuffix('.png')
    base_name = base_name.removesuffix('.pkl')
    unique_filename = os.path.join(directory, base_name + file_extension)
    counter = 1

    while os.path.exists(unique_filename):
        unique_filename = os.path.join(directory, f"{base_name}_({counter}){file_extension}")
        counter += 1

    file_path = os.path.abspath(unique_filename)
    base_path = os.path.dirname(file_path)
    unique_file = os.path.basename(file_path)

    return base_path, unique_file


# Function to set random seeds for reproducibility
def set_seed(seed, device):
    if seed is not None:
        print(f"Using seed: {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

def monitor_memory_usage(interval=5):
    process = psutil.Process(os.getpid())
    while True:
        memory_info = process.memory_info()
        print(f"RSS: {memory_info.rss / (1024 ** 2):.2f} MB, VMS: {memory_info.vms / (1024 ** 2):.2f} MB")
        time.sleep(interval)


def process_and_aggregate(store, start, stop):
    df_rewards = store.select('rewards', start=start, stop=stop)
    reward_history = df_rewards['total_reward'].tolist()

    targets_not_met_counts = Counter()
    flattened_reward_histories = defaultdict(list)

    for column in df_rewards.columns:
        if column == 'total_reward':
            continue
        if column == 'targets_not_met':
            for rewards in df_rewards[column].dropna():
                if isinstance(rewards, str):
                    rewards_list = rewards.split(',')
                    targets_not_met_counts.update(rewards_list)
        else:
            for rewards in df_rewards[column].dropna():
                flattened_reward_histories[column].append(rewards)

    return reward_history, flattened_reward_histories, targets_not_met_counts


def plot_reward_distribution(load_path, save_plot_path=None, chunk_size=10000, start_portion=0.0, end_portion=1.0):
    # Validate the input portions
    if not (0.0 <= start_portion < end_portion <= 1.0):
        raise ValueError("start_portion must be >= 0.0 and < end_portion, and end_portion must be <= 1.0.")
    
# Initialize the necessary containers
    total_reward_history = []
    total_flattened_reward_histories = defaultdict(list)
    total_targets_not_met_counts = Counter()

    # Read the HDF5 file in chunks
    with pd.HDFStore(load_path) as store:
        total_rows = store.get_storer('rewards').nrows

        # Calculate the starting and ending points based on the specified portions
        start_point = int(total_rows * start_portion)
        end_point = int(total_rows * end_portion)

        for start in range(start_point, end_point, chunk_size):
            reward_history, flattened_reward_histories, targets_not_met_counts = process_and_aggregate(store, start, min(start + chunk_size, end_point))

            total_reward_history.extend(reward_history)
            for category, values in flattened_reward_histories.items():
                total_flattened_reward_histories[category].extend(values)
            total_targets_not_met_counts.update(targets_not_met_counts)

    # Determine the layout of the plot
    num_rewards = len(total_flattened_reward_histories) + 2
    col = 7
    row = num_rewards // col
    if num_rewards % col != 0:
        row += 1

    fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 3))
    axes = np.ravel(axes)

    # Calculate the percentage of data displayed
    data_percentage = (end_portion - start_portion) * 100
    
    # Set the figure title
    fig.suptitle(f'Data Percentage Displayed: {data_percentage:.2f}%', fontsize=16)

    # Plot the total reward distribution
    axes[0].hist(total_reward_history, bins=50, alpha=0.75)
    axes[0].set_xlabel('Total reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Total Reward Distribution')

    index = 1
    # Plot the distributions of other reward categories
    for category, values in total_flattened_reward_histories.items():
        axes[index].hist(values, bins=50, alpha=0.75)
        axes[index].set_xlabel(f'{category.replace("_", " ").capitalize()}')
        axes[index].set_ylabel('Frequency')
        index += 1

    del total_flattened_reward_histories

    # Plot the counts of targets not met
    bars = axes[index].bar(
        [target.replace('_', ' ').capitalize() for target in total_targets_not_met_counts.keys()],
        total_targets_not_met_counts.values()
    )
    axes[index].set_xlabel('Targets Not Met')
    axes[index].set_ylabel('Frequency')
    axes[index].set_title('Targets Not Met Frequency')
    axes[index].tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        axes[index].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')

    # Hide any unused subplots
    for ax in axes[num_rewards:]:
        ax.set_visible(False)

    plt.tight_layout()

    # Save or show the plot
    if save_plot_path:
        plt.savefig(save_plot_path)
    else:
        plt.show()

if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000", ".zip")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))


