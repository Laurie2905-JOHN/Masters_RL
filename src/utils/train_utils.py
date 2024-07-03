import os
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import random
from typing import Union, Callable
from stable_baselines3.common.env_util import make_vec_env
import torch
# from models.envs.env import SchoolMealSelection
from models.envs.env_working import SchoolMealSelection
from models.envs.env import *
import os
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import psutil
import time
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from models.wrappers.common import RewardTrackingWrapper

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


# Function to set up the environment
def setup_environment(args, reward_save_path=None, eval=False):
    
    env_kwargs = {
                "ingredient_df": args.ingredient_df,
                "max_ingredients": args.max_ingredients,
                "action_scaling_factor": args.action_scaling_factor,
                "render_mode": args.render_mode,
                "seed": args.seed,
                "verbose": args.verbose,
                "initialization_strategy": args.initialization_strategy
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
        
        # # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode. Need to repeat this so if i want to experiment with different steps.
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = Monitor(env)  # wrap it with monitor env again to explicitely take the change into account
  
        return env

    env = make_vec_env(make_env, n_envs=args.num_envs, seed=args.seed)

    if eval:
        return env

    return VecNormalize(
        env, 
        norm_obs=args.vecnorm_norm_obs, 
        norm_reward=args.vecnorm_norm_reward, 
        clip_obs=args.vecnorm_clip_obs, 
        clip_reward=args.vecnorm_clip_reward, 
        gamma=args.gamma, 
        epsilon=args.vecnorm_epsilon, 
        norm_obs_keys=args.vecnorm_norm_obs_keys
    )

    
    
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



def run_episodes(env, num_episodes, steps_per_episode):
    successful_terminations = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0

        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _, info = env.step(action)  # Take a step in the environment

            episode_reward += reward

            if done:
                if "successful" in info.get("termination_reason", ""):
                    successful_terminations += 1
                break

        total_rewards.append(episode_reward)

    return successful_terminations, total_rewards

def optimize_scaling_factor(ingredient_df, num_episodes, steps_per_episode, scale_factors):
    best_scale_factor = None
    max_successful_terminations = 0
    scale_factor_results = {}

    for scale_factor in scale_factors:
        env = SchoolMealSelection(ingredient_df=ingredient_df, action_scaling_factor=scale_factor)
        successful_terminations, total_rewards = run_episodes(env, num_episodes, steps_per_episode)
        scale_factor_results[scale_factor] = {
            "successful_terminations": successful_terminations,
            "total_rewards": total_rewards
        }
        print(f"Scale Factor: {scale_factor}, Successful Terminations: {successful_terminations}")

        if successful_terminations > max_successful_terminations:
            max_successful_terminations = successful_terminations
            best_scale_factor = scale_factor

        env.close()

    return best_scale_factor, max_successful_terminations, scale_factor_results


# Function to set random seeds for reproducibility
def set_seed(seed, device):
    if seed is not None:
        print(f"Using seed: {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)


        
class InfoLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(InfoLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[0]
        for key, value in info.items():
            if key == 'current_meal_plan':
                continue
            if isinstance(value, (int, float, np.number)):
                self.logger.record(f'info/{key}', value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, np.number)):
                        self.logger.record(f'info/{key}/{sub_key}', sub_value)
                    elif isinstance(sub_value, dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            if isinstance(sub_sub_value, (int, float, np.number)):
                                self.logger.record(f'info/{key}/{sub_key}/{sub_sub_key}', sub_sub_value)
                                
        return True


class SaveVecNormalizeEvalCallback(BaseCallback):
    def __init__(self, save_path, vec_normalize_env, verbose=0):
        super(SaveVecNormalizeEvalCallback, self).__init__(verbose)
        self.save_path = save_path
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self) -> bool:
        # Save the VecNormalize statistics
        if self.vec_normalize_env is not None:
            save_path = os.path.join(self.save_path, 'vec_normalize_best.pkl')
            self.vec_normalize_env.save(save_path)
            if self.verbose > 0:
                print(f"Saved VecNormalize to {save_path}")
        return True

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





            
if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000", ".zip")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))
