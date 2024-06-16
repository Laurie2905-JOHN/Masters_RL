import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import random
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch
# from models.envs.env import SchoolMealSelection
from models.envs.env_working import SchoolMealSelection
import os
from models.wrappers.common import RewardTrackingWrapper
import psutil
import time
from collections import Counter, defaultdict
from gymnasium.wrappers import TimeLimit
import mmap
import json
import matplotlib.pyplot as plt

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
def setup_environment(args, seed, ingredient_df, reward_save_interval=2500, reward_save_path=None, eval=False):
    
    def make_env():
        env = SchoolMealSelection(
            ingredient_df=ingredient_df,
            render_mode=args.render_mode,
            reward_metrics=args.reward_metrics,
            verbose=args.verbose
        )
        
        # Apply the RewardTrackingWrapper if needed
        if args.plot_reward_history:
            if reward_save_path is None:
                raise ValueError("reward_save_path must be specified when plot_reward_history is True")
            env = RewardTrackingWrapper(
                env,
                reward_save_interval,
                reward_save_path,
                )
        
        # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        
        return env

    env = make_vec_env(make_env, n_envs=args.num_envs, seed=seed)
    
    if eval:
        return env
    
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    
    
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
            if isinstance(value, (int, float, np.number)):
                self.logger.record(f'info/{key}', value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, np.number)):
                        self.logger.record(f'info/{key}/{sub_key}', sub_value)
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


def plot_reward_distribution(load_path, save_plot_path=None):

    def process_line(line):
        try:
            reward_distribution = json.loads(line)
            return reward_distribution
        except json.JSONDecodeError:
            return None

    reward_history = []
    reward_details_history = []
    termination_reasons = Counter()

    with open(load_path, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        for line in iter(mm.readline, b""):
            reward_distribution = process_line(line.decode('utf-8'))
            if reward_distribution:
                reward_history.extend(reward_distribution['total_reward'])
                reward_details_history.extend(reward_distribution['reward_details'])
                termination_reasons.update(reward_distribution['termination_reasons'])
        mm.close()

    reason_str = termination_reasons

    # Flatten reward details
    flattened_reward_histories = defaultdict(lambda: defaultdict(list))
    targets_not_met = []

    for entry in reward_details_history:
        for category, rewards in entry.items():
            if category == 'targets_not_met':
                targets_not_met.extend(rewards)
            else:
                for key, value in rewards.items():
                    flattened_reward_histories[category][key].append(value)

    # Count occurrences of each string in 'targets_not_met'
    targets_not_met_counts = Counter(targets_not_met)

    # Dictionary for shorter labels
    label_mapping = {
        'nutrient_rewards': 'Nutrient',
        'ingredient_group_count_rewards': 'Ingredient Group',
        'ingredient_environment_count_rewards': 'Environment',
        'cost_rewards': 'Cost',
        'consumption_rewards': 'Consumption',
        'targets_not_met': 'Targets Not Met'
    }

    num_rewards = sum(len(rewards) for rewards in flattened_reward_histories.values()) + 3
    col = 7
    row = num_rewards // col
    if num_rewards % col != 0:
        row += 1

    fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 3))
    axes = np.ravel(axes)

    bars = axes[0].bar(
        [reason.replace('_', ' ').capitalize() for reason in termination_reasons.keys()],
        termination_reasons.values()
    )
    axes[0].set_xlabel('Termination Reason')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Termination Reason Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')

    axes[1].hist(reward_history, bins=50, alpha=0.75)
    axes[1].set_xlabel('Total reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Total Reward Distribution')

    index = 2
    for category, rewards in flattened_reward_histories.items():
        for key, values in rewards.items():
            short_label = label_mapping.get(category, category)
            axes[index].hist(values, bins=50, alpha=0.75)
            axes[index].set_xlabel(f'{short_label} - {key.replace("_", " ").capitalize()}')
            axes[index].set_ylabel('Frequency')
            index += 1

    bars = axes[index].bar(
        [target.replace('_', ' ').capitalize() for target in targets_not_met_counts.keys()],
        targets_not_met_counts.values()
    )
    axes[index].set_xlabel('Targets Not Met')
    axes[index].set_ylabel('Frequency')
    axes[index].set_title('Targets Not Met Frequency')
    axes[index].tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        axes[index].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')

    for ax in axes[num_rewards:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_plot_path:
        plt.savefig(save_plot_path)
    else:
        plt.show()
            
if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000", ".zip")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))
