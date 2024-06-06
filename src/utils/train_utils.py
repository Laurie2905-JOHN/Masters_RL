import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import random
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch
from models.envs.env import SchoolMealSelection
# Import your reward functions
from models.reward.reward import reward_nutrient_macro, reward_nutrient_macro_and_groups
from models.wrappers.common import RewardTrackingWrapper

# Mapping from reward function names to actual functions
REWARD_FUNCTIONS = {
    'reward_nutrient_macro': reward_nutrient_macro,
    'reward_nutrient_macro_and_groups': reward_nutrient_macro_and_groups,
    # Add more mappings as needed
}

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
    reward_func = REWARD_FUNCTIONS.get(args.reward_func, reward_nutrient_macro)
    
    def make_env():
        env = SchoolMealSelection(
            ingredient_df=ingredient_df,
            reward_func=reward_func,
            save_reward=args.plot_reward_history,
            render_mode=args.render_mode
        )
        if args.plot_reward_history:
            env = RewardTrackingWrapper(env, save_reward=True)
        return env

    env = make_vec_env(make_env, n_envs=args.num_envs, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    
    
def generate_random_seeds(n):
    return [random.randint(0, 2**32 - 1) for _ in range(n)]

def get_unique_directory(directory, base_name):
    """
    Generate a unique directory name in the given directory by appending a suffix if necessary.
    """
    unique_dir = os.path.join(directory, base_name)
    counter = 1

    while os.path.exists(unique_dir):
        unique_dir = os.path.join(directory, f"{base_name} ({counter})")
        counter += 1
    
    dir_path = os.path.abspath(unique_dir)
    base_path = os.path.dirname(dir_path)
    unique_subdir = os.path.basename(dir_path)

    return base_path, unique_subdir

def get_unique_image_directory(prefix):
    
    if len(prefix.split('.')[1].split(' ')) > 1:
        prefix = prefix.split('.')[0] + ' ' + prefix.split('.')[1].split(' ')[1] + '.png'
    else:
        prefix = prefix.split('.')[0] + '.png'
        
    return prefix

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

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, vec_normalize_env, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save the model
            path = os.path.join(self.save_path, f'{self.name_prefix}_{self.n_calls}_steps.zip')
            self.model.save(path)
            if self.verbose > 1:
                print(f'Saving model checkpoint to {path}')

            # Save VecNormalize statistics
            vec_normalize_path = os.path.join(self.save_path, f'{self.name_prefix}_{self.n_calls}_vec_normalize.pkl')
            self.vec_normalize_env.save(vec_normalize_path)
            if self.verbose > 1:
                print(f'Saving VecNormalize statistics to {vec_normalize_path}')

        return True

class SaveVecNormalizeEvalCallback(EvalCallback):
    def __init__(self, vec_normalize_env, *args, **kwargs):
        self.vec_normalize_env = vec_normalize_env
        super(SaveVecNormalizeEvalCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        result = super(SaveVecNormalizeEvalCallback, self)._on_step()
        if self.best_model_save_path is not None and self.n_calls % self.eval_freq == 0:
            if self.vec_normalize_env is not None:
                save_path = os.path.join(self.best_model_save_path, 'vec_normalize_best.pkl')
                self.vec_normalize_env.save(save_path)
        return result

if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))
