import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import random
from stable_baselines3.common.vec_env import VecNormalize

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

class EvalCallbackWithVecNormalize(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, vec_normalize_save_path, *args, **kwargs):
        super().__init__(eval_env, best_model_save_path=best_model_save_path, *args, **kwargs)
        self.vec_normalize_save_path = vec_normalize_save_path
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Check if we have a new best model
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.last_mean_reward)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                try:
                    # Save VecNormalize statistics for the best model
                    vec_normalize_env = self.model.get_env()
                    if isinstance(vec_normalize_env, VecNormalize):
                        vec_normalize_env.save(self.vec_normalize_save_path)
                        print(f"VecNormalize statistics saved to {self.vec_normalize_save_path}")
                    else:
                        print("Warning: Environment is not a VecNormalize instance. Skipping saving of normalization stats.")
                except Exception as e:
                    print(f"Error saving VecNormalize statistics: {e}")
        return result

    
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
        env = CalorieOnlyEnv(ingredient_df=ingredient_df, action_scaling_factor=scale_factor)
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

if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))
