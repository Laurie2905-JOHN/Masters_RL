import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import random

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
                # Save VecNormalize statistics for the best model
                self.model.get_env().save(self.vec_normalize_save_path)
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

if __name__ == "__main__":
    base, subdir = get_unique_directory("saved_models/tensorboard", "A2C_100000")
    print(f"Base Directory: {base}")
    print(f"Unique Subdirectory: {subdir}")
    print(os.path.abspath('saved_models/best_models'))
