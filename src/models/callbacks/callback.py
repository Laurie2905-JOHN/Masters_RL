import os
import numpy as np  
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

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
    
class SaveVecNormalizeBestCallback(BaseCallback):
    """
    Callback for saving the best VecNormalize wrapper when a new best model is found.

    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super(SaveVecNormalizeBestCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> None:
        # This callback only saves the VecNormalize stats if there is a new best model
        # Save the VecNormalize stats when a new best model is found
        path = os.path.join(self.save_path, "vecnormalize_best.pkl")

        if self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")
        return True