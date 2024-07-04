import os
import numpy as np  
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

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