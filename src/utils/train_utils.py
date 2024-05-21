from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

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