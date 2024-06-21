import gymnasium as gym
import numpy as np
import pandas as pd
import os

class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_interval, reward_save_path):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_interval = save_interval
        self.reward_save_path = reward_save_path
        self.step_count = 0
        self.reward_history = []
        self.reward_details_history = []
        # self.counter = 0
    def step(self, actions):
        # import time
        # start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.reward_history.append(float(reward))
        reward_dict = info.get('reward', {}).copy()
        reward_dict_flat = self.flatten_dict(reward_dict)
        reward_dict_flat = self.convert_specific_values_to_str(reward_dict_flat)
        self.reward_details_history.append(reward_dict_flat)
        self.step_count += 1
        # end = time.time()
        
        # self.counter += end - start
        if self.step_count % self.save_interval == 0 or self.step_count == 1:
            # print(self.counter)
            self.save_and_clear()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return super(RewardTrackingWrapper, self).reset(**kwargs)
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def convert_specific_values_to_str(self, d):
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = ','.join(map(str, v))  # Convert list to comma-separated string
            # Add more specific conversions if needed here
        return d

    def ensure_float32_dtypes(self, df):
        """Ensure all numerical columns are of type float32."""
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(np.float32)
        return df

    def save_and_clear(self):
        reward_distribution = {
            'total_reward': self.reward_history,
        }
        
        df_rewards = pd.DataFrame(reward_distribution)
        df_reward_details = pd.DataFrame(self.reward_details_history)
        df_combined = pd.concat([df_rewards, df_reward_details], axis=1)

        df_combined = self.ensure_float32_dtypes(df_combined)
        
        # Check if file exists and append or write new
        if os.path.exists(self.reward_save_path):
            with pd.HDFStore(self.reward_save_path) as store:
                store.append('rewards', df_combined, format='table', data_columns=True)
        else:
            with pd.HDFStore(self.reward_save_path) as store:
                store.put('rewards', df_combined, format='table', data_columns=True)

        # Clear the in-memory storage
        self.reward_history.clear()
        self.reward_details_history.clear()

    def close(self):
        self.save_and_clear()
        super(RewardTrackingWrapper, self).close()
