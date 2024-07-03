import gymnasium as gym
import numpy as np
import pandas as pd
import os
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict as SpaceDict, Box
import gc
import gymnasium as gym
import numpy as np
from gymnasium import spaces
    
class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_interval, reward_save_path):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_interval = save_interval
        self.reward_save_path = reward_save_path
        self.reward_history = []
        self.reward_details_history = []
        
    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        nsteps = self.get_wrapper_attr('nsteps')
        
        # Only record reward when it is actually calculated
        if nsteps % 25 == 0: 
            self.reward_history.append(float(reward))
            reward_dict = info.get('reward', {}).copy()
            reward_dict_flat = self.flatten_dict(reward_dict)
            reward_dict_flat = self.convert_specific_values_to_str(reward_dict_flat)
            self.reward_details_history.append(reward_dict_flat)
        
        if nsteps % self.save_interval == 0:
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
        
        # Ensure targets_not_met column exists
        if 'targets_not_met' not in df_combined.columns:
            df_combined['targets_not_met'] = ""

        # Check if file exists and append or write new
        if os.path.exists(self.reward_save_path):
            with pd.HDFStore(self.reward_save_path) as store:
                store.append('rewards', df_combined, format='table', data_columns=True)
        else:
            with pd.HDFStore(self.reward_save_path) as store:
                store.put('rewards', df_combined, format='table', data_columns=True, min_itemsize={'targets_not_met': 26})

        # Clear the in-memory storage
        self.reward_history.clear()
        self.reward_details_history.clear()

        # Explicitly delete dataframes to free up memory
        del df_rewards
        del df_reward_details
        del df_combined

        # Explicitly call garbage collection
        gc.collect()

    def close(self):
        # Save any remaining data when closing
        if self.reward_history or self.reward_details_history:
            self.save_and_clear()
        super(RewardTrackingWrapper, self).close()


# class BootStrapWrapper(gym.Wrapper):
#     def __init__(self, env, max_steps, value_estimator, discount_factor=0.99, test_mode=False):
#         super().__init__(env)
#         self.env = env
#         self.max_steps = max_steps
#         self.value_estimator = value_estimator
#         self.discount_factor = discount_factor
#         self.test_mode = test_mode

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

#     def step(self, action):

#         observation, reward, terminated, truncated, info = self.env.step(action)

#         if not self.test_mode:
#             if truncated:
#                 # Estimate the future value of the current state
#                 future_value = self.value_estimator(observation)
#                 # Bootstrap the reward with the discounted future value
#                 reward += self.discount_factor * future_value

#         return observation, reward, terminated, truncated, info

#     def value_estimator(self, observation):
#         # Placeholder for a method to estimate the value of the current state
#         # This should be replaced by a real estimation function
#         # Typically this would access the agent's value network or similar
#         return 0.0


