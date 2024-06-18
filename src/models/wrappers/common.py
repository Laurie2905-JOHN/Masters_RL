import gymnasium as gym
import numpy as np
import pandas as pd
import os
import json

class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_interval, reward_save_path):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_interval = save_interval
        self.reward_save_path = reward_save_path
        self.step_count = 0
        self.reward_history = []
        self.reward_details_history = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward_history.append(float(reward))
        reward_dict = info.get('reward', {}).copy()
        self.reward_details_history.append(json.dumps(reward_dict))  # Serialize dictionary to JSON string
        self.step_count += 1
        
        if self.step_count % self.save_interval == 0:
            self.save_and_clear()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def save_and_clear(self):
        reward_distribution = {
            'total_reward': self.reward_history,
            'reward_details': self.reward_details_history,
        }
        
        df_rewards = pd.DataFrame(reward_distribution)
        
        # Check if file exists and append or write new
        if os.path.exists(self.reward_save_path):
            with pd.HDFStore(self.reward_save_path) as store:
                store.append('rewards', df_rewards, format='table', data_columns=True)
        else:
            with pd.HDFStore(self.reward_save_path) as store:
                store.put('rewards', df_rewards, format='table', data_columns=True)

        # Clear the in-memory storage
        self.reward_history.clear()
        self.reward_details_history.clear()

    def close(self):
        self.save_and_clear()
