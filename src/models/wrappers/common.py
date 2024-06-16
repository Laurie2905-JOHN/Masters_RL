import json
import gymnasium as gym
import numpy as np
from collections import Counter


class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_interval, reward_save_path):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_interval = save_interval
        self.reward_save_path = reward_save_path
        self.step_count = 0
        self.reward_history = []
        self.reward_details_history = []
        self.termination = Counter()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.reward_history.append(float(reward))
        reward_dict = info.get('reward', {}).copy()
        self.reward_details_history.append(self._convert_to_serializable(reward_dict))
        if terminated or truncated:
            reason = info.get('termination_reason')
            if truncated:
                reason = 1  # Append 1 for truncated episodes
            reason_str = self._reason_to_string(reason)
            self.termination[reason_str] += 1
        
        self.step_count += 1
        
        if self.step_count % self.save_interval == 0 and self.reward_save_path or self.step_count==1:
            self.save_and_clear()
            if self.verbose > 0:
                print(f"Step: {self.step_count}")
                print(f"File updated and saved to: {self.reward_save_path}")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def save_and_clear(self):

        reward_distribution = {
            'total_reward': self.reward_history,
            'reward_details': self.reward_details_history,
            'termination_reasons': dict(self.termination)
        }
        with open(self.reward_save_path, 'a') as json_file:
            json_file.write(json.dumps(reward_distribution) + '\n')

        # Clear the in-memory storage
        self.reward_history.clear()
        self.reward_details_history.clear()
        self.termination.clear()

    def save_reward_distribution(self):
        self.save_and_clear()
        
    def close(self):
        self.save_reward_distribution()

    @staticmethod
    def _convert_to_serializable(data):
        if isinstance(data, dict):
            return {k: RewardTrackingWrapper._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [RewardTrackingWrapper._convert_to_serializable(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        else:
            return data
        
    @staticmethod
    def _reason_to_string(val):
        if val == 2:
            return 'all_targets_met'
        elif val == 1:
            return 'end_of_episode'
        elif val == -1:
            return 'targets_far_off'
        else:
            return 'unknown_reason'
