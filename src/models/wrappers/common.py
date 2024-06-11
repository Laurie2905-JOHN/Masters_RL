import json
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_reward=False, save_interval=100, save_path="reward_data.json"):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_reward = save_reward
        self.save_interval = save_interval
        self.save_path = save_path
        self.step_count = 0
        if self.save_reward:
            self.reward_history = []
            self.reward_details_history = []
            self.termination_reasons = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.save_reward:
            self.reward_history.append(float(reward))
            reward_dict = info.get('reward', {}).copy()
            self.reward_details_history.append(self._convert_to_serializable(reward_dict))
            if terminated or truncated:
                reason = info.get('termination_reason')
                if truncated:
                    reason = 1  # Append 1 for truncated episodes
                self.termination_reasons.append(reason)
            
            self.step_count += 1
            if self.step_count % self.save_interval == 0:
                self.save_and_clear(self.save_path)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.save_reward and self.termination_reasons:
            last_reason = self.termination_reasons[-1]
            if last_reason is None:
                self.termination_reasons[-1] = 'end_of_episode'
        return self.env.reset(**kwargs)

    def save_and_clear(self, save_path):
        if self.save_reward:
            reason_str = [self._reason_to_string(val) for val in self.termination_reasons]
            reward_distribution = {
                'total_reward': self.reward_history,
                'reward_details': self.reward_details_history,
                'termination_reasons': reason_str
            }
            with open(save_path, 'a') as json_file:
                json.dump(reward_distribution, json_file, indent=4)
                json_file.write('\n')  # Ensure new data starts on a new line

            # Clear the in-memory storage
            self.reward_history.clear()
            self.reward_details_history.clear()
            self.termination_reasons.clear()

    def save_reward_distribution(self, save_path):
        self.save_and_clear(save_path)

    def plot_reward_distribution(self, load_path, save_plot_path=None):
        if not self.save_reward:
            return

        # Read data from the save file
        reward_history = []
        reward_details_history = []
        termination_reasons = []

        with open(load_path, 'r') as json_file:
            for line in json_file:
                reward_distribution = json.loads(line)
                reward_history.extend(reward_distribution['total_reward'])
                reward_details_history.extend(reward_distribution['reward_details'])
                termination_reasons.extend(reward_distribution['termination_reasons'])

        reason_str = termination_reasons

        # Flatten reward details
        flattened_reward_histories = {
            'nutrient_rewards': {},
            'ingredient_group_count_rewards': {},
            'ingredient_environment_count_rewards': {},
            'cost_rewards': {},
            'consumption_rewards': {}
        }

        for entry in reward_details_history:
            for category, rewards in entry.items():
                if category in flattened_reward_histories:
                    for key, value in rewards.items():
                        if key not in flattened_reward_histories[category]:
                            flattened_reward_histories[category][key] = []
                        flattened_reward_histories[category][key].append(value)

        # Dictionary for shorter labels
        label_mapping = {
            'nutrient_rewards': 'Nutrient',
            'ingredient_group_count_rewards': 'Ingredient Group',
            'ingredient_environment_count_rewards': 'Environment',
            'cost_rewards': 'Cost',
            'consumption_rewards': 'Consumption'
        }

        num_rewards = sum(len(rewards) for rewards in flattened_reward_histories.values()) + 2
        col = 7
        row = num_rewards // col
        if num_rewards % col != 0:
            row += 1

        fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 3))
        axes = np.ravel(axes)

        termination_reason_counts = {reason: reason_str.count(reason) for reason in set(reason_str)}

        bars = axes[0].bar(
            [reason.replace('_', ' ').capitalize() for reason in termination_reason_counts.keys()],
            termination_reason_counts.values()
        )
        axes[0].set_xlabel('Termination Reason')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Termination Reason Frequency')
        axes[0].tick_params(axis='x', rotation=45)

        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')

        axes[1].hist(reward_history, bins=50, alpha=0.75)
        axes[1].set_xlabel('Total reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Total Reward Distribution')

        index = 2
        for category, rewards in flattened_reward_histories.items():
            for key, values in rewards.items():
                short_label = label_mapping.get(category, category)
                axes[index].hist(values, bins=50, alpha=0.75)
                axes[index].set_xlabel(f'{short_label} - {key.replace("_", " ").capitalize()}')
                axes[index].set_ylabel('Frequency')
                index += 1

        for ax in axes[num_rewards:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_plot_path:
            plt.savefig(save_plot_path)
        else:
            plt.show()

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
