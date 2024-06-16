import json
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_reward=False, save_interval=1000, reward_save_path=None):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_reward = save_reward
        self.save_interval = save_interval
        self.reward_save_path = reward_save_path
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
            if self.step_count % self.save_interval == 0 and self.reward_save_path:
                self.save_and_clear(self.reward_save_path)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def save_and_clear(self, reward_save_path):
        if self.save_reward:
            reason_str = [self._reason_to_string(val) for val in self.termination_reasons]
            reward_distribution = {
                'total_reward': self.reward_history,
                'reward_details': self.reward_details_history,
                'termination_reasons': reason_str
            }
            with open(reward_save_path, 'a') as json_file:
                json_file.write(json.dumps(reward_distribution) + '\n')

            # Clear the in-memory storage
            self.reward_history.clear()
            self.reward_details_history.clear()
            self.termination_reasons.clear()

    def save_reward_distribution(self, reward_save_path):
        self.save_and_clear(reward_save_path)

    def plot_reward_distribution(self, load_path, save_plot_path=None, chunk_size=8000):
        if not self.save_reward:
            return

        # Initialize counters
        reward_counter = Counter()
        nutrient_rewards_counter = Counter()
        ingredient_group_count_rewards_counter = Counter()
        ingredient_environment_count_rewards_counter = Counter()
        cost_rewards_counter = Counter()
        consumption_rewards_counter = Counter()
        targets_not_met_counter = Counter()
        termination_reasons_counter = Counter()

        # Function to process a chunk of data
        def process_chunk(chunk):
            for line in chunk:
                try:
                    reward_distribution = json.loads(line)
                    reward_counter.update(reward_distribution['total_reward'])
                    termination_reasons_counter.update(reward_distribution['termination_reasons'])

                    for entry in reward_distribution['reward_details']:
                        nutrient_rewards_counter.update(entry.get('nutrient_rewards', {}))
                        ingredient_group_count_rewards_counter.update(entry.get('ingredient_group_count_rewards', {}))
                        ingredient_environment_count_rewards_counter.update(entry.get('ingredient_environment_count_rewards', {}))
                        cost_rewards_counter.update(entry.get('cost_rewards', {}))
                        consumption_rewards_counter.update(entry.get('consumption_rewards', {}))
                        targets_not_met_counter.update(entry.get('targets_not_met', []))
                except json.JSONDecodeError:
                    continue

        # Read data in chunks
        with open(load_path, 'r') as json_file:
            chunk = []
            for line in json_file:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    process_chunk(chunk)
                    chunk = []
            if chunk:
                process_chunk(chunk)

        # Dictionary for shorter labels
        label_mapping = {
            'nutrient_rewards': 'Nutrient',
            'ingredient_group_count_rewards': 'Ingredient Group',
            'ingredient_environment_count_rewards': 'Environment',
            'cost_rewards': 'Cost',
            'consumption_rewards': 'Consumption',
            'targets_not_met': 'Targets Not Met'
        }

        # Plotting
        num_rewards = 7  # Number of distinct plots we need
        col = 7
        row = (num_rewards // col) + (1 if num_rewards % col != 0 else 0)

        fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 3))
        axes = np.ravel(axes)

        # Plot termination reasons
        termination_reasons_keys = list(termination_reasons_counter.keys())
        termination_reasons_values = list(termination_reasons_counter.values())
        bars = axes[0].bar(
            [reason.replace('_', ' ').capitalize() for reason in termination_reasons_keys],
            termination_reasons_values
        )
        axes[0].set_xlabel('Termination Reason')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Termination Reason Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')

        # Plot total reward distribution
        reward_values, reward_counts = zip(*reward_counter.items())
        axes[1].hist(reward_values, weights=reward_counts, bins=50, alpha=0.75)
        axes[1].set_xlabel('Total reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Total Reward Distribution')

        index = 2
        reward_categories = [
            ('nutrient_rewards', nutrient_rewards_counter),
            ('ingredient_group_count_rewards', ingredient_group_count_rewards_counter),
            ('ingredient_environment_count_rewards', ingredient_environment_count_rewards_counter),
            ('cost_rewards', cost_rewards_counter),
            ('consumption_rewards', consumption_rewards_counter),
            ('targets_not_met', targets_not_met_counter)
        ]

        for category, counter in reward_categories:
            if category == 'targets_not_met':
                targets_keys = list(counter.keys())
                targets_values = list(counter.values())
                bars = axes[index].bar(
                    [target.replace('_', ' ').capitalize() for target in targets_keys],
                    targets_values
                )
                axes[index].set_xlabel('Targets Not Met')
                axes[index].set_ylabel('Frequency')
                axes[index].set_title('Targets Not Met Frequency')
                axes[index].tick_params(axis='x', rotation=45)
                for bar in bars:
                    yval = bar.get_height()
                    axes[index].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')
                index += 1
            else:
                for key, values in counter.items():
                    short_label = label_mapping.get(category, category)
                    axes[index].hist([key]*values, bins=50, alpha=0.75)
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
