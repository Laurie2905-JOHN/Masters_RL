import json
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class RewardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, save_reward=False):
        super(RewardTrackingWrapper, self).__init__(env)
        self.save_reward = save_reward
        if self.save_reward:
            self.reward_history = []
            self.nutrient_reward_history = []
            self.group_reward_history = []
            self.termination_reasons = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.save_reward:
            self.reward_history.append(float(reward))
            self.nutrient_reward_history.append(self._convert_to_serializable(info.get('nutrient_averages', {}).copy()))
            self.group_reward_history.append(self._convert_to_serializable(info.get('group_counts', {}).copy()))
            if terminated or truncated:
                reason = info.get('termination_reason')
                if truncated:
                    reason = 0  # Append 0 for truncated episodes
                self.termination_reasons.append(reason)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the environment and clear episode-specific logs
        if self.save_reward and self.termination_reasons:
            last_reason = self.termination_reasons[-1]
            if last_reason is None:
                self.termination_reasons[-1] = 'end_of_episode'
        return self.env.reset(**kwargs)

    def plot_reward_distribution(self, save_path=None):
        if self.save_reward:
            reason_str = [self._reason_to_string(val) for val in self.termination_reasons]
            group_reward_history_reformat = {key: [] for key in self.group_reward_history[0].keys()}
            nutrient_reward_history_reformat = {key: [] for key in self.nutrient_reward_history[0].keys()}

            # Reformat group reward history
            for entry in self.group_reward_history:
                for key, value in entry.items():
                    group_reward_history_reformat[key].append(value)
            
            # Reformat nutrient reward history
            for entry in self.nutrient_reward_history:
                for key, value in entry.items():
                    nutrient_reward_history_reformat[key].append(value)

            # Calculate total number of rewards to plot
            num_rewards = len(nutrient_reward_history_reformat) + len(group_reward_history_reformat) + 2
            col = 6
            row = num_rewards // col
            if num_rewards % col != 0:
                row += 1

            fig, axes = plt.subplots(row, col, figsize=(col * 3, row * 3))

            axes = np.ravel(axes)

            termination_reason_counts = {reason: reason_str.count(reason) for reason in set(reason_str)}

            # Plot termination reason frequency
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
                axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

            # Plot total reward distribution
            axes[1].hist(self.reward_history, bins=50, alpha=0.75)
            axes[1].set_xlabel('Total reward')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Total Reward Distribution')

            # Plot nutrient reward distributions
            index = 2
            for key, values in nutrient_reward_history_reformat.items():
                axes[index].hist(values, bins=50, alpha=0.75)
                axes[index].set_xlabel('Reward')
                axes[index].set_ylabel('Frequency')
                axes[index].set_title(f'{key.replace("_", " ").capitalize()}')
                index += 1

            # Plot group reward distributions
            for key, values in group_reward_history_reformat.items():
                axes[index].hist(values, bins=50, alpha=0.75)
                axes[index].set_xlabel(key.replace('_', ' ').capitalize())
                axes[index].set_ylabel('Frequency')
                axes[index].set_title(f'{key.replace("_", " ").capitalize()} Group')
                index += 1

            # Hide any remaining unused subplots
            for ax in axes[num_rewards:]:
                ax.set_visible(False)

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

    def save_reward_distribution(self, filepath):
        if self.save_reward:
            reason_str = [self._reason_to_string(val) for val in self.termination_reasons]
            reward_distribution = {
                'total_reward': self.reward_history,
                'nutrient_rewards': self.nutrient_reward_history,
                'group_rewards': self.group_reward_history,
                'termination_reasons': reason_str
            }
            with open(filepath, 'w') as json_file:
                json.dump(reward_distribution, json_file, indent=4)

    @staticmethod
    def _convert_to_serializable(data):
        """Convert numpy data types to standard Python types for JSON serialization."""
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
