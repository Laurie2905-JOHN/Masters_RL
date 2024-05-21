import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
from utils.process_data import get_data

# # Register the environment
# from gymnasium.envs.registration import register

# register(
#     id='SimpleCalorieOnlyEnv-v0',
#     entry_point='models.envs.env:SimpleCalorieOnlyEnv',
#     max_episode_steps=1,  # Set max_episode_steps to 1 for a simple step environment
# )

class SimpleCalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, target_calories=2500, render_mode=None):
        super(SimpleCalorieOnlyEnv, self).__init__()

        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.target_calories = target_calories
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values
        self.render_mode = render_mode

        self.n_ingredients = len(ingredient_df)

        # Define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_ingredients,), dtype=np.float32)

        # Define observation space to include only caloric values per ingredient
        self.observation_space = spaces.Box(low=0, high=np.max(self.caloric_values), shape=(self.n_ingredients,), dtype=np.float32)

        self.current_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        total_selection = 0
        average_calories_per_day = 0

        info = {'Total Number of Ingredients Selected': total_selection, 'Average Calories per Day': average_calories_per_day}
        self.current_info = self._get_info(total_selection, average_calories_per_day)

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def calculate_reward(self, action):
        reward = 0

        # Set small values to zero to disregard them in the selection
        threshold = 0.1
        raw_action = np.where(action > threshold, action, 0)

        selected_flag = action > threshold
        # Evaluate the number of selected ingredients based on the adjusted actions
        total_selection = np.sum(selected_flag)

        # Separate the scaling factor from the rest of the action
        scaling_factor = (1300 * self.num_people) / (0.5 * total_selection)  # Reduced scaling factor for more reasonable values
        scaled_action = raw_action * scaling_factor
        
        # Calculate calories only from the selected ingredients
        selected_ingredients = scaled_action[selected_flag]
        selected_calories = self.caloric_values[selected_flag]
        calories_selected_ingredients = selected_ingredients * selected_calories / 100  # Divide by 100 as caloric values are per 100g

        # Calculate average calories per day per person
        average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

        # Check if both conditions are met
        if 5 <= total_selection <= 10 and 2000 <= average_calories_per_day <= 3000:
            reward += 500  # Massive reward for meeting both conditions
            terminated = True
        else:
            terminated = False

            # Apply shaped negative rewards for selecting more than 10 ingredients
            if total_selection > 10:
                reward -= 20 * (total_selection - 10)  # Increasing penalty for selecting more than 10 ingredients
            elif total_selection < 5:
                reward -= 10  # Penalty for selecting less than 5 ingredients
            else:
                reward += 20  # Reward for selecting between 5 and 10 ingredients

            # Reward based on the average calories per day
            if 2000 <= average_calories_per_day <= 3000:
                reward += 100  # Additional reward for meeting the calorie constraint
            else:
                reward += max(0, 50 - abs(average_calories_per_day - 2500) * 0.1)  # Small reward for getting close to the target
                reward -= 10 * abs((average_calories_per_day - 2500) / 2500)  # Shaped penalty for not meeting the calorie constraint

            # Penalty for extreme ingredient quantities
            reward -= np.sum(np.maximum(0, scaled_action - 500)) * 0.1

            # Reward for zero quantities explicitly
            reward += np.sum(scaled_action == 0) * 0.2

        info = self._get_info(total_selection, average_calories_per_day)

        return reward, info, terminated


    def step(self, action):
        reward, info, terminated = self.calculate_reward(action)
        observation = self._get_obs()

        self.current_info = info  # Store the info

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return self.caloric_values.astype(np.float32)

    def _get_info(self, total_selection, average_calories_per_day):
        return {'Total Number of Ingredients Selected': total_selection, 'Average Calories per Day': average_calories_per_day}

    def render(self):
        if self.render_mode == 'human':
            if self.current_info:
                print(f"Number of Ingredients Selected: {self.current_info.get('Total Number of Ingredients Selected', 'N/A')}")
                print(f"Average Calories per Day: {self.current_info.get('Average Calories per Day', 'N/A')}")

    def close(self):
        pass

# Unit Testing
if __name__ == '__main__':
    ingredient_df = get_data()
    env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')

    # Check the custom environment
    check_env(env.unwrapped)
    print("Environment is valid!")
