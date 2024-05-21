import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
from utils.process_data import get_data

# Register the environment
from gymnasium.envs.registration import register

register(
    id='SimpleCalorieOnlyEnv-v0',
    entry_point='models.envs.env:SimpleCalorieOnlyEnv',
    max_episode_steps=1000,
)

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

        # Define action space to include the scaling factor (last element)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_ingredients + 1,), dtype=np.float32)
        
        # Define observation space to include only caloric values per ingredient
        self.observation_space = spaces.Box(low=0, high=np.max(self.caloric_values), shape=(self.n_ingredients,), dtype=np.float32)

        # Initialize info to an empty dictionary
        self.current_info = {}
        self.previous_calories = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Construct the observation correctly
        observation = self._get_obs()
        
        # Initialize total_selection and average_calories_per_day for the first step
        total_selection = 0
        average_calories_per_day = 0
        
        info = {'Total Number of Ingredients Selected': total_selection, 'Average Calories per Day': average_calories_per_day}  # Initial info

        self.current_info = self._get_info(total_selection, average_calories_per_day)
        self.previous_calories = 0

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def calculate_reward(self, action):
        reward = 0

        # Separate the scaling factor from the rest of the action
        scaling_factor = action[-1] * 1000  # Assuming scaling factor is between 0 and 1000
        scaled_action = action[:-1] * scaling_factor
        
        # Reward based on the number of selected ingredients
        total_selection = np.sum(scaled_action > 0.05)
        
        if total_selection > 10:
            reward -= 10
        elif total_selection < 5:
            reward -= 10
        else:
            reward += 10
        
        calories_selected_ingredients = scaled_action * self.caloric_values / 100 # Devide by 100 as caloric values are per 100g
        
        # Calculate average calories per day per person
        average_calories_per_day = sum(calories_selected_ingredients) / self.num_people
        
        # Reward based on the average calories per day
        if 2000 <= average_calories_per_day <= 3000:
            reward += 100
            terminated = True
        else:
            # Reward based on proximity to the target range
            reward += max(0, 50 - abs(average_calories_per_day - 2500) * 0.1)
            terminated = False
            reward -= 10
        
        # Penalty for extreme ingredient quantities
        reward -= np.sum(np.maximum(0, scaled_action - 500)) * 0.1
        reward -= np.sum(np.maximum(0, 10 - scaled_action)) * 0.1
        
        # Reward for diversity of ingredients
        reward += np.sum(scaled_action > 0.05) * 2

        # Reward for improvement from the previous step
        if self.previous_calories is not None:
            improvement = abs(2500 - average_calories_per_day) - abs(2500 - self.previous_calories)
            reward += max(0, improvement * 0.5)  # Small reward for getting closer to the target

        self.previous_calories = average_calories_per_day

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
            # print(f"Target Calories: {self.target_calories}")
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
