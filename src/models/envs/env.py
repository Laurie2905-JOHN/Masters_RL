import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
from utils.process_data import get_data
from models.reward.reward import calculate_simple_reward
# # Register the environment
# from gymnasium.envs.registration import register

# register(
#     id='SimpleCalorieOnlyEnv-v0',
#     entry_point='models.envs.env:SimpleCalorieOnlyEnv',
#     max_episode_steps=1,  # Set max_episode_steps to 1 for a simple step environment
# )

class SimpleCalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, target_calories=530, render_mode=None):
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

    def step(self, action):
        reward, info, terminated = calculate_simple_reward(self, action)
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
