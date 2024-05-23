import gymnasium as gym
from gymnasium import spaces
import numpy as np
from models.reward.reward import calculate_simple_reward2

class CalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, target_calories=530, max_ingredients=10, render_mode=None):
        super(CalorieOnlyEnv, self).__init__()

        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.target_calories = target_calories
        self.max_ingredients = max_ingredients
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values
        self.render_mode = render_mode

        self.n_ingredients = len(ingredient_df)

        # Define the increments
        self.increments = [10, 50, 100, 500]

        # Define action space: MultiDiscrete action space for each ingredient with options for leave, increase, and decrease
        self.action_space = spaces.MultiDiscrete([2 * len(self.increments) + 1] * self.n_ingredients)

        # Define observation space: caloric values per ingredient, current total calories, and number of selected ingredients
        max_possible_calories = self.target_calories * 2  # Arbitrary upper bound for total calories
        self.observation_space = spaces.Box(
            low=0,
            high=max_possible_calories,
            shape=(self.n_ingredients + 2,),  # +2 to include total calories and number of selected ingredients
            dtype=np.float32
        )

        self.current_selection = np.zeros(self.n_ingredients)
        self.current_info = {}
        self.num_selected_ingredients = 0
        self.actions_taken = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_selection = np.zeros(self.n_ingredients)
        self.num_selected_ingredients = 0
        self.actions_taken = []
        observation = self._get_obs()
        self.current_info = self._get_info(
            total_selection=0,
            average_calories_per_day=0,
            calories_selected_ingredients=np.zeros(self.n_ingredients)
        )
        
        if self.render_mode == 'human':
            self.render()

        return observation, self.current_info

    def step(self, action):
        for ingredient_index in range(self.n_ingredients):
            action_value = action[ingredient_index]
            if action_value == 0:
                # Leave action
                self.actions_taken.append(f"{ingredient_index}: leave")
                continue

            increment_index = (action_value - 1) // 2
            increment = self.increments[increment_index]
            increment_value = increment / 100.0

            if (action_value - 1) % 2 == 0:
                # Increase action
                self.current_selection[ingredient_index] += increment_value
                self.num_selected_ingredients += 1
                self.actions_taken.append(f"{ingredient_index}: +{increment}")
            else:
                # Decrease action
                self.current_selection[ingredient_index] = max(0, self.current_selection[ingredient_index] - increment_value)
                self.num_selected_ingredients = max(0, self.num_selected_ingredients - 1)
                self.actions_taken.append(f"{ingredient_index}: -{increment}")

        # Calculate the reward
        reward, info, terminated = calculate_simple_reward2(self, action)

        # Observation update
        observation = self._get_obs()
        self.current_info = info

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, False, self.current_info

    def _get_obs(self):
        current_total_calories = np.sum(self.current_selection * self.caloric_values / 100)
        return np.append(self.caloric_values, [current_total_calories, self.num_selected_ingredients]).astype(np.float32)

    def _get_info(self, total_selection, average_calories_per_day, calories_selected_ingredients):
        info = {
            'Total Number of Ingredients Selected': total_selection,
            'Average Calories per Day': average_calories_per_day,
            'Action': self.actions_taken,
            'Calories per selected': calories_selected_ingredients
        }
        return info

    def render(self):
        if self.render_mode == 'human':
            if self.current_info:
                print(f"Number of Ingredients Selected: {self.current_info.get('Total Number of Ingredients Selected', 'N/A')}")
                print(f"Average Calories per Day: {self.current_info.get('Average Calories per Day', 'N/A')}")
                print(f"Action: {self.current_info.get('Action', 'N/A')}")

    def close(self):
        pass

# Register the environment
from gymnasium.envs.registration import register

register(
    id='CalorieOnlyEnv-v1',
    entry_point='models.envs.env2:CalorieOnlyEnv',
    max_episode_steps=1000,  # Allow multiple steps per episode, adjust as needed
)

# Unit Testing
if __name__ == '__main__':
    from utils.process_data import get_data
    ingredient_df = get_data()
    env = gym.make('CalorieOnlyEnv-v1', ingredient_df=ingredient_df, render_mode='human')

    # Check the custom environment
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped)
    print("Environment is valid!")
    
    # Sample an action and interpret it
    action = env.action_space.sample()
    obs, _ = env.reset()
    print(f"Sampled action: {action}")
    print(f"Observation after reset: {obs}")

    # Step through the environment with the sampled action
    obs, reward, done, _, info = env.step(action)
    print(f"Observation after step: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
