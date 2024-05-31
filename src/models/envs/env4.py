import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from models.reward.reward import reward_no_selection as calculate_reward

class CalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, target_calories=530, max_ingredients=10, render_mode=None, initial_ingredients=None, min_ingredient_amount=100):
        super(CalorieOnlyEnv, self).__init__()

        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.target_calories = target_calories
        self.max_ingredients = max_ingredients
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values / 100  # Convert to per gram
        self.render_mode = render_mode
        self.initial_ingredients = initial_ingredients if initial_ingredients is not None else []

        self.n_ingredients = len(ingredient_df)
        self.episode_count = 0

        # Define action space: Continuous action space for more granular control
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)

        # Define observation space: caloric values per ingredient, current selection, current total calories, and number of selected ingredients
        self.max_possible_calories = self.target_calories * 10  # Arbitrary upper bound for total calories
        
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_possible_calories,
            shape=(2 * self.n_ingredients + 1,),  # calorific values of ingredients, current selection, +2 to include total calories and number of selected ingredients
            dtype=np.float32
        )

        self.current_selection = np.ones(self.n_ingredients)
        self.current_info = {}
        self.actions_taken = []

        # Reward tracking
        self.reward_history = []
        self.selection_reward_history = []
        self.calorie_reward_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_selection = np.ones(self.n_ingredients)
        self.actions_taken = []
        self.episode_count += 1

        observation = self._get_obs()
        self.current_info = self._get_info(
            average_calories_per_day=np.sum(self.current_selection * self.caloric_values) / self.num_people,
            calories_selected_ingredients=self.caloric_values * self.current_selection
        )

        if self.render_mode == 'human':
            self.render()

        return observation, self.current_info

    def step(self, action):
        # Apply actions to modify ingredient selection
        for ingredient_index in range(self.n_ingredients):
            action_value = action[ingredient_index]
            change = action_value * self.num_people * 10 # Scale action value to meaningful change in ingredient amount
            self.current_selection[ingredient_index] = max(0, self.current_selection[ingredient_index] + change)

            self.actions_taken.append(f"{ingredient_index}: {change:+.2f}")

        num_selected_ingredients = np.sum(self.current_selection > 0)

        # Enforce the maximum number of selected ingredients
        if num_selected_ingredients > self.max_ingredients:
            excess_indices = np.argsort(-self.current_selection)  # Sort by amount in descending order
            for idx in excess_indices[self.max_ingredients:]:
                self.current_selection[idx] = 0
                self.actions_taken.append(f"{idx}: set to 0 to maintain ingredient limit")


        # Calculate the reward
        reward, selection_reward, calorie_reward, info, terminated = calculate_reward(self, action)

        # Update observation and info
        observation = self._get_obs()
        self.current_info = info

        # Track rewards for visualization
        self.reward_history.append(reward)
        self.selection_reward_history.append(selection_reward)
        self.calorie_reward_history.append(calorie_reward)

        if self.render_mode == 'human':
            self.render(step=len(self.reward_history))

        return observation, reward, terminated, False, self.current_info

    def _get_obs(self):
        current_total_calories = np.sum(self.current_selection * self.caloric_values / self.num_people)
        return np.concatenate((self.caloric_values, self.current_selection, [current_total_calories])).astype(np.float32)

    def _get_info(self, average_calories_per_day, calories_selected_ingredients):
        info = {
            'Average Calories per Day': average_calories_per_day,
            'Action': self.actions_taken,
            'Calories per selected': calories_selected_ingredients
        }
        return info

    def render(self, step=None):
        if self.render_mode == 'human':
            if self.current_info:
                print(f"Step: {step}")
                print(f"Average Calories per Day: {self.current_info.get('Average Calories per Day', 'N/A')}")
                print(f"Actions Taken: {self.current_info.get('Action', 'N/A')}")

    def close(self):
        pass

    def plot_reward_distribution(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.hist(self.reward_history, bins=50, alpha=0.75)
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.title('Total Reward Distribution')
        
        plt.subplot(1, 3, 2)
        plt.hist(self.selection_reward_history, bins=50, alpha=0.75)
        plt.xlabel('Selection Reward')
        plt.ylabel('Frequency')
        plt.title('Selection Reward Distribution')

        plt.subplot(1, 3, 3)
        plt.hist(self.calorie_reward_history, bins=50, alpha=0.75)
        plt.xlabel('Calorie Reward')
        plt.ylabel('Frequency')
        plt.title('Calorie Reward Distribution')

        plt.tight_layout()
        plt.show()

# Register the environment
from gymnasium.envs.registration import register

register(
    id='CalorieOnlyEnv-v3',
    entry_point='models.envs.env4:CalorieOnlyEnv',
    max_episode_steps=1000,  # Allow multiple steps per episode, adjust as needed
)

if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env

    # Load the ingredient data
    ingredient_df = get_data()

    # Create the environment with initial ingredients
    env = gym.make('CalorieOnlyEnv-v3', ingredient_df=ingredient_df, render_mode=None)

    # Check the custom environment (optional)
    check_env(env.unwrapped)
    print("Environment is valid!")

    # Set numpy print options to suppress scientific notation and set precision
    np.set_printoptions(suppress=True)

    # Number of episodes to test
    num_episodes = 10
    steps_per_episode = 1000  # Number of steps per episode

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment at the start of each episode
        print(f"Episode {episode + 1} starting observation: {obs}")

        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _, info = env.step(action)  # Take a step in the environment

            # Print the results after each step
            print(f"  Step {step + 1}:")
            print(f"    Observation: {obs}")
            print(f"    Reward: {reward}")

            if done:
                print(f"Episode {episode + 1} ended early after {step + 1} steps.")
                break

        # print(f"Episode {episode + 1} final observation: \n {obs}\n")

    # Plot reward distribution
    env.plot_reward_distribution()
