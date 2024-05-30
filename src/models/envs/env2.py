import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt  # Make sure to import pyplot


class CalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, target_calories=530, max_ingredients=10, render_mode=None):
        super(CalorieOnlyEnv, self).__init__()

        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.target_calories = target_calories
        self.max_ingredients = max_ingredients
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values / 100  # Convert to per gram
        self.render_mode = render_mode

        self.n_ingredients = len(ingredient_df)

        # Define the increments
        self.increments = [100, 250, 500]  # Multiple increments

        # Define action space: MultiDiscrete action space for each ingredient with options for leave, increase, and decrease
        self.action_space = spaces.MultiDiscrete([2 * len(self.increments) + 1] * self.n_ingredients)

        # Define observation space: caloric values per ingredient, current selection, current total calories, and number of selected ingredients
        max_possible_calories = self.target_calories * 10  # Arbitrary upper bound for total calories
        self.observation_space = spaces.Box(
            low=0,
            high=max_possible_calories,
            shape=(2 * self.n_ingredients + 2,),  # +2 to include total calories and number of selected ingredients
            dtype=np.float32
        )

        self.current_selection = np.zeros(self.n_ingredients)
        self.current_info = {}
        self.num_selected_ingredients = 0
        self.actions_taken = []

        # Reward tracking
        self.reward_history = []
        self.selection_reward_history = []
        self.calorie_reward_history = []

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
            increment = self.increments[increment_index] * self.num_people

            if (action_value - 1) % 2 == 0:
                # Increase action
                self.current_selection[ingredient_index] += increment
                self.actions_taken.append(f"{ingredient_index}: +{increment}")
            else:
                # Decrease action
                self.current_selection[ingredient_index] = max(0, self.current_selection[ingredient_index] - increment)
                self.actions_taken.append(f"{ingredient_index}: -{increment}")
                
        self.num_selected_ingredients = np.sum(self.current_selection > 0)
        
        # Calculate the reward
        reward, selection_reward, calorie_reward, info, terminated = self.calculate_simple_reward3(action)

        # Observation update
        observation = self._get_obs()
        self.current_info = info

        # Track rewards for visualization
        self.reward_history.append(reward)
        self.selection_reward_history.append(selection_reward)
        self.calorie_reward_history.append(calorie_reward)

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, False, self.current_info

    def _get_obs(self):
        current_total_calories = np.sum(self.current_selection * self.caloric_values / self.num_people)
        return np.concatenate((self.caloric_values, self.current_selection, [current_total_calories, self.num_selected_ingredients])).astype(np.float32)

    def _get_info(self, total_selection, average_calories_per_day, calories_selected_ingredients):
        info = {
            'Total Number of Ingredients Selected': total_selection,
            'Average Calories per Day': average_calories_per_day,
            'Action': self.actions_taken,
            'Calories per selected': calories_selected_ingredients
        }
        return info

    def calculate_simple_reward3(self, action):
        total_selection = np.sum(self.current_selection > 0)
        action_selection = np.sum(action > 0)
        calories_selected_ingredients = self.caloric_values * self.current_selection  
        average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

        reward = 0
        selection_reward = 0
        calorie_reward = 0

        target_calories_min = self.target_calories - 50
        target_calories_max = self.target_calories + 50

        # Calorie reward calculation
        if target_calories_min <= average_calories_per_day <= target_calories_max:
            calorie_reward += 1000  # Moderate reward for meeting calorie criteria
        else:
            calories_distance = min(abs(average_calories_per_day - target_calories_min), abs(average_calories_per_day - target_calories_max))
            calorie_reward -= (calories_distance ** 1) / 500

        # Selection reward calculation
        if total_selection < 10:
            selection_reward += (10 - total_selection) ** 2 * 1.0  # Increased weight for selection reward
        elif total_selection > 10:
            selection_reward -= (total_selection - 10) ** 2 * 1.0

        # Large bonus for meeting both criteria
        if target_calories_min <= average_calories_per_day <= target_calories_max and total_selection == 10:
            reward += 1e6  # Large reward for meeting both criteria
            terminated = False
        else:
            terminated = False

        # Action penalty
        if action_selection > 5:
            reward -= (action_selection - 5) ** 2 * 5

        reward += calorie_reward + selection_reward

        info = self._get_info(total_selection, average_calories_per_day, calories_selected_ingredients)
        return reward, selection_reward, calorie_reward, info, terminated


    def render(self):
        if self.render_mode == 'human':
            if self.current_info:
                print(f"Number of Ingredients Selected: {self.current_info.get('Total Number of Ingredients Selected', 'N/A')}")
                print(f"Average Calories per Day: {self.current_info.get('Average Calories per Day', 'N/A')}")
                print(f"Action: {self.current_info.get('Action', 'N/A')}")

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
    id='CalorieOnlyEnv-v1',
    entry_point='models.envs.env2:CalorieOnlyEnv',
    max_episode_steps=100,  # Allow multiple steps per episode, adjust as needed
)

if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env

    # Load the ingredient data
    ingredient_df = get_data()

    # Create the environment
    env = gym.make('CalorieOnlyEnv-v1', ingredient_df=ingredient_df, render_mode=None)

    # Check the custom environment (optional)
    # check_env(env.unwrapped)
    # print("Environment is valid!")
    # Set numpy print options to suppress scientific notation and set precision
    np.set_printoptions(suppress=True)
    # Number of episodes to test
    num_episodes = 1
    steps_per_episode = 500  # Number of steps per episode

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment at the start of each episode
        print(f"Episode {episode + 1} starting observation: {obs}")

        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _, info = env.step(action)  # Take a step in the environment

            # Print the results after each step
            print(f"  Step {step + 1}:")
            print(f"    Sampled action: {action}")
            print(f"    Observation: {obs}")
            # print(f"    Reward: {reward}")
            # print(f"    Done: {done}")
            # print(f"    Info: {info}")

            if done:
                print(f"Episode {episode + 1} ended early after {step + 1} steps.")
                break

        print(f"Episode {episode + 1} final observation: \n {obs}\n")

    # Plot reward distribution
    env.plot_reward_distribution()

