import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from models.reward.reward import reward_nutrient_env4 as calculate_reward

class CalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, max_ingredients=10, render_mode=None, initial_ingredients=None, min_ingredient_amount=100):
        super(CalorieOnlyEnv, self).__init__()

        # Define data from the ingredient dataframe and other inputs
        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.max_ingredients = max_ingredients
        
        # Nutrient data
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values / 100
        self.Fat_g = ingredient_df['Fat_g'].values / 100
        self.Saturates_g = ingredient_df['Saturates_g'].values / 100
        self.Carbs_g = ingredient_df['Carbs_g'].values / 100
        self.Sugars_g = ingredient_df['Sugars_g'].values / 100
        self.Fibre_g = ingredient_df['Fibre_g'].values / 100
        self.Protein_g = ingredient_df['Protein_g'].values / 100
        self.Salt_g = ingredient_df['Salt_g'].values / 100
        
        # Nutrient regulation data
        self.target_calories = 530 # Target max calories per person
        self.target_Fat_g = 20.6 # Target max fat per person
        self.target_Saturates_g = 6.5 # Target max saturates per person
        self.target_Carbs_g = 70.6  # Target min carbs per person
        self.target_Sugars_g = 15.5 # Target max sugars per person
        self.target_Fibre_g = 4.2 # Target min fibre per person
        self.target_Protein_g = 7.5 # Target min protein per person
        self.target_Salt_g = 0.499 # Target max salt per person
        
        # Environment
        self.Animal_Welfare_Rating = ingredient_df['Animal Welfare Rating'].values
        self.Rainforest_Rating = ingredient_df['Rainforest Rating'].values
        self.Water_Scarcity_Rating = ingredient_df['Water Scarcity Rating'].values
        self.CO2_FU_Rating = ingredient_df['CO2 FU Rating'].values
        self.CO2_kg_per_100g = ingredient_df['CO2_kg_per_100g'].values / 100
        
        self.target_CO2_kg_g = 0.5 # Target max CO2 per 100g
        
        # Group categories
        self.Group_A_veg = ingredient_df['Group A veg'].values
        self.Group_A_fruit = ingredient_df['Group A fruit'].values
        self.Group_B = ingredient_df['Group B'].values
        self.Oily_Fish = ingredient_df['Oily Fish'].values
        self.Red_Meat = ingredient_df['Red Meat'].values
        self.Group_C = ingredient_df['Group C'].values
        self.Group_D = ingredient_df['Group D'].values
        self.Group_E = ingredient_df['Group E'].values
        self.Oil = ingredient_df['Oil'].values
        self.Bread = ingredient_df['Bread'].values
        self.Confectionary = ingredient_df['Confectionary'].values
        
        # Consumption data
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values
        self.StandardDeviation = ingredient_df['StandardDeviation'].values
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values
        
        self.render_mode = render_mode
        self.initial_ingredients = initial_ingredients if initial_ingredients is not None else []
        self.n_ingredients = len(ingredient_df)
        self.episode_count = 0

        # Define action space: Continuous action space for more granular control
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)

        # Define observation space: average values per data 
        self.max_possible_calories = self.target_calories * 10  # Arbitrary upper bound for total calories
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_possible_calories,
            shape=(self.n_ingredients + 8,),  
            dtype=np.float32
        )

        # Define current selection and other state variables
        self.current_selection = np.zeros(self.n_ingredients)
        self.current_info = {}
        self.actions_taken = []

        # Reward tracking
        self.reward_history = []
        self.nutrient_reward_history = []
        
        # Initialising values
        self.average_calories_per_day = 0
        self.average_fat_per_day = 0
        self.average_saturates_per_day = 0
        self.average_carbs_per_day = 0
        self.average_sugar_per_day = 0
        self.average_fibre_per_day = 0
        self.average_protein_per_day = 0
        self.average_salt_per_day = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset current selection, actions taken
        self.current_selection = np.zeros(self.n_ingredients)
        self.actions_taken = []
        
        # Counter for episodes
        self.episode_count += 1
        
        # Reset initial nutrient values
        self.average_calories_per_day = 0
        self.average_fat_per_day = 0
        self.average_saturates_per_day = 0
        self.average_carbs_per_day = 0
        self.average_sugar_per_day = 0
        self.average_fibre_per_day = 0
        self.average_protein_per_day = 0
        self.average_salt_per_day = 0

        # Update observation and info
        observation = self._get_obs() 
        self.current_info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, self.current_info

    def step(self, action):
        # Apply actions to modify ingredient selection
        for ingredient_index in range(self.n_ingredients):
            action_value = action[ingredient_index]
            change = action_value * self.num_people * 15 # Scale action value to meaningful change in ingredient amount
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
        reward, nutrient_reward, info, terminated = calculate_reward(self, action)

        # Update the environment state with the info calculated in the reward function
        self.average_calories_per_day = info.get('Average Calories per Day', self.average_calories_per_day)
        self.average_fat_per_day = info.get('Average Fat per Day', self.average_fat_per_day)
        self.average_saturates_per_day = info.get('Average Saturates per Day', self.average_saturates_per_day)
        self.average_carbs_per_day = info.get('Average Carbs per Day', self.average_carbs_per_day)
        self.average_sugar_per_day = info.get('Average Sugar per Day', self.average_sugar_per_day)
        self.average_fibre_per_day = info.get('Average Fibre per Day', self.average_fibre_per_day)
        self.average_protein_per_day = info.get('Average Protein per Day', self.average_protein_per_day)
        self.average_salt_per_day = info.get('Average Salt per Day', self.average_salt_per_day)

        # Update observation and info
        observation = self._get_obs()
        self.current_info = info

        # Track rewards for visualization
        self.reward_history.append(reward)
        self.nutrient_reward_history.append(nutrient_reward)

        if self.render_mode == 'human':
            self.render(step=len(self.reward_history))

        return observation, reward, terminated, False, self.current_info

    def _get_obs(self):
        # Use the stored average nutrient values
        obs = np.concatenate((
            self.current_selection,
            [
                self.average_calories_per_day,
                self.average_fat_per_day,
                self.average_saturates_per_day,
                self.average_carbs_per_day,
                self.average_sugar_per_day,
                self.average_fibre_per_day,
                self.average_protein_per_day,
                self.average_salt_per_day,
            ]
        )).astype(np.float32)
        return obs

    def _get_info(self):
        info = {
            'Average Calories per Day': self.average_calories_per_day,
            'Average Fat per Day': self.average_fat_per_day,
            'Average Saturates per Day': self.average_saturates_per_day,
            'Average Carbs per Day': self.average_carbs_per_day,
            'Average Sugar per Day': self.average_sugar_per_day,
            'Average Fibre per Day': self.average_fibre_per_day,
            'Average Protein per Day': self.average_protein_per_day,
            'Average Salt per Day': self.average_salt_per_day,
            'Current Selection': self.current_selection,
            'Action': self.actions_taken,
        }
        return info

    def render(self, step=None):
        if self.render_mode == 'human':
            if self.current_info:
                print(f"Step: {step}")
                print(f"Average Calories per Day: {self.current_info.get('Average Calories per Day', 'N/A')}")
                print(f"Actions Taken: {self.current_info.get('Action', 'N/A')}")
        if self.render_mode == 'step':
            if step is not None:
                print(f"Step: {step}")

    def close(self):
        pass

    def plot_reward_distribution(self):

        # Create a dictionary to hold lists of values for each nutrient reward type
        nutrient_reward_history_reformat = {key: [] for key in self.nutrient_reward_history[0].keys()}

        # Populate the dictionary with values from each entry in nutrient_reward_history
        for entry in self.nutrient_reward_history:
            for key, value in entry.items():
                nutrient_reward_history_reformat[key].append(value)

        # Calculate the number of subplots required
        num_rewards = len(nutrient_reward_history_reformat) + 1  # Plus one for the total reward
        col = 4
        row = num_rewards // col
        if num_rewards % col != 0:
            row += 1

        # Create a single figure with subplots
        fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 3))

        # Ensure axes is always iterable
        axes = np.ravel(axes)

        # Plot total reward distribution
        axes[0].hist(self.reward_history, bins=50, alpha=0.75)
        axes[0].set_xlabel('Total reward')
        axes[0].set_ylabel('Frequency')

        # Plot histograms for each nutrient reward type
        for ax, (key, values) in zip(axes[1:], nutrient_reward_history_reformat.items()):
            ax.hist(values, bins=50, alpha=0.75)
            ax.set_xlabel(key.replace('_', ' ').capitalize())
            ax.set_ylabel('Frequency')

        # Hide any unused subplots
        for ax in axes[num_rewards:]:
            ax.set_visible(False)

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
    num_episodes = 100
    steps_per_episode = 100  # Number of steps per episode

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment at the start of each episode
        print(f"Episode {episode + 1} starting observation: {obs}")

        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _, info = env.step(action)  # Take a step in the environment

            # Print the results after each step
            # print(f"  Step {step + 1}:")
            # print(f"    Observation: {obs}")
            # print(f"    Reward: {reward}")

            if done:
                print(f"Episode {episode + 1} ended early after {step + 1} steps.")
                break

    # Plot reward distribution
    env.plot_reward_distribution()
