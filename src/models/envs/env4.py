import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json
from models.reward.reward import reward_nutrient_macro as calculate_reward

class CalorieOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, num_people=50, max_ingredients=10, action_scaling_factor=21.25, render_mode=None, initial_ingredients=None):
        super(CalorieOnlyEnv, self).__init__()

        self.ingredient_df = ingredient_df
        self.num_people = num_people
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        
        # Nutritional values
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values / 100
        self.Fat_g = ingredient_df['Fat_g'].values / 100
        self.Saturates_g = ingredient_df['Saturates_g'].values / 100
        self.Carbs_g = ingredient_df['Carbs_g'].values / 100
        self.Sugars_g = ingredient_df['Sugars_g'].values / 100
        self.Fibre_g = ingredient_df['Fibre_g'].values / 100
        self.Protein_g = ingredient_df['Protein_g'].values / 100
        self.Salt_g = ingredient_df['Salt_g'].values / 100
        
        # Nutritional targets
        self.target_calories = 530
        self.target_Fat_g = 20.6
        self.target_Saturates_g = 6.5
        self.target_Carbs_g = 70.6
        self.target_Sugars_g = 15.5
        self.target_Fibre_g = 4.2
        self.target_Protein_g = 7.5
        self.target_Salt_g = 0.499
        

        # Ingredient groups for nutritional regulation
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
        
        ### Ingredient day group targets: from UK regulations
        # 1 portion of fruit and veg a day
        self.target_Group_A_fruit = 1
        self.target_Group_A_veg = 1
        # 1 portion from group B unless a portion from group C is (provided meat and processed meat)
        self.target_Group_B = 1
        self.target_Group_C = 1
        # 1 portion from group D every day (carbs, starch etc)
        self.target_Group_D = 1
        # 1 portion must be provided from group E every day (milk and dairy)
        self.target_Group_E = 1
        # 1 portion of bread each day
        self.target_Bread = 1
        # No confectionary is allowed in the diet
        self.target_Confectionary = 0
        
        
        
        
        
        self.Animal_Welfare_Rating = ingredient_df['Animal Welfare Rating'].values
        self.Rainforest_Rating = ingredient_df['Rainforest Rating'].values
        self.Water_Scarcity_Rating = ingredient_df['Water Scarcity Rating'].values
        self.CO2_FU_Rating = ingredient_df['CO2 FU Rating'].values
        self.CO2_kg_per_100g = ingredient_df['CO2_kg_per_100g'].values / 100
        self.target_CO2_kg_g = 0.5
        
        
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values
        self.StandardDeviation = ingredient_df['StandardDeviation'].values
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values
        
        self.render_mode = render_mode
        self.initial_ingredients = initial_ingredients if initial_ingredients is not None else []
        self.n_ingredients = len(ingredient_df)
        self.episode_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(self.n_ingredients + 8,),  
            dtype=np.float32
        )

        self.current_selection = np.zeros(self.n_ingredients)
        self.current_info = {}
        self.actions_taken = []

        self.reward_history = []
        self.nutrient_reward_history = []
        # Setting termination reasons to empty list for plotting random distribution and reason for info logging
        self.termination_reasons = []
        self.termination_reason = 0
        
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
            
            self.current_selection = np.zeros(self.n_ingredients)
            self.actions_taken = []
            
            self.episode_count += 1
            
            self.average_calories_per_day = 0
            self.average_fat_per_day = 0
            self.average_saturates_per_day = 0
            self.average_carbs_per_day = 0
            self.average_sugar_per_day = 0
            self.average_fibre_per_day = 0
            self.average_protein_per_day = 0
            self.average_salt_per_day = 0
            self.termination_reason = 0  # Reset termination reason

            observation = self._get_obs() 
            self.current_info = self._get_info()

            if self.render_mode == 'human':
                self.render()

            return observation, self.current_info

    def step(self, action):
        for ingredient_index in range(self.n_ingredients):
            action_value = action[ingredient_index]
            change = action_value * self.num_people * self.action_scaling_factor
            self.current_selection[ingredient_index] = max(0, self.current_selection[ingredient_index] + change)

            self.actions_taken.append(f"{ingredient_index}: {change:+.2f}")

        num_selected_ingredients = np.sum(self.current_selection > 0)

        if num_selected_ingredients > self.max_ingredients:
            excess_indices = np.argsort(-self.current_selection)
            for idx in excess_indices[self.max_ingredients:]:
                self.current_selection[idx] = 0
                self.actions_taken.append(f"{idx}: set to 0 to maintain ingredient limit")

        reward, nutrient_reward, info, terminated, termination_reason = calculate_reward(self, action)

        self.average_calories_per_day = info.get('Average Calories per Day', self.average_calories_per_day)
        self.average_fat_per_day = info.get('Average Fat per Day', self.average_fat_per_day)
        self.average_saturates_per_day = info.get('Average Saturates per Day', self.average_saturates_per_day)
        self.average_carbs_per_day = info.get('Average Carbs per Day', self.average_carbs_per_day)
        self.average_sugar_per_day = info.get('Average Sugar per Day', self.average_sugar_per_day)
        self.average_fibre_per_day = info.get('Average Fibre per Day', self.average_fibre_per_day)
        self.average_protein_per_day = info.get('Average Protein per Day', self.average_protein_per_day)
        self.average_salt_per_day = info.get('Average Salt per Day', self.average_salt_per_day)

        observation = self._get_obs()
        self.current_info = info

        self.reward_history.append(reward)
        self.nutrient_reward_history.append(nutrient_reward)

        if terminated:
            self.termination_reasons.append(termination_reason.get('reason', 'unknown'))
            if termination_reason['reason'] == 'all_targets_met':
                self.termination_reason = 1 # targets met is 1
            elif termination_reason['reason'] == 'any_target_far_off':
                self.termination_reason = -1 # target far off -1
            else:
                self.termination_reason = 0 # no termination is zero

        info['Termination Reason'] = self.termination_reason

        if self.render_mode == 'human':
            self.render(step=len(self.reward_history))

        return observation, reward, terminated, False, self.current_info
    
    def _get_obs(self):
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
            'Termination Reason': self.termination_reason
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
        nutrient_reward_history_reformat = {key: [] for key in self.nutrient_reward_history[0].keys()}

        for entry in self.nutrient_reward_history:
            for key, value in entry.items():
                nutrient_reward_history_reformat[key].append(value)

        num_rewards = len(nutrient_reward_history_reformat) + 2
        col = 5
        row = num_rewards // col
        if num_rewards % col != 0:
            row += 1

        fig, axes = plt.subplots(row, col, figsize=(col * 3, row * 3))

        axes = np.ravel(axes)

        termination_reason_counts = {reason: self.termination_reasons.count(reason) for reason in set(self.termination_reasons)}
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

        axes[1].hist(self.reward_history, bins=50, alpha=0.75)
        axes[1].set_xlabel('Total reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Total Reward Distribution')

        for ax, (key, values) in zip(axes[2:], nutrient_reward_history_reformat.items()):
            ax.hist(values, bins=50, alpha=0.75)
            ax.set_xlabel(key.replace('_', ' ').capitalize())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{key.replace("_", " ").capitalize()} Reward Distribution')

        for ax in axes[num_rewards:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def save_reward_distribution(self, filepath):
        reward_distribution = {
            'total_reward': self.reward_history,
            'nutrient_rewards': self.nutrient_reward_history,
            'termination_reasons': self.termination_reasons
        }
        with open(filepath, 'w') as json_file:
            json.dump(reward_distribution, json_file, indent=4)

from gymnasium.envs.registration import register

register(
    id='CalorieOnlyEnv-v3',
    entry_point='models.envs.env4:CalorieOnlyEnv',
    max_episode_steps=1000,
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
    num_episodes = 1000
    steps_per_episode = 1000  # Number of steps per episode

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment at the start of each episode
        print(f"Episode {episode + 1}")

        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Sample a random action
            obs, reward, done, _, info = env.step(action)  # Take a step in the environment

            # Print the results after each step
            print(f"    Reward: {reward}")

            if done:
                print(f"Episode {episode + 1} ended early after {step + 1} steps.")
                break

    # Plot reward distribution
    env.plot_reward_distribution()
    # Save reward distribution to a file
    env.save_reward_distribution('reward_distribution.json')

