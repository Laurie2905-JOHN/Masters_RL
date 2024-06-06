import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json
from gymnasium.wrappers import TimeLimit
from models.reward.reward import reward_nutrient_macro, reward_nutrient_macro_and_groups

class SchoolMealSelection(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, reward_func=reward_nutrient_macro_and_groups, max_ingredients=10, action_scaling_factor=15, render_mode=None, initial_ingredients=None):
        super(SchoolMealSelection, self).__init__()

        self.ingredient_df = ingredient_df
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        self.calculate_reward = reward_func  # Store the reward function
        
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
        
        # Define target ranges and initialize rewards
        self.nutrient_target_ranges = {
            'calories': (self.target_calories * 0.95, self.target_calories * 1.05),
            'fat': (self.target_Fat_g * 0.5, self.target_Fat_g),
            'saturates': (0, self.target_Saturates_g),
            'carbs': (self.target_Carbs_g, self.target_Carbs_g * 2),
            'sugar': (0, self.target_Sugars_g),
            'fibre': (self.target_Fibre_g, self.target_Fibre_g * 2),
            'protein': (self.target_Protein_g, self.target_Protein_g * 2),
            'salt': (0, self.target_Salt_g)
        }
        
        self.nutrient_averages = {k: 0 for k in self.nutrient_target_ranges.keys()}
        
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
        self.Group_C = ingredient_df['Group C'].values
        self.Group_D = ingredient_df['Group D'].values
        self.Group_E = ingredient_df['Group E'].values
        self.Bread = ingredient_df['Bread'].values
        self.Confectionary = ingredient_df['Confectionary'].values
        
        ## May be needed later for multiple days
        # self.Oily_Fish = ingredient_df['Oily Fish'].values
        # self.Red_Meat = ingredient_df['Red Meat'].values
        # self.Oil = ingredient_df['Oil'].values
        
        # Ingredient day group targets: from UK school meal regulation  
        # Define group target ranges
        self.ingredient_group_count_targets = {
            'fruit': 1, # 1 fruit a day per meal
            'veg': 1, # 1 veg per day per meal
            'non_processed_meat': 1, # Portion of non processed meat has to be provided accept if a portion of processed meat is provided. This current env is one day meal selection.
            'processed_meat': 1, # Processed meat, see above ^
            'carbs': 1, # Starchy food , a portion of this should be provided every day
            'dairy': 1, # Dairy, a portion of this should be provided every day
            'bread': 1, # Bread should be provided as well as a portion of starchy food
            'confectionary': 0 # No confectionary should be provided
        }

        self.ingredient_group_count = {k: 0 for k in self.ingredient_group_count_targets.keys()}
        
        # Consumption data
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values
        self.StandardDeviation = ingredient_df['StandardDeviation'].values
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values

        self.render_mode = render_mode
        self.initial_ingredients = initial_ingredients if initial_ingredients is not None else []
        self.n_ingredients = len(ingredient_df)
        self.episode_count = 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(self.n_ingredients + len(self.ingredient_group_count.keys()) + len(self.nutrient_averages.keys()),),  # Assuming 8 nutrient averages to be part of observation
            dtype=np.float32
        )

        self.current_selection = np.zeros(self.n_ingredients)
        self.actions_taken = []

        self.reward_history = []
        self.nutrient_reward_history = []
        self.termination_reasons = []
        self.termination_reason = None



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_selection = np.zeros(self.n_ingredients)
        self.actions_taken = []
        self.episode_count += 1
        
        self.nutrient_averages = {k: 0 for k in self.nutrient_averages}
        
        self.ingredient_group_count = {k: 0 for k in self.ingredient_group_count_targets.keys()}
        
        self.termination_reason = None

        observation = self._get_obs()
         
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        action = np.clip(action, -1, 1)  # Ensure actions are within bounds

        change = action * self.action_scaling_factor
        self.current_selection = np.maximum(0, self.current_selection + change)

        num_selected_ingredients = np.sum(self.current_selection > 0)

        if num_selected_ingredients > self.max_ingredients:
            excess_indices = np.argsort(-self.current_selection)
            self.current_selection[excess_indices[self.max_ingredients:]] = 0

        reward, nutrient_reward, info, terminated = self.calculate_reward(self)  # Use the stored reward function

        self.nutrient_averages.update(info['nutrient_averages'])

        obs = self._get_obs()

        self.reward_history.append(reward)
        
        self.nutrient_reward_history.append(nutrient_reward)

        if terminated:            
            self.termination_reasons.append(self.termination_reason)
            
        if self.render_mode == 'human':
            self.render(step=len(self.reward_history))
            
        info = self._get_info()
        return obs, reward, terminated, False, info
    
    def _get_obs(self):
        obs = np.concatenate((
            list(self.nutrient_averages.values()),
            list(self.ingredient_group_count.values()),
            self.current_selection,
        )).astype(np.float32)
        return obs

    def _get_info(self):
        info = {
            'nutrient_averages': self.nutrient_averages,
            'group_counts': self.ingredient_group_count,
            'Current Selection': self.current_selection,
            'Termination Reason': self.termination_reason
        }
        return info

    def render(self, step=None):
        if self.render_mode == 'human':
            print(f"Step: {step}")
            print(f"Average Calories per Day")
        if self.render_mode == 'step':
            if step is not None:
                print(f"Step: {step}")

    def close(self):
        pass

    def plot_reward_distribution(self):
        reason_str = [self._reason_to_string(val) if val != 0 else '' for val in self.termination_reasons]
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

        termination_reason_counts = {reason: reason_str.count(reason) for reason in set(reason_str)}

        
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
        reason_str = [self._reason_to_string(val) if val != 0 else '' for val in self.termination_reasons]
                
        reward_distribution = {
            'total_reward': self.reward_history,
            'nutrient_rewards': self.nutrient_reward_history,
            'termination_reasons': reason_str
        }
        with open(filepath, 'w') as json_file:
            json.dump(reward_distribution, json_file, indent=4)
            
    def _reason_to_string(self, val):
        if val == 2:
            return 'all_targets_met'
        elif val == 1:
            return 'end_of_episode'
        elif val == -1:
            return 'half_of_targets_far_off'

from gymnasium.envs.registration import register

register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env_working:SchoolMealSelection',
    max_episode_steps=1000,
)

if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env

    ingredient_df = get_data()
    
    max_episode_steps = 1000
    
    env = gym.make('SchoolMealSelection-v0', ingredient_df=ingredient_df, render_mode=None)

    check_env(env.unwrapped)
    print("Environment is valid!")

    np.set_printoptions(suppress=True)

    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"Episode {episode + 1}")

        for step in range(max_episode_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"    Observation: {info}")

            if terminated:
                print(f"Episode {episode + 1} ended early after {step + 1} steps.")
                break
            if truncated:
                print(f"Episode {episode + 1} ended early after max steps. {step + 1} steps.")
                break

    env.plot_reward_distribution()
    env.save_reward_distribution('reward_distribution.json')
