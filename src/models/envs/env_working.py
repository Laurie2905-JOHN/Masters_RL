import gymnasium as gym
from gymnasium import spaces
import numpy as np
from models.reward.reward import reward_nutrient_macro, reward_nutrient_macro_and_groups
import os
from utils.process_data import get_data



class SchoolMealSelection(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, reward_func=reward_nutrient_macro_and_groups, max_ingredients=10, action_scaling_factor=21.25, render_mode=None, initial_ingredients=None):
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

        reward, info, terminated = self.calculate_reward(self)  # Use the stored reward function

        self.nutrient_averages.update(info['nutrient_averages'])
        self.ingredient_group_count.update(info['group_counts'])

        obs = self._get_obs()

        if terminated:            
            self.termination_reasons.append(self.termination_reason)
            
        if self.render_mode == 'human':
            self.render(step=len(self.termination_reasons))
            
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
            'current_selection': self.current_selection,
            'termination_reason': self.termination_reason
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

from gymnasium.envs.registration import register

register(
    id='SchoolMealSelection-v1',
    entry_point='models.envs.env_working:SchoolMealSelection',
    max_episode_steps=1000,
)

if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env

    # Define arguments
class Args:
    reward_func = 'reward_nutrient_macro_and_groups'
    render_mode = None
    num_envs = 1
    plot_reward_history=True

if __name__ == '__main__':
    ingredient_df = get_data()
    
    args = Args()
    seed = 42

    from utils.train_utils import get_unique_directory, get_unique_image_directory, setup_environment
    
    num_episodes = 100
    max_episode_steps = 1
    
    env = setup_environment(args, seed, ingredient_df, max_steps_per_episode=max_episode_steps)

    # Check the first environment in the VecEnv
    check_env(env.unwrapped.envs[0].unwrapped)
    
    print("Environment is valid!")

    np.set_printoptions(suppress=True)
    
    for episode in range(num_episodes):
            obs = env.reset()
            print(f"Episode {episode + 1}")

            for step in range(max_episode_steps):
                action = env.action_space.sample()
                obs, rewards, dones, infos = env.step(action)
                
                # VecEnv will return arrays of values
                terminated = dones[0]
                truncated = infos[0].get('TimeLimit.truncated', False)
                
                print(infos)
                
                print(f"Step {step + 1}: terminated={terminated}, truncated={truncated}")
                
                if terminated or truncated:
                    break

    # # Access the underlying RewardTrackingWrapper for saving rewards
    # if args.plot_reward_history: 
        
    #     # Save reward distribution for each environment in the vectorized environment
    #     for i, env_instance in enumerate(env.envs):
                
    #         reward_dir, reward_prefix = get_unique_directory(reward_dir, f"{reward_prefix}_seed{seed}_env{i}")
            
    #         env_instance.save_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix)))
            
    #         reward_prefix_instance = get_unique_image_directory(reward_dir, reward_prefix)

    #         env_instance.plot_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix_instance)))
            

