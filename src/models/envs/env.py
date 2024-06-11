import gymnasium as gym
from gymnasium import spaces
import numpy as np
from models.reward.reward import reward_nutrient, reward_nutrient_food_groups, reward_nutrient_food_groups_environment
import os
from utils.process_data import get_data

class SchoolMealSelection(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients=10, action_scaling_factor=21.25, render_mode=None, initial_ingredients=None, reward_metrics=None):
        super(SchoolMealSelection, self).__init__()

        self.ingredient_df = ingredient_df
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        self.reward_metrics = reward_metrics if reward_metrics else ['nutrients', 'groups', 'environment']
        
        self.calculate_reward = self.select_reward_function()

        # Nutritional values
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values.astype(np.float32) / 100
        self.Fat_g = ingredient_df['Fat_g'].values.astype(np.float32) / 100
        self.Saturates_g = ingredient_df['Saturates_g'].values.astype(np.float32) / 100
        self.Carbs_g = ingredient_df['Carbs_g'].values.astype(np.float32) / 100
        self.Sugars_g = ingredient_df['Sugars_g'].values.astype(np.float32) / 100
        self.Fibre_g = ingredient_df['Fibre_g'].values.astype(np.float32) / 100
        self.Protein_g = ingredient_df['Protein_g'].values.astype(np.float32) / 100
        self.Salt_g = ingredient_df['Salt_g'].values.astype(np.float32) / 100
        
        # Nutritional targets
        self.target_calories = np.float32(530)
        self.target_Fat_g = np.float32(20.6)
        self.target_Saturates_g = np.float32(6.5)
        self.target_Carbs_g = np.float32(70.6)
        self.target_Sugars_g = np.float32(15.5)
        self.target_Fibre_g = np.float32(4.2)
        self.target_Protein_g = np.float32(7.5)
        self.target_Salt_g = np.float32(0.499)
        
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
        
        # Average values for nutrients over time
        self.nutrient_averages = {k: np.float32(0) for k in self.nutrient_target_ranges.keys()}

        # Group categories
        self.Group_A_veg = ingredient_df['Group A veg'].values.astype(np.float32)
        self.Group_A_fruit = ingredient_df['Group A fruit'].values.astype(np.float32)
        self.Group_B = ingredient_df['Group B'].values.astype(np.float32)
        self.Group_C = ingredient_df['Group C'].values.astype(np.float32)
        self.Group_D = ingredient_df['Group D'].values.astype(np.float32)
        self.Group_E = ingredient_df['Group E'].values.astype(np.float32)
        self.Bread = ingredient_df['Bread'].values.astype(np.float32)
        self.Confectionary = ingredient_df['Confectionary'].values.astype(np.float32)
        
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
        # Count of ingredient groups
        self.ingredient_group_count = {k: np.int8(0) for k in self.ingredient_group_count_targets.keys()}
        
        
        # Environment
        # A - E ratings - converted to a mapping 1 - 5
        # Define the mapping dictionary
        rating_to_int = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

        # Apply the mapping to the 'Animal Welfare Rating' column
        self.Animal_Welfare_Rating = np.array([rating_to_int[val] for val in ingredient_df['Animal Welfare Rating'].values], dtype=np.int8)
        self.Rainforest_Rating = np.array([rating_to_int[val] for val in ingredient_df['Rainforest Rating'].values], dtype=np.int8)
        self.Water_Scarcity_Rating = np.array([rating_to_int[val] for val in ingredient_df['Water Scarcity Rating'].values], dtype=np.int8)
        self.CO2_FU_Rating = np.array([rating_to_int[val] for val in ingredient_df['CO2 FU Rating'].values], dtype=np.int8)
        
        # CO2 values
        self.CO2_g_per_1g = ingredient_df['CO2_g_per_100g'].values.astype(np.float32) / 100 # Retrieve the data and convert to per gram

        self.target_CO2_g_per_meal = np.float32(500) # Target max CO2 per meal
        
        # For A - E ratings will be converted to numbers an average will be taken for each env target
        self.ingredient_environment_count = {
            'animal_welfare': np.float32(0), 
            'rainforest': np.float32(0),
            'water': np.float32(0), 
            'CO2_rating': np.float32(0), 
            'CO2_g': np.float32(0), 
        }
        
        
        # Consumption data
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values.astype(np.float32)
        self.StandardDeviation = ingredient_df['StandardDeviation'].values.astype(np.float32)
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values.astype(np.float32)

        self.render_mode = render_mode
        self.initial_ingredients = initial_ingredients if initial_ingredients is not None else []
        self.n_ingredients = len(ingredient_df)
        self.episode_count = 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)
        
        # Define the observation space based on selected reward metrics
        obs_shape = self.n_ingredients
        if 'nutrients' in self.reward_metrics:
            obs_shape += len(self.nutrient_averages.keys())
        if 'groups' in self.reward_metrics:
            obs_shape += len(self.ingredient_group_count.keys())
        if 'environment' in self.reward_metrics:
            obs_shape += len(self.ingredient_environment_count.keys())

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(obs_shape,),
            dtype=np.float32
        )

        self.current_selection = np.zeros(self.n_ingredients, dtype=np.float32)

        self.termination_reasons = []
        self.termination_reason = None
        
        # Create an empty reward dictionary
        self.reward_dict = {
            'nutrient_rewards': {},
            'ingredient_group_count_rewards': {},
            'ingredient_environment_count_rewards': {}
        }
        
    def select_reward_function(self):
        if set(self.reward_metrics) == {'nutrients'}:
            return reward_nutrient
        elif set(self.reward_metrics) == {'nutrients', 'groups'}:
            return reward_nutrient_food_groups
        elif set(self.reward_metrics) == {'nutrients', 'groups', 'environment'}:
            return reward_nutrient_food_groups_environment
        else:
            raise ValueError("Invalid combination of reward metrics")

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            self.current_selection = np.zeros(self.n_ingredients, dtype=np.float32)
            
            self.episode_count += 1
            
            self.nutrient_averages = {k: np.float32(0) for k in self.nutrient_averages.keys()}
            
            self.ingredient_group_count = {k: np.int8(0) for k in self.ingredient_group_count.keys()}
            
            self.ingredient_environment_count = {k: np.float32(0) for k in self.ingredient_environment_count.keys()}
            
            self.termination_reason = None
            
            self.reward_dict = {
                'nutrient_rewards': {},
                'ingredient_group_count_rewards': {},
                'ingredient_environment_count_rewards': {}
            }

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

        obs = self._get_obs()

        if terminated:            
            self.termination_reasons.append(self.termination_reason)
            
        if self.render_mode == 'human':
            self.render(step=len(self.termination_reasons))
            
        info = self._get_info()
        return obs, reward, terminated, False, info
    
    def _get_obs(self):
        obs_parts = []
        if 'nutrients' in self.reward_metrics:
            obs_parts.append(list(self.nutrient_averages.values()))
        if 'groups' in self.reward_metrics:
            obs_parts.append(list(self.ingredient_group_count.values()))
        if 'environment' in self.reward_metrics:
            obs_parts.append(list(self.ingredient_environment_count.values()))
        obs_parts.append(self.current_selection)
        
        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs

    def _get_info(self):
        info = {
            'nutrient_averages': self.nutrient_averages,
            'ingredient_group_count': self.ingredient_group_count,
            'ingredient_environment_count': self.ingredient_environment_count,
            'reward': self.reward_dict,
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
    reward_metrics = ['nutrients', 'groups', 'environment']
    render_mode = None
    num_envs = 1
    plot_reward_history = True

if __name__ == '__main__':
    ingredient_df = get_data()
    
    args = Args()
    
    seed = 42
 
    num_episodes = 100
    
    max_episode_steps = 1000
    
    from utils.train_utils import setup_environment, get_unique_directory
    
    env = setup_environment(args, seed, ingredient_df)

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
                targets_met = infos[0].get('all_targets_met', False)
                    
                
                if terminated or truncated:
                    break

    # Access the underlying RewardTrackingWrapper for saving rewards
    if args.plot_reward_history: 
        
        reward_dir = os.path.abspath(os.path.join('saved_models', 'reward'))
        reward_prefix = "test"       
        # Save reward distribution for each environment in the vectorized environment
        for i, env_instance in enumerate(env.envs):
                
            reward_dir, reward_prefix = get_unique_directory(reward_dir, f"{reward_prefix}_seed{seed}_env{i}")
            
            env_instance.save_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix)))
            
            reward_prefix_instance = get_unique_directory(reward_dir, reward_prefix, '.png')

            env_instance.plot_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix_instance)))
