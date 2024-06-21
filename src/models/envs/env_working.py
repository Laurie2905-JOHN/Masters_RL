import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from models.reward.reward_sparse import RewardCalculator
import os
import random
from gymnasium.envs.registration import register
import torch 
register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env_working:SchoolMealSelection',
)

class SchoolMealSelection(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients=6, action_scaling_factor=10, render_mode=None, reward_metrics=None, verbose=0, seed = None):
        super(SchoolMealSelection, self).__init__()

        self.ingredient_df = ingredient_df
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        self.reward_metrics = reward_metrics if reward_metrics else ['nutrients', 'groups', 'environment', 'cost', 'consumption']
        self.verbose = verbose
        self.seed = seed

        # Initialize nutrient values
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values.astype(np.float32) / 100
        self.Fat_g = ingredient_df['Fat_g'].values.astype(np.float32) / 100
        self.Saturates_g = ingredient_df['Saturates_g'].values.astype(np.float32) / 100
        self.Carbs_g = ingredient_df['Carbs_g'].values.astype(np.float32) / 100
        self.Sugars_g = ingredient_df['Sugars_g'].values.astype(np.float32) / 100
        self.Fibre_g = ingredient_df['Fibre_g'].values.astype(np.float32) / 100
        self.Protein_g = ingredient_df['Protein_g'].values.astype(np.float32) / 100
        self.Salt_g = ingredient_df['Salt_g'].values.astype(np.float32) / 100

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

        # Initialize average nutrient values
        self.nutrient_averages = {k: 0.0 for k in self.nutrient_target_ranges.keys()}

        # Group categories
        self.Group_A_veg = ingredient_df['Group A veg'].values.astype(np.float32)
        self.Group_A_fruit = ingredient_df['Group A fruit'].values.astype(np.float32)
        self.Group_B = ingredient_df['Group B'].values.astype(np.float32)
        self.Group_C = ingredient_df['Group C'].values.astype(np.float32)
        self.Group_D = ingredient_df['Group D'].values.astype(np.float32)
        self.Group_E = ingredient_df['Group E'].values.astype(np.float32)
        self.Bread = ingredient_df['Bread'].values.astype(np.float32)
        self.Confectionary = ingredient_df['Confectionary'].values.astype(np.float32)

        # Group target ranges
        self.ingredient_group_count_targets = {
            'fruit': 1,
            'veg': 1,
            'non_processed_protein': 0.5,
            'processed_protein': 0.5,
            'carbs': 1,
            'dairy': 1,
            'bread': 1,
            'confectionary': 0
        }

        self.ingredient_group_portion_targets = {
            'fruit': (40, 110),
            'veg': (40, 110),
            'non_processed_protein': (70, 150),
            'processed_protein': (70, 150),
            'carbs': (25, 300),
            'dairy': (20, 200),
            'bread': (40, 90),
            'confectionary': (0, 0)
        }
        

        # Define group indexes and probabilities together in a dictionary for initialisation after reset
        self.group_info = {
            'fruit': {'indexes': np.nonzero(self.Group_A_veg)[0], 'probability': 0.9},
            'veg': {'indexes': np.nonzero(self.Group_A_fruit)[0], 'probability': 0.9},
            'non_processed_protein': {'indexes': np.nonzero(self.Group_B)[0], 'probability': 0.3},
            'processed_protein': {'indexes': np.nonzero(self.Group_C)[0], 'probability': 0.7},
            'carbs': {'indexes': np.nonzero(self.Group_D)[0], 'probability': 0.9},
            'dairy': {'indexes': np.nonzero(self.Group_E)[0], 'probability': 0.9},
            'bread': {'indexes': np.nonzero(self.Bread)[0], 'probability': 0.9},
            'confectionary': {'indexes': np.nonzero(self.Confectionary)[0], 'probability': 0.1}
        }
        
        self.n_ingredients = len(ingredient_df)
        
                # # Group counts and portions
        # self.ingredient_group_count = {k: 0 for k in self.ingredient_group_count_targets.keys()}
        # self.ingredient_group_portion = {k: 0.0 for k in self.ingredient_group_portion_targets.keys()}

        # Environmental ratings
        rating_to_int = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        self.Animal_Welfare_Rating = np.array([rating_to_int[val] for val in ingredient_df['Animal Welfare Rating'].values], dtype=np.float32)
        self.Rainforest_Rating = np.array([rating_to_int[val] for val in ingredient_df['Rainforest Rating'].values], dtype=np.float32)
        self.Water_Scarcity_Rating = np.array([rating_to_int[val] for val in ingredient_df['Water Scarcity Rating'].values], dtype=np.float32)
        self.CO2_FU_Rating = np.array([rating_to_int[val] for val in ingredient_df['CO2 FU Rating'].values], dtype=np.float32)

        self.CO2_g_per_1g = ingredient_df['CO2_g_per_100g'].values.astype(np.float32) / 100
        self.target_CO2_g_per_meal = 500

        # self.ingredient_environment_count = {
        #     'animal_welfare': 0,
        #     'rainforest': 0,
        #     'water': 0,
        #     'CO2_rating': 0,
        #     'CO2_g': 0,
        # }

        # Consumption data
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values.astype(np.float32)
        self.StandardDeviation = ingredient_df['StandardDeviation'].values.astype(np.float32)
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values.astype(np.float32)

        # self.consumption_average = {
        #     'average_mean_consumption': 0.0,
        #     'average_cv_ingredients': 0.0
        # }

        # Cost data
        self.Cost_per_1g = ingredient_df['Cost_100g'].values.astype(np.float32) / 100
        # self.menu_cost = {'menu_cost': 0.0}
        self.target_cost_per_meal = 2
        


        self.render_mode = render_mode


        self.episode_count = -1
        self.nsteps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)
        
        # Initialize current_selection to all zeroes
        self.current_selection = np.zeros(self.n_ingredients)
        # Will calculate and initialize the metrics based on the selection
        # self._get_metrics()

        # # Dictionary to hold different parts of the observation space
        # obs_spaces = {
        #     'current_selection': spaces.Box(low=0, high=100, shape=(self.n_ingredients,), dtype=np.float32)
        # }

        # if 'nutrients' in self.reward_metrics:
        #     obs_spaces['nutrients'] = spaces.Box(low=0, high=1000, shape=(len(self.nutrient_averages),), dtype=np.float32)
        # if 'groups' in self.reward_metrics:
        #     obs_spaces['groups'] = spaces.Box(low=0, high=1000, shape=(len(self.ingredient_group_count),), dtype=np.float32)
        # if 'environment' in self.reward_metrics:
        #     obs_spaces['environment'] = spaces.Box(low=0, high=1000, shape=(len(self.ingredient_environment_count),), dtype=np.float32)
        # if 'cost' in self.reward_metrics:
        #     obs_spaces['cost'] = spaces.Box(low=0, high=1000, shape=(len(self.menu_cost),), dtype=np.float32)
        # if 'consumption' in self.reward_metrics:
        #     obs_spaces['consumption'] = spaces.Box(low=0, high=1000, shape=(len(self.consumption_average),), dtype=np.float32)

        #         # Set the observation space to a Dict containing all the sub-spaces
        # self.observation_space = spaces.Dict(obs_spaces)
        
        self.observation_space = spaces.Dict({
            'current_selection': spaces.Box(low=0, high=5000, shape=(self.n_ingredients,), dtype=np.float32),
            'nutrients': spaces.Box(low=0, high=5000, shape=(8,), dtype=np.float32),
            'groups': spaces.Box(low=0, high=5000, shape=(8,), dtype=np.float32),
            'environment': spaces.Box(low=0, high=5000, shape=(5,), dtype=np.float32),
            'cost': spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),
            'consumption': spaces.Box(low=0, high=5000, shape=(2,), dtype=np.float32)
        })


        # Will update the current selection with initial ingredients and calculate the initial stat values
        # This is so the agent is presented with a reasonable starting point
        self._initialise_selection()
        
        # Counter for what targets are met and not met
        self.target_not_met_counters = Counter()
        
        # Function to reward dictionary key name to reward function mapping
        self.reward_to_function_mapping = {
            'nutrient_reward': 'nutrient_reward',
            'ingredient_group_count_reward': 'group_count_reward',
            'ingredient_environment_count_reward': 'environment_count_reward',
            'cost_reward': 'cost_reward',
            'consumption_reward': 'consumption_reward',
        }
        
        # Reward key name to name for reward metric mapping
        self.reward_to_metric_mapping = {
            'nutrient_reward': 'nutrients',
            'ingredient_group_count_reward': 'groups',
            'ingredient_environment_count_reward': 'environment',
            'cost_reward': 'cost_reward',
            'consumption_reward': 'consumption',
            'targets_not_met': 'targets_not_met',
            'termination_reward': 'termination'
        }

    def _calculate_reward(self):

        reward = 0
        terminated = False
        
        # Calculating metrics based on current meal plan
        self._get_metrics()
        
        # Initialising reward dictionary to all zero
        self._initialize_rewards()
        
        terminated = self._evaluate_rewards()
        
        reward += sum(sum(r.values()) for r in self.reward_dict.values() if isinstance(r, dict))

        return reward, terminated
    
    def _get_metrics(self):
        
        self.nutrient_averages = self._calculate_nutrient_averages()
        non_zero_mask = self.current_selection != 0
        non_zero_values = self._calculate_non_zero_values(non_zero_mask)
        self.ingredient_group_count = self._sum_dict_values(non_zero_values)
        self.ingredient_group_portion = self._sum_dict_values(non_zero_values, multiply_with=self.current_selection)
        self.ingredient_environment_count = self._calculate_ingredient_environment_count(non_zero_mask)
        self.menu_cost = self._calculate_menu_cost()
        self.consumption_average = self._calculate_consumption_average(non_zero_mask)

    def _calculate_nutrient_averages(self):
        return {
            'calories': sum(self.caloric_values * self.current_selection),
            'fat': sum(self.Fat_g * self.current_selection),
            'saturates': sum(self.Saturates_g * self.current_selection),
            'carbs': sum(self.Carbs_g * self.current_selection),
            'sugar': sum(self.Sugars_g * self.current_selection),
            'fibre': sum(self.Fibre_g * self.current_selection),
            'protein': sum(self.Protein_g * self.current_selection),
            'salt': sum(self.Salt_g * self.current_selection)
        }

    def _calculate_non_zero_values(self, non_zero_mask):
        return {
            'fruit': self.Group_A_fruit * non_zero_mask,
            'veg': self.Group_A_veg * non_zero_mask,
            'non_processed_protein': self.Group_B * non_zero_mask,
            'processed_protein': self.Group_C * non_zero_mask,
            'carbs': self.Group_D * non_zero_mask,
            'dairy': self.Group_E * non_zero_mask,
            'bread': self.Bread * non_zero_mask,
            'confectionary': self.Confectionary * non_zero_mask,
        }

    def _sum_dict_values(self, data, multiply_with=None):
        if multiply_with is None:
            return {key: sum(values) for key, values in data.items()}
        else:
            return {key: sum(values * multiply_with) for key, values in data.items()}

    def _calculate_ingredient_environment_count(self, non_zero_mask):
        return {
            'animal_welfare': round(sum(self.Animal_Welfare_Rating * non_zero_mask) / self.max_ingredients),
            'rainforest': round(sum(self.Rainforest_Rating * non_zero_mask) / self.max_ingredients),
            'water': round(sum(self.Water_Scarcity_Rating * non_zero_mask) / self.max_ingredients),
            'CO2_rating': round(sum(self.CO2_FU_Rating * non_zero_mask) / self.max_ingredients),
            'CO2_g': sum(self.CO2_g_per_1g * self.current_selection),
        }

    def _calculate_consumption_average(self, non_zero_mask):
        return {
            'average_mean_consumption': sum(non_zero_mask * self.Mean_g_per_day) / self.max_ingredients,
            'average_cv_ingredients': sum(non_zero_mask * self.Coefficient_of_Variation) / self.max_ingredients
        }
    
    def _calculate_menu_cost(self):
        return {
            'menu_cost': sum(self.Cost_per_1g * self.current_selection),
        }


    def _initialize_rewards(self):
        reward_keys = ['nutrient_reward', 'ingredient_group_count_reward', 'ingredient_environment_count_reward', 'cost_reward', 'consumption_reward']
        self.reward_dict = {key: {} for key in reward_keys}

        self.reward_dict['nutrient_reward'] = {k: 0 for k in self.nutrient_averages.keys()}
        self.reward_dict['ingredient_group_count_reward'] = {k: 0 for k in self.ingredient_group_count.keys()}
        self.reward_dict['ingredient_environment_count_reward'] = {k: 0 for k in self.ingredient_environment_count.keys()}
        self.reward_dict['cost_reward'] = {'from_target': 0}
        self.reward_dict['consumption_reward'] = {'average_mean_consumption': 0, 'cv_penalty': 0},
        self.reward_dict['targets_not_met'] = []
        self.reward_dict['termination_reward'] = 0
        self.reward_dict['step_penalty'] = 0

    def _evaluate_rewards(self):
        group_targets_met = True
        environment_targets_met = True
        cost_targets_met = True
        consumption_targets_met = True
        nutrition_targets_met = True

        for metric in self.reward_dict.keys():
            if metric == 'step_penalty':
                continue
            elif self.reward_to_metric_mapping[metric] in self.reward_metrics:
                method = self.reward_to_function_mapping[metric]
                reward_func = getattr(RewardCalculator, method)
                self.reward_dict[metric], targets_met = reward_func(self)
                if metric == 'nutrient_reward':
                    nutrition_targets_met = targets_met
                elif metric == 'ingredient_group_count_reward':
                    group_targets_met = targets_met
                elif metric == 'ingredient_environment_count_reward':
                    environment_targets_met = targets_met
                elif metric == 'cost_reward':
                    cost_targets_met = targets_met
                elif metric == 'consumption_reward':
                    consumption_targets_met = targets_met    
                else:
                    print(f"Metric: {metric} not found in RewardCalculator.")   
                
            if metric == 'termination_reward':
                terminated, self.reward_dict[metric], targets_not_met = RewardCalculator.termination_reason(
                                                                                                        self,
                                                                                                        nutrition_targets_met,
                                                                                                        group_targets_met,
                                                                                                        environment_targets_met,
                                                                                                        cost_targets_met,
                                                                                                        consumption_targets_met,
                                                                                                    )
                self.reward_dict['targets_not_met'] = targets_not_met
                if metric == 'targets_not_met':
                    pass
                
        self.reward_dict['step_penalty'] = -1
        
        return terminated


    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        self.nsteps = 0
        
        self.episode_count += 1

        # Function will randomly initialise ingredients and reset metric to that of the initial ingredients
        self._initialise_selection()

        # Will reset rewards to zero for each metric
        self._initialize_rewards()

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.nsteps += 1
        
        action = np.clip(action, -1, 1)
        
        if self.verbose > 1:
            # Ensure the action is an array and has the correct shape
            if not isinstance(action, np.ndarray) or action.shape != (self.n_ingredients,):
                raise ValueError(f"Expected action to be an np.ndarray of shape {(self.n_ingredients,)}, but got {type(action)} with shape {action.shape}")
    
            if action.shape != self.action_space.shape:
                raise ValueError(f"Action shape {action.shape} is not equal to action space shape {self.action_space.shape}")
            
        change = action * self.action_scaling_factor
        
        self.current_selection = np.maximum(0, self.current_selection + change)

        num_selected_ingredients = np.sum(self.current_selection > 0)
        if num_selected_ingredients > self.max_ingredients:
            excess_indices = np.argsort(-self.current_selection)
            self.current_selection[excess_indices[self.max_ingredients:]] = 0

        reward, terminated = self._calculate_reward()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        obs = {}
        # Assuming current_selection should also be part of the observation dictionary
        obs['current_selection'] = np.array(self.current_selection, dtype=np.float32)
        if 'nutrients' in self.reward_metrics:
            obs['nutrients'] = np.array(list(self.nutrient_averages.values()), dtype=np.float32)
        if 'groups' in self.reward_metrics:
            obs['groups'] = np.array(list(self.ingredient_group_count.values()), dtype=np.float32)
        if 'environment' in self.reward_metrics:
            obs['environment'] = np.array(list(self.ingredient_environment_count.values()), dtype=np.float32)
        if 'cost' in self.reward_metrics:
            obs['cost'] = np.array(list(self.menu_cost.values()), dtype=np.float32)
        if 'consumption' in self.reward_metrics:
            obs['consumption'] = np.array(list(self.consumption_average.values()), dtype=np.float32)
            
        return obs


    def _get_info(self):
        
        self.target_not_met_counters.update(self.reward_dict['targets_not_met'])

        info = {
            'nutrient_averages': self.nutrient_averages,
            'ingredient_group_count': self.ingredient_group_count,
            'ingredient_environment_count': self.ingredient_environment_count,
            'consumption_average': self.consumption_average,
            'cost': self.menu_cost,
            'reward': self.reward_dict,
            'targets_not_met_count': dict(self.target_not_met_counters),
            'current_meal_plan': self._current_meal_plan()
        }

        return info

    def render(self):
        if self.render_mode == 'step':
            print(f"Step: {self.nsteps}")
        if self.render_mode == "human":
            print(f"Episode: {self.episode_count}")

    def close(self):
        pass
    
    def _current_meal_plan(self):
        
        self.current_meal_plan = {}
        
        if self.verbose > 0:
            
            nonzero_indices = np.nonzero(self.current_selection)
            
            if len(nonzero_indices[0]) != 0:
                for idx in nonzero_indices[0]:
                    category_value = self.ingredient_df['Category7'].iloc[idx]
                    self.current_meal_plan[category_value] = self.current_selection[idx]
    
    def _initialise_selection(self):
        # Initialize current_selection to all zeroes
        self.current_selection = np.zeros(self.n_ingredients)
        
        # Set the seed if provided
        if self.seed is not None:
            random.seed(self.seed)
        
        # Initialize a list to store indices
        selected_indices = []
        
        # Dictionary to store count of selected indices per category
        selected_counts = {category: 0 for category in self.group_info}
        
        # Total number of indices to select
        num_indices_to_select = min(self.max_ingredients, self.n_ingredients)
        
        # Sample indices based on probabilities for each category
        while len(selected_indices) < num_indices_to_select:
            # Choose a category based on its probability
            category = random.choices(list(self.group_info.keys()), 
                                    weights=[info['probability'] for info in self.group_info.values()])[0]
            
            # Select an index from the chosen category
            indices_to_sample = list(set(self.group_info[category]['indexes']) - set(selected_indices))
            
            if indices_to_sample:
                idx = random.choice(indices_to_sample)
                selected_indices.append(idx)
                selected_counts[category] += 1  # Increment count for selected indices in this category
        
        # Ensure we have exactly self.max_ingredients unique indices
        selected_indices = random.sample(selected_indices, min(self.max_ingredients, len(selected_indices)))
        
        # Assign the values to the selected indices
        values_to_assign = [random.randint(*self.ingredient_group_portion_targets[group]) for group in
                            ['fruit', 'veg', 'non_processed_protein', 'processed_protein', 'carbs', 'dairy']]
        for idx, value in zip(selected_indices, values_to_assign):
            self.current_selection[idx] = value
        if self.verbose > 1:
            # Print counts of selected indices for each group
            for category, count in selected_counts.items():
                print(f"Number of indices selected for '{category}': {count}")
                
        self._get_metrics()
        
        self._get_obs()
        
        

def final_meal_plan(ingredient_df, terminal_observation):
    
    current_selection = terminal_observation['current_selection']
    current_meal_plan = {}
    
    nonzero_indices = np.nonzero(current_selection)
    if len(nonzero_indices[0]) != 0:
        for idx in nonzero_indices[0]:
            category_value = ingredient_df['Category7'].iloc[idx]
            current_meal_plan[category_value] = current_selection[idx]
    
    return current_meal_plan


if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env
    from gymnasium.wrappers import TimeLimit, NormalizeObservation, NormalizeReward
    from stable_baselines3.common.monitor import Monitor
    from utils.train_utils import setup_environment, get_unique_directory, monitor_memory_usage, plot_reward_distribution, set_seed
    reward_dir, reward_prefix = get_unique_directory("saved_models/reward", 'reward_test34', '')

    class Args:
        reward_metrics = ['nutrients', 'groups', 'environment', 'consumption', 'cost']
        render_mode = None
        num_envs = 1
        plot_reward_history = False
        max_episode_steps = 1000
        verbose = 2
        action_scaling_factor = 10
        memory_monitor = True
        gamma = 0.99
        max_ingredients = 6
        action_scaling_factor = 10
        reward_save_interval = 1000
        vecnorm_norm_obs = True
        vecnorm_norm_reward = True
        vecnorm_clip_obs = 10
        vecnorm_clip_reward = 10
        vecnorm_epsilon = 1e-8
        vecnorm_norm_obs_keys = None
        ingredient_df = get_data()
        seed = 10
        
    args = Args()

    num_episodes = 1000
    
    reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
    
    env = setup_environment(args, reward_save_path=reward_save_path, eval=False)
    
    check_env(env.unwrapped.envs[0].unwrapped)
    
    print("Environment is valid!")
    
    del env
    
    env_kwargs = {
            "ingredient_df": args.ingredient_df,
            "max_ingredients": args.max_ingredients,
            "action_scaling_factor": args.action_scaling_factor,
            "render_mode": args.render_mode,
            "reward_metrics": args.reward_metrics,
            "seed": args.seed,
            "verbose": args.verbose
            }
        

    
    def make_env():
        
        env = gym.make("SchoolMealSelection-v0", **env_kwargs)
            
        # # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode. Need to repeat this so if i want to experiment with different steps.
        env = TimeLimit(env, max_episode_steps=1000)
        
        env = Monitor(env)  # wrap it with monitor env again to explicitely take the change into account
        
        return env     
    
    env = make_env()
    
    np.set_printoptions(suppress=True)

    if args.memory_monitor:
        import threading
        monitoring_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
        monitoring_thread.start()

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode {episode + 1}")

        terminated = False
        truncated = False
        n_steps = 0
        obs = env.reset()  # Reset the environment at the start of each episode

        while not terminated or not truncated:
            actions = env.action_space.sample()  # Sample actions
            obs, rewards, terminated, truncated, infos = env.step(actions)  # Step the environment
            
            if terminated or truncated:
                print(f"Final meal plan (episode: {episode} at {n_steps}):", 
                    final_meal_plan(args.ingredient_df, infos.get('terminal_observation', obs)))
                # Optionally print additional info at the end of each episode
                print(f"Episode {episode + 1} completed in {n_steps} steps.")
                break  # Break the loop when the episode is done

        env.close()  # Ensure the environment is closed properlyy

    if args.plot_reward_history:
        reward_prefix = reward_prefix.split(".")[0]
        dir, pref = get_unique_directory(reward_dir, f"{reward_prefix}_plot", '.png')
        plot_path = os.path.abspath(os.path.join(dir, pref))
        plot_reward_distribution(reward_save_path, plot_path)
