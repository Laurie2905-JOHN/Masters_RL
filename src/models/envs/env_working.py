import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from models.reward.reward_sparse_shaped_nutrient import RewardCalculator
import os
import random
from gymnasium.envs.registration import register
import torch
from collections import OrderedDict

register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env_working:SchoolMealSelection',
)

class SchoolMealSelection(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients=6, action_scaling_factor=10, render_mode=None, verbose=0, seed = None, initialization_strategy='zero', max_episode_steps=1000):
        super(SchoolMealSelection, self).__init__()

        self.ingredient_df = ingredient_df
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        self.max_episode_steps = max_episode_steps
            
        self.verbose = verbose
        self.seed = seed
        self.initialization_strategy = initialization_strategy

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
        self.Group_BC = ingredient_df['Group BC'].values.astype(np.float32)
        self.Group_D = ingredient_df['Group D'].values.astype(np.float32)
        self.Group_E = ingredient_df['Group E'].values.astype(np.float32)
        self.Bread = ingredient_df['Bread'].values.astype(np.float32)
        self.Confectionary = ingredient_df['Confectionary'].values.astype(np.float32)

        # Group target ranges
        self.ingredient_group_count_targets = {
            'fruit': 1,
            'veg': 1,
            'protein': 1,
            'carbs': 1,
            'dairy': 1,
            'bread': 1,
            'confectionary': 0
        }

        self.ingredient_group_portion_targets = {
            'fruit': (40, 110),
            'veg': (40, 110),
            'protein': (70, 150),
            'carbs': (25, 300),
            'dairy': (20, 200),
            'bread': (40, 90),
            'confectionary': (0, 50) # dummy targets to ensure all groups are included and are not assigned a zero value
        }
        

        # Define group indexes and probabilities together in a dictionary for initialisation after reset
        self.group_info = {
            'fruit': {'indexes': np.nonzero(self.Group_A_veg)[0], 'probability': 0.95},
            'veg': {'indexes': np.nonzero(self.Group_A_fruit)[0], 'probability': 0.95},
            'protein': {'indexes': np.nonzero(self.Group_BC)[0], 'probability': 0.7},
            'carbs': {'indexes': np.nonzero(self.Group_D)[0], 'probability': 0.95},
            'dairy': {'indexes': np.nonzero(self.Group_E)[0], 'probability': 0.95},
            'bread': {'indexes': np.nonzero(self.Bread)[0], 'probability': 0.95},
            'confectionary': {'indexes': np.nonzero(self.Confectionary)[0], 'probability': 0.01}
        }
        
        self.n_ingredients = len(ingredient_df)

        # Environmental ratings
        rating_to_int = {'A': 1, 'B': 0.5, 'C': 0, 'D': -0.5, 'E': -1}
        self.Animal_Welfare_Rating = np.array([rating_to_int[val] for val in ingredient_df['Animal Welfare Rating'].values], dtype=np.float32)
        self.Rainforest_Rating = np.array([rating_to_int[val] for val in ingredient_df['Rainforest Rating'].values], dtype=np.float32)
        self.Water_Scarcity_Rating = np.array([rating_to_int[val] for val in ingredient_df['Water Scarcity Rating'].values], dtype=np.float32)
        self.CO2_FU_Rating = np.array([rating_to_int[val] for val in ingredient_df['CO2 FU Rating'].values], dtype=np.float32)
        
        self.co2g = {'CO2_g': 0.0}
        self.CO2_g_per_1g = ingredient_df['CO2_g_per_100g'].values.astype(np.float32) / 100
        self.target_CO2_g_per_meal = 500

        # Consumption data
        self.Mean_g_per_day = ingredient_df['Mean_g_per_day'].values.astype(np.float32)
        self.StandardDeviation = ingredient_df['StandardDeviation'].values.astype(np.float32)
        self.Coefficient_of_Variation = ingredient_df['Coefficient of Variation'].values.astype(np.float32)

        # Cost data
        self.Cost_per_1g = ingredient_df['Cost_100g'].values.astype(np.float32) / 100
        self.menu_cost = {'cost': 0.0}
        self.target_cost_per_meal = 2


        self.render_mode = render_mode
        
        # Combine all indices from all categories
        self.all_indices = [i for i in range(1, self.n_ingredients)]
        
        self.episode_count = -1
        
        self.nsteps = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)
        
        # Initialize current_selection to all zeroes
        self.current_selection = np.zeros(self.n_ingredients)
        
        self.ingredient_group_count = {k: 0 for k in self.ingredient_group_count_targets.keys()}
        
        self.ingredient_group_portion = {k: 0.0 for k in self.ingredient_group_portion_targets.keys()}
        
        self.ingredient_environment_count = {
            'animal_welfare': 0,
            'rainforest': 0,
            'water': 0,
            'CO2_rating': 0,
        }
        
        self.consumption_average = {
            'average_mean_consumption': 0.0,
            'average_cv_ingredients': 0.0
        }
        
        # Initialize observation space
        self._initialize_observation_space()

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
            'co2g_reward': 'co2g_reward',
        }
        
        # Reward key name to name for reward metric mapping
        self.reward_to_metric_mapping = {
            'nutrient_reward': 'nutrients',
            'ingredient_group_count_reward': 'groups',
            'ingredient_environment_count_reward': 'environment',
            'cost_reward': 'cost_reward',
            'co2g_reward': 'co2g_reward',
            'consumption_reward': 'consumption',
            'targets_not_met': 'targets_not_met',
            'termination_reward': 'termination'
        }

    def _initialize_observation_space(self):

        # # Define the observation space
        # self.observation_space = spaces.Dict({
        #         'current_selection_value': spaces.Box(low=0, high=500, shape=(self.max_ingredients,), dtype=np.float32),
        #         'current_selection_index': spaces.MultiDiscrete([self.n_ingredients] * self.max_ingredients),
        #         'time_feature': spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.float32),
        #         'nutrients': spaces.Box(low=0, high=2000, shape=(len(self.nutrient_averages),), dtype=np.float32),
        #         'groups': spaces.MultiDiscrete([self.max_ingredients] * len(self.ingredient_group_count)),
        #         'environment_counts': spaces.MultiDiscrete([5] * len(self.ingredient_environment_count)),
        #         'cost': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
        #         'co2g': spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),  # 'CO2_g_per_1g
        #         'consumption': spaces.Box(low=0, high=100, shape=(len(self.consumption_average),), dtype=np.float32)
        #     })
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            'current_selection_value': spaces.Box(low=0, high=500, shape=(self.max_ingredients,), dtype=np.float64),
            'current_selection_index': spaces.Box(low=0, high=self.n_ingredients, shape=(self.max_ingredients,), dtype=np.float64),
            'time_feature': spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.float64),
            'nutrients': spaces.Box(low=0, high=2000, shape=(len(self.nutrient_averages),), dtype=np.float64),
            'groups': spaces.Box(low=0, high=self.max_ingredients, shape=(len(self.ingredient_group_count),), dtype=np.float64),
            'environment_counts': spaces.Box(low=-1, high=1, shape=(len(self.ingredient_environment_count),), dtype=np.float64),
            'cost': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float64),
            'co2g': spaces.Box(low=0, high=2000, shape=(1,), dtype=np.float64),
            'consumption': spaces.Box(low=0, high=100, shape=(len(self.consumption_average),), dtype=np.float64)
        })

    def _calculate_reward(self):

        reward = 0
        
        terminated = False
        
        # Calculating metrics based on current meal plan
        self._get_metrics()
        
        # Initialising reward dictionary to all zero
        self._initialize_rewards()
        
        terminated = self._evaluate_rewards()
        reward += sum(value for subdict in self.reward_dict.values() if isinstance(subdict, dict) for value in subdict.values() if isinstance(value, (int, float))) + \
            sum(value for value in self.reward_dict.values() if isinstance(value, (int, float)))
            
        return reward, terminated
    
    def _get_metrics(self):
        
        self.nutrient_averages = self._calculate_nutrient_averages()
        non_zero_mask = self.current_selection != 0
        non_zero_values = self._calculate_non_zero_values(non_zero_mask)
        self.ingredient_group_count = self._sum_dict_values(non_zero_values)
        self.ingredient_group_portion = self._sum_dict_values(non_zero_values, multiply_with=self.current_selection)
        self.ingredient_environment_count = self._calculate_ingredient_environment_count(non_zero_mask)
        self.menu_cost = self._calculate_cost()
        self.co2g = self._calculate_co2g()
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
            'protein': self.Group_BC * non_zero_mask,
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
        }

    def _calculate_consumption_average(self, non_zero_mask):
        return {
            'average_mean_consumption': sum(non_zero_mask * self.Mean_g_per_day) / self.max_ingredients,
            'average_cv_ingredients': sum(non_zero_mask * self.Coefficient_of_Variation) / self.max_ingredients
        }
    
    def _calculate_cost(self):
        return {
            'cost': sum(self.Cost_per_1g * self.current_selection),
        }
        
    def _calculate_co2g(self):
        return {
            'CO2_g': sum(self.CO2_g_per_1g * self.current_selection),
            }


    def _initialize_rewards(self):
        
        self.reward_dict = {}
        self.reward_dict['nutrient_reward'] = {k: 0 for k in self.nutrient_averages.keys()}
        self.reward_dict['ingredient_group_count_reward'] = {k: 0 for k in self.ingredient_group_count.keys()}
        self.reward_dict['ingredient_environment_count_reward'] = {k: 0 for k in self.ingredient_environment_count.keys()}
        self.reward_dict['cost_reward'] = {'cost': 0}
        self.reward_dict['co2g_reward'] = {'co2g': 0}
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

        terminate = False
        termination_reasons = []

        for metric in self.reward_dict.keys():
            if metric in ['step_penalty', 'targets_not_met', 'termination_reward']:
                continue
            method = self.reward_to_function_mapping[metric]
            reward_func = getattr(RewardCalculator, method)
            self.reward_dict[metric], targets_met, terminate = reward_func(self)
            if metric == 'nutrient_reward':
                nutrition_targets_met = targets_met
                nutrition_terminate = terminate
                if nutrition_terminate:
                    termination_reasons.append("nutrition_terminate")
            elif metric == 'ingredient_group_count_reward':
                group_targets_met = targets_met
                ingredient_group_count_terminate = terminate
                if ingredient_group_count_terminate:
                    termination_reasons.append("ingredient_group_count_terminate")
            elif metric == 'ingredient_environment_count_reward':
                environment_targets_met = targets_met
                ingredient_environment_count_terminate = terminate
                if ingredient_environment_count_terminate:
                    termination_reasons.append("ingredient_environment_count_terminate")                    
            elif metric == 'cost_reward':
                cost_targets_met = targets_met
                cost_terminate = terminate
                if cost_terminate:
                    termination_reasons.append("cost_terminate")                    
            elif metric == 'consumption_reward':
                consumption_targets_met = targets_met
                consumption_terminate = terminate
                if consumption_terminate:
                    termination_reasons.append("consumption_terminate")
            elif metric == 'co2g_reward':
                co2g_targets_met = targets_met
                co2g_terminate = terminate
                if co2g_terminate:
                    termination_reasons.append("co2g_terminate")
            else:
                raise ValueError(f"Metric: {metric} not found in RewardCalculator.")

        
        targets = {
            'Nutrition': nutrition_targets_met,
            'Group': group_targets_met,
            'Environment': environment_targets_met,
            'Cost': cost_targets_met,
            'Consumption': consumption_targets_met,
            'CO2G': co2g_targets_met
        }
        
        terminated, self.reward_dict['termination_reward'], targets_not_met = RewardCalculator.termination_reason(
                                                                                                    self,
                                                                                                    targets,
                                                                                                    termination_reasons
                                                                                                    )
        self.reward_dict['targets_not_met'] = targets_not_met

        self.reward_dict['step_penalty'] = -1
        
        return terminated

    def reset(self, seed=None, options=None):
        
        if self.verbose > 1:
            current_meal_plan, _, _ = self._current_meal_plan()
            print(f"\nFinal plan: {current_meal_plan}")
            # Print portions of selected food groups
            print(f"\nFinal portion size of groups: {self.ingredient_group_portion}"),
        
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
        
        _, current_selection_index, current_selection_value = self._current_meal_plan()

        time_feature = 1 - (self.nsteps / self.max_episode_steps)

        obs = {}
        # Assuming current_selection should also be part of the observation dictionary      
        obs['current_selection_value'] = np.array(current_selection_value, dtype=np.float64)
        obs['current_selection_index'] = np.array(current_selection_index, dtype=np.float64)
        obs['time_feature'] = np.array([time_feature], dtype=np.float64)
        obs['nutrients'] = np.array(list(self.nutrient_averages.values()), dtype=np.float64)
        obs['groups'] = np.array(list(self.ingredient_group_count.values()), dtype=np.float64)
        obs['environment_counts'] = np.array(list(self.ingredient_environment_count.values()), dtype=np.float64)
        obs['cost'] = np.array(list(self.menu_cost.values()), dtype=np.float64)
        obs['co2g'] = np.array(list(self.co2g.values()), dtype=np.float64)
        obs['consumption'] = np.array(list(self.consumption_average.values()), dtype=np.float64)

        return obs


    def _get_info(self):
        
        self.target_not_met_counters.update(self.reward_dict['targets_not_met'])
        current_meal_plan, _, _ = self._current_meal_plan()

        info = {
            'nutrient_averages': self.nutrient_averages,
            'ingredient_group_count': self.ingredient_group_count,
            'ingredient_environment_count': self.ingredient_environment_count,
            'consumption_average': self.consumption_average,
            'cost': self.menu_cost,
            'co2g': self.co2g,
            'reward': self.reward_dict,
            'group_portions': self.ingredient_group_portion,
            'targets_not_met_count': dict(self.target_not_met_counters),
            'current_meal_plan': current_meal_plan
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
        current_meal_plan = {}

        # Get non-zero indices and values
        nonzero_indices = np.nonzero(self.current_selection)[0]
        nonzero_values = self.current_selection[nonzero_indices]

        # Ensure there is at least one zero value and index
        zero_indices = np.where(self.current_selection == 0)[0]

        num_selected_ingredients = len(nonzero_indices)

        # Ensure the number of max_ingredients are included
        if num_selected_ingredients != self.max_ingredients:
            additional_indices_needed = self.max_ingredients - num_selected_ingredients
            
            # Handle cases where there are not enough zero indices
            if additional_indices_needed > len(zero_indices):
                raise ValueError("Not enough zero indices to reach the required max_ingredients.")

            additional_indices = np.random.choice(
                zero_indices,
                size=additional_indices_needed,
                replace=False
            )
            nonzero_indices = np.concatenate((nonzero_indices, additional_indices))
            nonzero_values = np.concatenate((nonzero_values, self.current_selection[additional_indices]))

        for idx, value in zip(nonzero_indices, nonzero_values):
            category_value = self.ingredient_df['Category7'].iloc[idx]
            current_meal_plan[category_value] = value

        return current_meal_plan, nonzero_indices, nonzero_values


        
    def _initialise_selection(self):
        # Initialize current_selection to all zeroes
        self.current_selection = np.zeros(self.n_ingredients)

        # Create a separate random number generator instance to ensure meal plan is always initialized randomly
        rng = random.Random(None)  # Set seed to None to get a random seed even if seed is set
        # Initialize a list to store indices
        selected_indices = []

        # Dictionary to store count of selected indices per category
        selected_counts = {category: 0 for category in self.ingredient_group_count_targets}

        # Total number of indices to select
        num_indices_to_select = min(self.max_ingredients, self.n_ingredients)

        if self.initialization_strategy == 'perfect':
            total_target_count = sum(self.ingredient_group_count_targets.values())
            if total_target_count > num_indices_to_select:
                raise ValueError(f"Total target counts {total_target_count} exceed max ingredients {num_indices_to_select}")

            # Select indices based on group targets
            for key, value in self.ingredient_group_count_targets.items():
                for _ in range(value):
                    selected_index = rng.choice(self.group_info[key]['indexes'])
                    selected_indices.append(selected_index)
                    selected_counts[key] += 1
            
            if num_indices_to_select < self.max_ingredients:
                remaining_indices = set(self.all_indices) - set(selected_indices)
                selected_indices.extend(rng.sample(remaining_indices, num_indices_to_select - total_target_count))
                # Randomly select remaining indices to reach max_ingredients
                remaining_count = num_indices_to_select - total_target_count
                if remaining_count > 0:
                    remaining_indices = set(self.all_indices) - set(selected_indices)
                    selected_indices.extend(rng.sample(remaining_indices, remaining_count))
                raise Warning(f"Total target counts {total_target_count} less than max ingredients {num_indices_to_select} so value was added randomly")

        elif self.initialization_strategy == "zero":
            # If zero initialization is selected, keep current_selection as all zeros
            pass

        elif self.initialization_strategy == "probabilistic":
            # Ensure we have exactly self.max_ingredients unique indices
            selected_indices = rng.sample(self.all_indices, num_indices_to_select)
            # Increment the selected counts for each category
            for idx in selected_indices:
                for category, info in self.group_info.items():
                    if idx in info['indexes']:
                        selected_counts[category] += 1
                        break

        else:
            raise ValueError(f"Invalid value for initialization strategy: {self.initialization_strategy}")

        if self.initialization_strategy != "zero":
            # Assign the values to the selected indices
            values_to_assign = []
            for group, count in selected_counts.items():
                if count > 0:
                    # Generate values and ensure none are zero
                    for _ in range(count):
                        value = rng.randint(*self.ingredient_group_portion_targets[group])
                        while value == 0:
                            value = rng.randint(*self.ingredient_group_portion_targets[group])
                        values_to_assign.append(value)

            for idx, value in zip(selected_indices, values_to_assign):
                self.current_selection[idx] = value


        if self.verbose > 1:
            current_meal_plan, _, _ = self._current_meal_plan()
            print(f"\nInitialized plan: {current_meal_plan}")
            # Print counts of selected indices for each group
            for category, count in selected_counts.items():
                print(f"Number of indices selected for '{category}': {count}")

        self._get_metrics()
        self._get_obs()
        
        

# def final_meal_plan(ingredient_df, terminal_observation):
    
#     nonzero_indices = terminal_observation['current_selection_index']
#     non_zero_values = terminal_observation['current_selection_value']
#     current_meal_plan = {}
    
#     if len(nonzero_indices) != 0:
#         for idx in nonzero_indices:
#             category_value = ingredient_df['Category7'].iloc[idx]
#             current_meal_plan[category_value] = non_zero_values[idx]
    
#     return current_meal_plan


if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env
    from gymnasium.wrappers import TimeLimit, NormalizeObservation, NormalizeReward
    from stable_baselines3.common.monitor import Monitor
    from utils.train_utils import setup_environment, get_unique_directory, monitor_memory_usage, plot_reward_distribution, set_seed
    reward_dir, reward_prefix = get_unique_directory("saved_models/reward", 'reward_test34', '')

    class Args:
        render_mode = None
        num_envs = 2
        plot_reward_history = False
        max_episode_steps = 1000
        verbose = 0
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
        ingredient_df = get_data("small_data.csv")
        seed = 10
        env_name = 'SchoolMealSelection-v0'
        initialization_strategy = 'perfect'
        vecnorm_norm_obs_keys = ['current_selection_value', 'current_selection_index', 'groups', 'cost', 'environment_counts', 'consumption', 'co2g', 'nutrients'] # time feature already normalized
    args = Args()

    num_episodes = 1000
    
    reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
    
    env = setup_environment(args, reward_save_path=reward_save_path, eval=False)

    check_env(env.envs[0].unwrapped)

    print("Environment is valid!")
    
    # del env
    
    env_kwargs = {
            "ingredient_df": args.ingredient_df,
            "max_ingredients": args.max_ingredients,
            "action_scaling_factor": args.action_scaling_factor,
            "render_mode": args.render_mode,
            "seed": args.seed,
            'initialization_strategy': args.initialization_strategy,
            "verbose": args.verbose
            }

    # Function to test the env step without normalization
    def make_env():
        
        env = gym.make(args.env_name, **env_kwargs)
            
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
        while not terminated and not truncated:
            actions = env.action_space.sample()  # Sample actions
            n_steps += 1
            obs, reward, terminated, truncated, get_info = env.step(actions)

        print(f"Episode {episode + 1} completed in {n_steps} steps.")

    env.close()  # Ensure the environment is closed properly

    if args.plot_reward_history:
        reward_prefix = reward_prefix.split(".")[0]
        dir, pref = get_unique_directory(reward_dir, f"{reward_prefix}_plot", '.png')
        plot_path = os.path.abspath(os.path.join(dir, pref))
        plot_reward_distribution(reward_save_path, plot_path)
