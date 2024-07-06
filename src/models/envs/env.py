import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from models.reward import reward
from models.initialization.initialization import IngredientSelectionInitializer
from gymnasium.envs.registration import register
import os
from collections import deque

class BaseEnvironment(gym.Env):
    """
    A base class for custom Gym environments.
    Contains common functionalities for initializing and managing the environment.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients: int = 6, action_scaling_factor: int = 10, render_mode: str = None, 
                 verbose: int = 0, seed: int = None, reward_type: str = 'sparse', 
                 initialization_strategy: str = 'zero', max_episode_steps: int = 100):
        super().__init__()

        self._initialize_parameters(ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, initialization_strategy, max_episode_steps)
        self._initialize_nutrient_values(ingredient_df)
        self._initialize_nutrient_targets()
        self._create_index_value_mapping()
        self._initialize_group_categories(ingredient_df)
        self._initialize_group_targets()
        self._initialize_environmental_ratings(ingredient_df)
        self._initialize_consumption_data(ingredient_df)
        self._initialize_cost_data(ingredient_df)
        self._initialize_selection_variables()
        self.reward_calculator = self._get_reward_calculator(reward_type)
        self.ingredient_initializer = IngredientSelectionInitializer()
        self.initialize_action_space()
        self._initialize_observation_space()
        
        self.reward_to_function_mapping = {
            'nutrient_reward': 'calculate_nutrient_reward',
            'ingredient_group_count_reward': 'calculate_group_count_reward',
            'ingredient_environment_count_reward': 'calculate_environment_count_reward',
            'cost_reward': 'calculate_cost_reward',
            'consumption_reward': 'calculate_consumption_reward',
            'co2g_reward': 'calculate_co2g_reward',
            'step_reward': 'calculate_step_reward'
        }
        
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
    
    def _initialize_parameters(self, ingredient_df, max_ingredients: int, action_scaling_factor: int, render_mode: str, 
                               verbose: int, seed: int, initialization_strategy: str, max_episode_steps: int) -> None:
        self.ingredient_df = ingredient_df
        self.max_ingredients = max_ingredients
        self.action_scaling_factor = action_scaling_factor
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self.seed = seed
        self.initialization_strategy = initialization_strategy
        self.n_ingredients = len(ingredient_df)
        self.render_mode = render_mode
        self.episode_count = -1
        self.nsteps = 0

    def _initialize_nutrient_values(self, ingredient_df) -> None:
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values.astype(np.float32) / 100
        self.fat_g = ingredient_df['Fat_g'].values.astype(np.float32) / 100
        self.saturates_g = ingredient_df['Saturates_g'].values.astype(np.float32) / 100
        self.carbs_g = ingredient_df['Carbs_g'].values.astype(np.float32) / 100
        self.sugars_g = ingredient_df['Sugars_g'].values.astype(np.float32) / 100
        self.fibre_g = ingredient_df['Fibre_g'].values.astype(np.float32) / 100
        self.protein_g = ingredient_df['Protein_g'].values.astype(np.float32) / 100
        self.salt_g = ingredient_df['Salt_g'].values.astype(np.float32) / 100

    def _initialize_nutrient_targets(self) -> None:
        self.target_calories = 530
        self.target_fat_g = 20.6
        self.target_saturates_g = 6.5
        self.target_carbs_g = 70.6
        self.target_sugars_g = 15.5
        self.target_fibre_g = 4.2
        self.target_protein_g = 7.5
        self.target_salt_g = 1.5  # Temporary target

        self.nutrient_target_ranges = {
            'calories': (self.target_calories * 0.95, self.target_calories * 1.05),
            'fat': (self.target_fat_g * 0.1, self.target_fat_g),
            'saturates': (0, self.target_saturates_g),
            'carbs': (self.target_carbs_g, self.target_carbs_g * 3),
            'sugar': (0, self.target_sugars_g),
            'fibre': (self.target_fibre_g, self.target_fibre_g * 3),
            'protein': (self.target_protein_g, self.target_protein_g * 3),
            'salt': (0, self.target_salt_g)
        }

        self.nutrient_averages = {k: 0.0 for k in self.nutrient_target_ranges.keys()}

    def _initialize_group_categories(self, ingredient_df) -> None:
        self.group_a_veg = ingredient_df['Group A veg'].values.astype(np.float32)
        self.group_a_fruit = ingredient_df['Group A fruit'].values.astype(np.float32)
        self.group_bc = ingredient_df['Group BC'].values.astype(np.float32)
        self.group_d = ingredient_df['Group D'].values.astype(np.float32)
        self.group_e = ingredient_df['Group E'].values.astype(np.float32)
        self.bread = ingredient_df['Bread'].values.astype(np.float32)
        self.confectionary = ingredient_df['Confectionary'].values.astype(np.float32)
    

    def _initialize_group_targets(self) -> None:
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
            'fruit': (20, 110),
            'veg': (20, 110),
            'protein': (40, 150),
            'carbs': (20, 300),
            'dairy': (20, 200),
            'bread': (30, 100),
            'confectionary': (0, 50)
        }


        self.group_info = {
            'fruit': {'indexes': np.nonzero(self.group_a_fruit)[0], 'probability': 0.8},
            'veg': {'indexes': np.nonzero(self.group_a_veg)[0], 'probability': 0.8},
            'protein': {'indexes': np.nonzero(self.group_bc)[0], 'probability': 0.8},
            'carbs': {'indexes': np.nonzero(self.group_d)[0], 'probability': 0.8},
            'dairy': {'indexes': np.nonzero(self.group_e)[0], 'probability': 0.8},
            'bread': {'indexes': np.nonzero(self.bread)[0], 'probability': 0.8},
            'confectionary': {'indexes': np.nonzero(self.confectionary)[0], 'probability': 0.01}
        }

    def _initialize_environmental_ratings(self, ingredient_df) -> None:
        self.rating_to_int = {'A': 1, 'B': 0.5, 'C': 0, 'D': -0.5, 'E': -1}
        self.animal_welfare_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Animal Welfare Rating'].values], dtype=np.float32)
        self.rainforest_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Rainforest Rating'].values], dtype=np.float32)
        self.water_scarcity_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Water Scarcity Rating'].values], dtype=np.float32)
        self.co2_fu_rating = np.array([self.rating_to_int[val] for val in ingredient_df['CO2 FU Rating'].values], dtype=np.float32)
        self.co2_g = {'co2_g': 0.0}
        self.co2_g_per_1g = ingredient_df['CO2_g_per_100g'].values.astype(np.float32) / 100
        self.target_co2_g_per_meal = 1000  # Temporary CO2 target

    def _initialize_consumption_data(self, ingredient_df) -> None:
        self.mean_g_per_day = ingredient_df['Mean_g_per_day'].values.astype(np.float32)
        self.standard_deviation = ingredient_df['StandardDeviation'].values.astype(np.float32)
        self.coefficient_of_variation = ingredient_df['Coefficient of Variation'].values.astype(np.float32)

    def _initialize_cost_data(self, ingredient_df) -> None:
        self.cost_per_1g = ingredient_df['Cost_100g'].values.astype(np.float32) / 100
        self.menu_cost = {'cost': 0.0}
        self.target_cost_per_meal = 2

    def _initialize_selection_variables(self) -> None:
        self.all_indices = list(self.ingredient_df.index)
        self.current_selection = np.zeros(self.n_ingredients, dtype=np.float32)
        self.ingredient_group_count = {k: 0 for k in self.ingredient_group_count_targets.keys()}
        self.ingredient_group_portion = {k: 0.0 for k in self.ingredient_group_portion_targets.keys()}
        self.ingredient_environment_count = {
            'animal_welfare': 0,
            'rainforest': 0,
            'water': 0,
            'co2_rating': 0,
        }
        self.consumption_average = {
            'average_mean_consumption': 0.0,
            'average_cv_ingredients': 0.0
        }
        self.consumption_target = {
            'average_mean_consumption': 5.8,
            'average_cv_ingredients': 8.5
        }
        self.target_not_met_counters = Counter()
        
        self.targets_not_met_history = deque(maxlen=50)

        self.total_quantity_ratio = self.current_selection / (sum(self.current_selection) + 1e-9)
        
    def _create_index_value_mapping(self) -> None:
        """
        Generate evenly spaced values between 0 and 1 for ingredient indices.
        Creates a mapping of original indices to new values.
        """
        new_values = np.linspace(0, 1, self.n_ingredients)
        self.index_value_mapping = {i: new_values[i] for i in range(self.n_ingredients)}
        # Adding case for when ingredients are not selected yet
        self.index_value_mapping[-1] = -1

    def _get_reward_calculator(self, reward_type: str):
        reward_calculator_mapping = {
            'shaped': reward.ShapedRewardCalculator,
            'sparse': reward.SparseRewardCalculator,
            # Add other mappings as needed
        }
        return reward_calculator_mapping[reward_type]
    
    def _initialize_observation_space(self) -> None:
        """
        Define the observation space for the environment.
        The observation space is a dictionary containing various components such as:
        current selection values, indices, time feature, nutrient values, group counts, environment counts, cost, and CO2 emissions.
        """
        self.observation_space = spaces.Dict({
            'current_selection_value': spaces.Box(low=0, high=1000, shape=(self.max_ingredients,), dtype=np.float64),
            'current_selection_index': spaces.Box(low=-1, high=1, shape=(self.max_ingredients,), dtype=np.float64),
            'time_feature': spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.float64),
            'nutrients': spaces.Box(low=0, high=3000, shape=(len(self.nutrient_averages),), dtype=np.float64),
            'groups': spaces.Box(low=0, high=self.max_ingredients, shape=(len(self.ingredient_group_count),), dtype=np.float64),
            'environment_counts': spaces.Box(low=-1, high=1, shape=(len(self.ingredient_environment_count),), dtype=np.float64),
            'cost': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float64),
            'co2_g': spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float64),
            'consumption': spaces.Box(low=0, high=100, shape=(len(self.consumption_average),), dtype=np.float64)
        })

    def calculate_reward(self, action) -> tuple:  
          
        """
        Calculate the rewards based on the current state of the environment.
        Calculates rewards for each metric and checks if termination conditions are met.
        Returns a boolean indicating if the episode should be terminated.
        """
        # Initialize flags for target metrics
        target_flags = {
            'Nutrition': True,
            'Group': True,
            'Environment': True,
            'Cost': True,
            'Consumption': True,
            'CO2G': True
        }

        termination_reasons = []  # List to store reasons for termination

        # Iterate through each reward in the reward dictionary
        for reward in self.reward_dict.keys():
            # Skip step reward, targets not met, and termination reward rewards
            if reward in ['step_reward', 'targets_not_met', 'termination_reward', 'action_reward']:
                continue
            
            # Get the corresponding reward function from the RewardCalculator
            method = self.reward_to_function_mapping[reward]

            reward_func = getattr(self.reward_calculator, method)
            
            # Calculate the reward, targets met, and termination condition for the current reward
            self.reward_dict[reward], targets_met, terminate = reward_func(self)
            
            # Update the target met flags and termination reasons based on the reward
            termination_reasons, target_flags = self._update_flags_and_reasons(reward, targets_met, terminate, target_flags, termination_reasons)
        
        return termination_reasons, target_flags
    
    def sum_reward_values(self) -> float:
        # Sum the reward dictionary values to get the total reward
        return sum(value for subdict in self.reward_dict.values() if isinstance(subdict, dict) for value in subdict.values() if isinstance(value, (int, float))) + \
            sum(value for value in self.reward_dict.values() if isinstance(value, (int, float)))
            
    def get_metrics(self) -> None:
        """
        Calculate various metrics based on the current selection of ingredients.
        Metrics include nutrient averages, ingredient group counts and portions, environmental impact, cost, and consumption averages.
        """
        self.total_quantity_ratio = self.current_selection / (sum(self.current_selection) + 1e-9)
        self.nutrient_averages = self._calculate_nutrient_averages()
        non_zero_mask = self.current_selection != 0
        non_zero_values = self._calculate_non_zero_values(non_zero_mask)
        self.ingredient_group_count = self._sum_dict_values(non_zero_values)
        self.ingredient_group_portion = self._sum_dict_values(non_zero_values, multiply_with=self.current_selection)
        self.ingredient_environment_count = self._calculate_ingredient_environment_count()
        self.menu_cost = self._calculate_cost()
        self.co2_g = self._calculate_co2g()
        self.consumption_average = self._calculate_consumption_average()

    def _calculate_nutrient_averages(self) -> dict:
        """
        Calculate the average values of various nutrients based on the current selection of ingredients.
        Returns a dictionary of nutrient averages.
        """
        return {
            'calories': sum(self.caloric_values * self.current_selection),
            'fat': sum(self.fat_g * self.current_selection),
            'saturates': sum(self.saturates_g * self.current_selection),
            'carbs': sum(self.carbs_g * self.current_selection),
            'sugar': sum(self.sugars_g * self.current_selection),
            'fibre': sum(self.fibre_g * self.current_selection),
            'protein': sum(self.protein_g * self.current_selection),
            'salt': sum(self.salt_g * self.current_selection)
        }

    def _calculate_non_zero_values(self, non_zero_mask: np.ndarray) -> dict:
        """
        Calculate non-zero values for ingredient groups based on the current selection mask.
        Returns a dictionary of non-zero values for each ingredient group.
        """
        return {
            'fruit': self.group_a_fruit * non_zero_mask,
            'veg': self.group_a_veg * non_zero_mask,
            'protein': self.group_bc * non_zero_mask,
            'carbs': self.group_d * non_zero_mask,
            'dairy': self.group_e * non_zero_mask,
            'bread': self.bread * non_zero_mask,
            'confectionary': self.confectionary * non_zero_mask,
        }

    def _sum_dict_values(self, data: dict, multiply_with: np.ndarray = None) -> dict:
        """
        Sum values in a dictionary, optionally multiplying each value with a corresponding array.
        If multiply_with is provided, it multiplies each value with the corresponding element in multiply_with before summing.
        Returns a dictionary of summed values.
        """
        if multiply_with is None:
            return {key: sum(values) for key, values in data.items()}
        else:
            return {key: sum(values * multiply_with) for key, values in data.items()}


    def _calculate_ingredient_environment_count(self) -> dict:
        """
        Calculate the environmental impact counts for the current selection of ingredients.
        Returns a dictionary of rounded environmental ratings for each impact category.
        """
        def _round_to_nearest_rating(value: float) -> float:
            values = list(self.rating_to_int.values())
            closest_value = min(values, key=lambda x: abs(x - value))
            return closest_value
                
        return {
            'animal_welfare': _round_to_nearest_rating(sum(self.animal_welfare_rating * self.total_quantity_ratio)),
            'rainforest': _round_to_nearest_rating(sum(self.rainforest_rating * self.total_quantity_ratio)),
            'water': _round_to_nearest_rating(sum(self.water_scarcity_rating * self.total_quantity_ratio)),
            'co2_rating': _round_to_nearest_rating(sum(self.co2_fu_rating * self.total_quantity_ratio)),
        }

    def _calculate_consumption_average(self) -> dict:
        """
        Calculate the average consumption values based on the current selection of ingredients.
        Returns a dictionary of average mean consumption and coefficient of variation.
        """
        return {
            'average_mean_consumption': sum(self.mean_g_per_day * self.total_quantity_ratio),
            'average_cv_ingredients': sum(self.coefficient_of_variation * self.total_quantity_ratio)
        }

    def _calculate_cost(self) -> dict:
        """
        Calculate the total cost based on the current selection of ingredients.
        Returns a dictionary with the total cost.
        """
        return {
            'cost': sum(self.cost_per_1g * self.current_selection),
        }
        
    def _calculate_co2g(self) -> dict:
        """
        Calculate the total CO2 emissions based on the current selection of ingredients.
        Returns a dictionary with the total CO2 emissions.
        """
        return {
            'co2_g': sum(self.co2_g_per_1g * self.current_selection),
        }

    def initialize_rewards(self) -> None:
        """
        Initialize the reward dictionary with default values.
        Each reward component is set to zero.
        """
        self.reward_dict = {}
        self.reward_dict['nutrient_reward'] = {k: 0 for k in self.nutrient_averages.keys()}
        self.reward_dict['ingredient_group_count_reward'] = {k: 0 for k in self.ingredient_group_count.keys()}
        self.reward_dict['ingredient_environment_count_reward'] = {k: 0 for k in self.ingredient_environment_count.keys()}
        self.reward_dict['cost_reward'] = {'cost': 0}
        self.reward_dict['co2g_reward'] = {'co2_g': 0}
        self.reward_dict['consumption_reward'] = {'average_mean_consumption': 0, 'cv_reward': 0}
        self.reward_dict['targets_not_met'] = []
        self.reward_dict['termination_reward'] = 0
        self.reward_dict['step_reward'] = 0
        self.reward_dict['action_reward'] = 0

    def _update_flags_and_reasons(self, reward: str, targets_met: bool, terminate: bool, target_flags: dict, termination_reasons: list) -> None:
        """
        Helper function to update target flags and termination reasons based on the reward.
        """
        reward_to_flag = {
            'nutrient_reward': 'Nutrition',
            'ingredient_group_count_reward': 'Group',
            'ingredient_environment_count_reward': 'Environment',
            'cost_reward': 'Cost',
            'consumption_reward': 'Consumption',
            'co2g_reward': 'CO2G'
        }

        if reward in reward_to_flag:
            flag = reward_to_flag[reward]
            target_flags[flag] = targets_met
            if terminate:
                termination_reasons.append(f"{flag.lower()}_terminate")
        else:
            raise ValueError(f"Reward Type: {reward} not found in RewardCalculator.")

        return termination_reasons, target_flags

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        Reset the environment to its initial state.
        Resets the episode count, steps, and initializes the selection of ingredients.
        Also resets the rewards and renders the environment if required.
        """
        # If verbosity level is greater than 1, print the final plan and portion sizes
        if self.verbose > 1:
            current_meal_plan, _, _ = self.get_current_meal_plan()
            print(f"\nFinal plan (Step: {self.nsteps}): {current_meal_plan}")
            print(f"\nFinal portion size of groups (Step: {self.nsteps}): {self.ingredient_group_portion}"),
        
        
        # Call the parent class's reset method with the provided seed
        super().reset(seed=seed)

        # Reset the number of steps taken to 0
        self.nsteps = 0
        
        # Increment the episode count
        self.episode_count += 1

        # Initialize the selection of ingredients and reset metrics to initial values
        self._initialize_current_selection()

        # Reset rewards to zero for each metric
        self.initialize_rewards()

        # If render mode is 'human', render the environment
        if self.render_mode == 'human':
            self.render()

        # Return the initial observation and information
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one time step within the environment.
        Updates the current selection of ingredients based on the action taken.
        Calculates the reward and checks for termination conditions.
        Returns the new observation, reward, termination flag, truncated flag, and additional info.
        """
        terminated = False  # Initialize termination flag

        # Increment the number of steps taken
        self.nsteps += 1
        
        reward, terminated = self.validate_process_action_calculate_reward(action)

        # Return the new observation, reward, termination flag, truncated flag, and additional info
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def cut_current_selection(self) -> bool:
        # Ensure that the number of selected ingredients does not exceed the maximum allowed
        num_selected_ingredients = np.sum(self.current_selection > 0)
        if num_selected_ingredients > self.max_ingredients:
            
            if self.verbose > 1:
                print("Current Selection:", self.get_current_meal_plan())
            
            # Sort and cut extra indexes
            excess_indices = np.argsort(-self.current_selection)
            self.current_selection[excess_indices[self.max_ingredients:]] = 0
            
            if self.verbose > 1:
                print("Cut Current Selection:", self.get_current_meal_plan())
        
    def _get_obs(self) -> dict:
        """
        Get the current observation of the environment.
        Includes various components such as current selection values, indices, time feature, nutrient values, group counts, environment counts, cost, and CO2 emissions.
        Returns the observation dictionary.
        """
        # Get current meal plan details
        _, current_selection_index, current_selection_value = self.get_current_meal_plan()
        
        # Map the current selection index to new values
        mapped_index_value = [self.index_value_mapping[index] for index in current_selection_index]
        
        # Calculate the time feature as a fraction of steps remaining
        time_feature = 1 - (self.nsteps / self.max_episode_steps)

        # Create the observation dictionary with various components
        obs = {}
        obs['current_selection_value'] = np.array(current_selection_value, dtype=np.float64)
        obs['current_selection_index'] = np.array(mapped_index_value, dtype=np.float64)
        obs['time_feature'] = np.array([time_feature], dtype=np.float64)
        obs['nutrients'] = np.array(list(self.nutrient_averages.values()), dtype=np.float64)
        obs['groups'] = np.array(list(self.ingredient_group_count.values()), dtype=np.float64)
        obs['environment_counts'] = np.array(list(self.ingredient_environment_count.values()), dtype=np.float64)
        obs['cost'] = np.array(list(self.menu_cost.values()), dtype=np.float64)
        obs['co2_g'] = np.array(list(self.co2_g.values()), dtype=np.float64)
        obs['consumption'] = np.array(list(self.consumption_average.values()), dtype=np.float64)

        return obs

    def _get_info(self) -> dict:
        """
        Get additional information about the current state of the environment.
        Includes nutrient averages, ingredient group counts, environmental impact, cost, CO2 emissions, rewards, group portions, targets not met count, and the current meal plan.
        Returns the info dictionary.
        """
        # Update the counters for targets not met with the current reward dictionary values
        self.target_not_met_counters.update(self.reward_dict['targets_not_met'])
        
        # Calculate the running average
        running_average_targets_not_met = sum(self.targets_not_met_history) / len(self.targets_not_met_history) if self.targets_not_met_history else 0
        
        # Get current meal plan details
        current_meal_plan, _, _ = self.get_current_meal_plan()

        # Create the info dictionary with various components
        info = {
            'nutrient_averages': self.nutrient_averages,
            'ingredient_group_count': self.ingredient_group_count,
            'ingredient_environment_count': self.ingredient_environment_count,
            'consumption_average': self.consumption_average,
            'cost': self.menu_cost,
            'co2_g': self.co2_g,
            'reward': self.reward_dict,
            'group_portions': self.ingredient_group_portion,
            'targets_not_met_count': dict(self.target_not_met_counters),
            'Running_average_count_targets_not_met': running_average_targets_not_met,
            'current_meal_plan': current_meal_plan
        }

        return info

    def render(self) -> None:
        """
        Render the current state of the environment.
        Displays the current step or episode based on the render mode.
        """
        if self.render_mode == 'step':
            print(f"Step: {self.nsteps}")
        if self.render_mode == "human":
            print(f"Episode: {self.episode_count}")

    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        pass

    def get_current_meal_plan(self) -> tuple:
        """
        Get the current meal plan based on the selected ingredients.
        Ensures that the number of selected ingredients matches the max_ingredients.
        Returns the current meal plan, non-zero indices, and non-zero values.
        """
        current_meal_plan = {}

        # Get non-zero indices and values
        nonzero_indices = np.nonzero(self.current_selection)[0]
        nonzero_values = self.current_selection[nonzero_indices]

        # Ensure the number of selected ingredients matches max_ingredients
        num_selected_ingredients = len(nonzero_indices)

        if num_selected_ingredients < self.max_ingredients:
            # Calculate the number of zeros needed
            zeros_needed = self.max_ingredients - num_selected_ingredients

            # Create an array of zero values to fill the remaining spaces
            zero_values = np.zeros(zeros_needed, dtype=self.current_selection.dtype)

            # Append zero values to the selected values
            filled_values = np.concatenate((nonzero_values, zero_values))

            # For indices, append -1 to indicate these are zero-filled entries
            filled_indices = np.concatenate((nonzero_indices, -1 * np.ones(zeros_needed, dtype=int)))
        else:
            filled_values = nonzero_values
            filled_indices = nonzero_indices

        # Create the current meal plan dictionary with category values
        for idx, value in zip(filled_indices, filled_values):
            if idx != -1:  # Ignore the padded indices
                category_value = self.ingredient_df['Category7'].iloc[idx]
                current_meal_plan[category_value] = value

        return current_meal_plan, filled_indices, filled_values
        
    def _initialize_current_selection(self) -> None:
        """
        Initialize the selection of ingredients based on the chosen strategy.
        Supports different initialization strategies like 'perfect', 'zero', and 'probabilistic'.
        Updates the current selection and calculates initial metrics.
        """
        self.current_selection = np.zeros(self.n_ingredients, dtype=np.float32)
        
        self.current_selection = self.ingredient_initializer.initialize_selection(self)

        # Verbose output to show the initialized plan
        if self.verbose > 1:
            current_meal_plan, _, _ = self.get_current_meal_plan()
            print(f"Initialized plan: {current_meal_plan}")

        # Calculate initial metrics and update observations
        self.get_metrics()
        self._get_obs()
        
    def print_current_meal_plan(self) -> None:
        # If verbosity level is greater than 1, print the step plan and portion sizes
        if self.verbose > 1:
            current_meal_plan, _, _ = self.get_current_meal_plan()
            print(f"\nPlan at (Step: {self.nsteps}): {current_meal_plan}")
            print(f"Portion size of groups (Step: {self.nsteps}): {self.ingredient_group_portion}\n")

    def validate_action_shape(self, action: np.ndarray, shape: int) -> None:
        """
        Validate the shape of the action array.
        """
        if self.verbose > 1:
            if not isinstance(action, np.ndarray) or action.shape != (shape,):
                raise ValueError(f"Expected action to be an np.ndarray of shape {(shape,)}, but got {type(action)} with shape {action.shape}")
            if action.shape != self.action_space.shape:
                raise ValueError(f"Action shape {action.shape} is not equal to action space shape {self.action_space.shape}")
    
    def determine_final_termination_and_reward(self, target_flags, termination_reasons, main_terminate, far_flag_terminate):
        
        terminated = False
        
        if main_terminate:
            # Determine episode termination and reward based on the overall targets and reasons
            terminated, self.reward_dict['termination_reward'], failed_targets = self.reward_calculator.determine_final_termination(
                self, target_flags
            )
            
            if terminated:
                # Add the current length of targets_not_met to the history
                self.targets_not_met_history.append(len(failed_targets))
        
        if termination_reasons and far_flag_terminate:
            if self.verbose > 1:
                print("Termination triggered due to:", ", ".join(termination_reasons))
            terminated = True
            self.reward_dict['termination_reward'] = 0

            if self.verbose > 1:
                print("Metrics far away:", ", ".join(termination_reasons))
                

        
        return terminated
    
    def action_reward(self, action):
        """
        Calculate the reward for the action taken.
        """
        reward = self.determine_action_quality(action)
        
        # Assign the calculated reward to the reward dictionary
        self.reward_dict['action_reward'] = reward

    
    # Methods to be overridden
    def update_and_process_selection(self, action: np.ndarray) -> None:
        pass
    
    def validate_process_action_calculate_reward(self, action: np.ndarray) -> tuple:
        pass
    
    def initialize_action_space(self) -> None:
        pass
        
# Register all environments

register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env:SchoolMealSelectionContinuous',
    max_episode_steps=100
)

register(
    id='SchoolMealSelection-v1',
    entry_point='models.envs.env:SchoolMealSelectionDiscrete',
    max_episode_steps=100
)

register(
    id='SchoolMealSelection-v2',
    entry_point='models.envs.env:SchoolMealSelectionDiscreteDone',
    max_episode_steps=1000
)

class SchoolMealSelectionContinuous(BaseEnvironment):
    """
    A custom Gym environment for selecting school meals that meet certain nutritional and environmental criteria.
    The environment allows actions to adjust the quantity of ingredients and calculates rewards based on multiple metrics,
    such as nutrient values, environmental impact, cost, and consumption patterns.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients: int = 6, action_scaling_factor: int = 10, render_mode: str = None, 
                 verbose: int = 0, seed: int = None, reward_type: str = 'sparse', 
                 initialization_strategy: str = 'zero', max_episode_steps: int = 100):
        super().__init__(ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, reward_type, initialization_strategy, max_episode_steps)

    def initialize_action_space(self) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_ingredients,), dtype=np.float32)
        
    def validate_process_action_calculate_reward(self, action: np.ndarray) -> tuple:
        """
        Process the action to update the current selection and calculate the reward.
        """
        
        # Process the action to update the current selection
        self.update_and_process_selection(action)
        
        reward = 0
        terminated = False
        main_terminate = False
        far_flag_terminate = False
        
        if self.nsteps > 25:
            far_flag_terminate = True
            
        # Calculate the reward for the action
        termination_reasons, target_flags = self.calculate_reward(action)
        
        if self.nsteps == self.max_episode_steps-1:
            main_terminate = True
            
            
        # Calculate if terminated and termination reward
        terminated = self.determine_final_termination_and_reward(target_flags, termination_reasons, main_terminate, far_flag_terminate)
        
        reward = self.sum_reward_values()
                 
        return reward, terminated
        
    def update_and_process_selection(self, action: np.ndarray) -> None:
        """
        Update the current selection of ingredients based on the action.
        """
        # Clip the action values to be within the range [-1, 1]
        action = np.clip(action, -1, 1)
        
        # Validate action shape
        self.validate_action_shape(action, self.n_ingredients)
        
        # Action reward is specific to environment so it is calculated out of the reward calculator. Calculated before current selection updated
        self.action_reward(action)
        
        # Apply the action scaling factor to the action and update the current selection
        change = action * self.action_scaling_factor
        
        self.current_selection = np.maximum(0, self.current_selection + change)
        
        # Ensure current selection isnt greater than max_ingredients
        self.cut_current_selection()
        
        # Calculate metrics for the current selection
        self.get_metrics()
        
        # Initialize reward dictionary
        self.initialize_rewards()
        
        # Print current meal plan if verbosity
        self.print_current_meal_plan()
    
    def determine_action_quality(self, action):
        
        reward, number_of_selected_ingredients = self._calculate_number_of_action_distance(action)
        
        reward += self._calculate_if_actions_are_on_chosen_ingredients(action, number_of_selected_ingredients)
        
        reward = reward / 2

        return reward
    
    def _calculate_number_of_action_distance(self, action):
        # Use bitwise operators to combine conditions
        number_of_selected_ingredients = len(np.where((action > 0.01) | (action < -0.01))[0])
        distance = 1 - (abs(number_of_selected_ingredients - self.max_ingredients) / self.n_ingredients)
        
        return distance, number_of_selected_ingredients
    
    def _calculate_if_actions_are_on_chosen_ingredients(self, action, number_of_selected_ingredients):
        count = 0
        for idx, value in enumerate(action):
            if value > 0.01 or value < -0.01:
                if self.current_selection[idx] > 0:
                    count += 1
                    
        distance = count / number_of_selected_ingredients
        return distance

class SchoolMealSelectionDiscrete(BaseEnvironment):
    """
    A custom Gym environment for selecting school meals that meet certain nutritional and environmental criteria.
    The environment allows actions to adjust the quantity of ingredients and calculates rewards based on multiple metrics,
    such as nutrient values, environmental impact, cost, and consumption patterns.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients: int = 6, action_scaling_factor: int = 10, render_mode: str = None, 
                 verbose: int = 0, seed: int = None, reward_type: str = 'shaped', 
                 initialization_strategy: str = 'zero', max_episode_steps: int = 100):
        super().__init__(ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, reward_type, initialization_strategy, max_episode_steps)

    def initialize_action_space(self) -> None:
        self.action_space = gym.spaces.MultiDiscrete([3, self.n_ingredients])
        
    def validate_process_action_calculate_reward(self, action: np.ndarray) -> tuple:
        """
        Process the action to update the current selection and calculate the reward.
        """
        
        # Process the action to update the current selection
        self.update_and_process_selection(action)
        
        reward = 0
        terminated = False
        main_terminate = False
        far_flag_terminate = False
        
        if self.nsteps > 50:
            far_flag_terminate = True
            
        # Calculate the reward for the action
        termination_reasons, target_flags = self.calculate_reward(action)
        
        if self.nsteps == self.max_episode_steps-1:
            main_terminate = True
            
        # Calculate if terminated and termination reward
        terminated = self.determine_final_termination_and_reward(target_flags, termination_reasons, main_terminate, far_flag_terminate)
        
        reward = self.sum_reward_values()
                 
        return reward, terminated
        
    def update_and_process_selection(self, action: np.ndarray) -> None:
        """
        Update the current selection of ingredients based on the action.
        """
        # Validate action shape
        self.validate_action_shape(action, 2)
        
        # Action reward is specific to environment so it is calculated out of the reward calculator. Calculated before current selection updated
        self.action_reward(action)
        
        if self.verbose > 2:
            print(f"Initial current_selection: {self.current_selection}")
            print(f"Action received: {action}")
            
        if action[0] == 0:
            self.current_selection[action[1]] = 0 # zero value
        elif action[0] == 1: # Decrease
            self.current_selection[action[1]] = max(0, self.current_selection[action[1]] - self.action_scaling_factor)
        elif action[0] == 2: # Increase
            self.current_selection[action[1]] += self.action_scaling_factor
        
        # Ensure current selection isnt greater than max_ingredients
        self.cut_current_selection()
        
        # Calculate metrics for the current selection
        self.get_metrics()
        
        # Initialize reward dictionary
        self.initialize_rewards()
        
        # Print current meal plan if verbosity
        self.print_current_meal_plan()
        
    def determine_action_quality(self, action):
        reward = 0
        
        reward += self._calculate_if_actions_are_on_groups_which_are_needed(action)

        return reward
    
    def _calculate_if_actions_are_on_groups_which_are_needed(self, action):
        reward = 0
        # Iterate through each ingredient group to set the action mask
        for key, target in self.ingredient_group_count_targets.items():
            value = self.ingredient_group_count[key]  # Current count of the ingredient group
            indexes = self.group_info[key]['indexes']  # Indexes of ingredients in this group
            selected = [idx for idx in indexes if self.current_selection[idx] > 0]  # Selected ingredients in this group
            
            if target != value:
                # If the target is not met, the agent should select ingredients from this group so will be reward if it does
                for idx in selected:
                    if action[1] == idx:
                        reward += 1
        return reward

class SchoolMealSelectionDiscreteDone(BaseEnvironment):
    """
    A custom Gym environment for selecting school meals that meet certain nutritional and environmental criteria.
    The environment allows actions to adjust the quantity of ingredients and calculates rewards based on multiple metrics,
    such as nutrient values, environmental impact, cost, and consumption patterns. Uses discrete actions with done signal.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients: int = 6, action_scaling_factor: int = 10, render_mode: str = None, 
                 verbose: int = 0, seed: int = None, reward_type: str = 'shaped', 
                 initialization_strategy: str = 'zero', max_episode_steps: int = 1000):
        super().__init__(ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, reward_type, initialization_strategy, max_episode_steps)

    def initialize_action_space(self) -> None:
        # [zero, decrease, increase] [selected ingredient to perform action] [done signal]
        self.action_space = gym.spaces.MultiDiscrete([4, 2, self.n_ingredients])
        
    def validate_process_action_calculate_reward(self, action: np.ndarray) -> tuple:
        """
        Process the action to update the current selection and calculate the reward.
        """
        
        # Process the action to update the current selection
        self.update_and_process_selection(action)
        
        reward = 0
        terminated = False
        main_terminate = False
        far_flag_terminate = False
        target_flags = None
        termination_reasons = None
        
        if self.nsteps > 100:
            far_flag_terminate = True
        
        # Calculate the reward for the action
        termination_reasons, target_flags = self.calculate_reward(action)
            
        if action[1] == 1:
            main_terminate = True
            
        # Calculate if terminated and termination reward
        terminated = self.determine_final_termination_and_reward(target_flags, termination_reasons, main_terminate, far_flag_terminate)
            
        reward += self.sum_reward_values()      
                 
        return reward, terminated
            
    def update_and_process_selection(self, action: np.ndarray) -> None:
        """
        Update the current selection of ingredients based on the action.
        """
        # Validate action shape
        self.validate_action_shape(action, 3)
        
        # Action reward is specific to environment so it is calculated out of the reward calculator. Calculated before current selection updated
        self.action_reward(action)
        
        if self.verbose > 2:
            print(f"Initial current_selection: {self.current_selection}")
            print(f"Action received: {action}")
            
        if action[0] == 0:
            self.current_selection[action[2]] = 0 # zero value
        if action[0] == 1:
            pass # Do nothing
        elif action[0] == 2: # Decrease
            self.current_selection[action[2]] = max(0, self.current_selection[action[2]] - self.action_scaling_factor)
        elif action[0] == 3: # Increase
            self.current_selection[action[2]] += self.action_scaling_factor
            
        # Ensure current selection isnt greater than max_ingredients
        self.cut_current_selection()
        
        # Calculate metrics for the current selection
        self.get_metrics()
        
        # Initialize reward dictionary
        self.initialize_rewards()
        
        # Print current meal plan if verbosity
        self.print_current_meal_plan()
        
    
    def determine_action_quality(self, action):
        
        reward = self._calculate_reward_termination_action(action)
        
        reward += self._calculate_if_actions_are_on_groups_which_are_needed(action)
        
        reward = reward / 2

        return reward
    
    def _calculate_reward_termination_action(self, action):
        reward = 0
        # if agent terminates and is below 50 steps, it gets a negative reward. In masking this will never happen as mask
        if action[1] == 1:  # Termination action
            if self.nsteps > 50:
                reward += 0.5
            if action[0] == 1:  # Selecting 'do nothing'
                reward += 1
        else:  # Non-termination action
            if action[0] != 1:  # Not selecting 'do nothing'
                reward += 1

        return reward
    
    def _calculate_if_actions_are_on_groups_which_are_needed(self, action):
        reward = 0
        # Iterate through each ingredient group to set the action mask
        for key, target in self.ingredient_group_count_targets.items():
            value = self.ingredient_group_count[key]  # Current count of the ingredient group
            indexes = self.group_info[key]['indexes']  # Indexes of ingredients in this group
            selected = [idx for idx in indexes if self.current_selection[idx] > 0]  # Selected ingredients in this group
            
            # Will not happen unless non masked algo is being used
            if target != value:
                # If the target is not met, the agent should select ingredients from this group so will be reward if it does
                for idx in selected:
                    if action[2] == idx:
                        reward += 1
                        if value > target: # If the value is over the target reward for a decrease or zero action
                            if action[0] == 0 or action[0] == 2:
                                reward += 1
                        elif value < target: # If the value is under the target reward for an increase action
                            if action[0] == 3:
                                reward += 1
        return reward

if __name__ == '__main__':
    from utils.process_data import get_data
    from gymnasium.utils.env_checker import check_env
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.monitor import Monitor
    from utils.train_utils import setup_environment, get_unique_directory, monitor_memory_usage, plot_reward_distribution, set_seed
    reward_dir, reward_prefix = get_unique_directory("saved_models/reward", 'reward_test34', '')
    from models.envs.env import * 
    class Args:
        render_mode = None
        num_envs = 2
        plot_reward_history = False
        max_episode_steps = 100
        verbose = 3
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
        env_name = 'SchoolMealSelection-v1'
        initialization_strategy = 'zero'
        vecnorm_norm_obs_keys = ['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
        reward_type = 'shaped'
        algo = 'MASKED_PPO'
    args = Args()

    num_episodes = 500
    
    reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
    
    env = setup_environment(args, reward_save_path=reward_save_path, eval=False)
    
    check_env(env.envs[0].unwrapped)

    print("Environment is valid!")
    
    del env

    env_kwargs = {
        "ingredient_df": args.ingredient_df,
        "max_ingredients": args.max_ingredients,
        "action_scaling_factor": args.action_scaling_factor,
        "render_mode": args.render_mode,
        "seed": args.seed,
        'initialization_strategy': args.initialization_strategy,
        "verbose": args.verbose,
        "reward_type":  args.reward_type
        }

    # from models.wrappers.common import RewardTrackingWrapper
    
    # def make_env():
    #     env = gym.make(args.env_name, **env_kwargs)
    #     # Apply the RewardTrackingWrapper if needed
    #     if args.plot_reward_history:
    #         if reward_save_path is None:
    #             raise ValueError("reward_save_path must be specified when plot_reward_history is True")
    #         env = RewardTrackingWrapper(
    #             env,
    #             args.reward_save_interval,
    #             reward_save_path,
    #             )
    #     env = TimeLimit(env, max_episode_steps=100)
    #     env = Monitor(env)
    #     return env     

    # env = make_env()
    # np.set_printoptions(suppress=True)

    # if args.memory_monitor:
    #     import threading
    #     monitoring_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #     monitoring_thread.start()

    # for episode in range(num_episodes):
    #     if episode % 100 == 0:
    #         print(f"Episode {episode + 1}")

    #     terminated = False
    #     truncated = False
    #     n_steps = 0
    #     obs = env.reset()
    #     while not terminated and not truncated:
    #         actions = env.action_space.sample()
    #         n_steps += 1
    #         obs, reward, terminated, truncated, info = env.step(actions)

    # env.close()
    #     # print(f"Episode {episode + 1} completed in {n_steps} steps.")

    # if args.plot_reward_history:
    #     reward_prefix = reward_prefix.split(".")[0]
    #     dir, pref = get_unique_directory(reward_dir, f"{reward_prefix}_plot", '.png')
    #     plot_path = os.path.abspath(os.path.join(dir, pref))
    #     plot_reward_distribution(reward_save_path, plot_path)