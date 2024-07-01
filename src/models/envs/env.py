import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from models.reward import reward
from models.initialization.initialization import IngredientSelectionInitializer


class BaseEnvironment(gym.Env):
    """
    A base class for custom Gym environments.
    Contains common functionalities for initializing and managing the environment.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, ingredient_df, max_ingredients=6, action_scaling_factor=10, render_mode=None, verbose=0, seed=None, reward_calculator_type='sparse', initialization_strategy='zero', max_episode_steps=100):
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
        self.reward_calculator = self._get_reward_calculator(reward_calculator_type)
        self.ingredient_initializer = IngredientSelectionInitializer()
        self.action_space = spaces.MultiBinary([self.n_ingredients, 4])
        self._initialize_observation_space()

    def _initialize_parameters(self, ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, initialization_strategy, max_episode_steps):
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

    def _initialize_nutrient_values(self, ingredient_df):
        self.caloric_values = ingredient_df['Calories_kcal_per_100g'].values.astype(np.float32) / 100
        self.fat_g = ingredient_df['Fat_g'].values.astype(np.float32) / 100
        self.saturates_g = ingredient_df['Saturates_g'].values.astype(np.float32) / 100
        self.carbs_g = ingredient_df['Carbs_g'].values.astype(np.float32) / 100
        self.sugars_g = ingredient_df['Sugars_g'].values.astype(np.float32) / 100
        self.fibre_g = ingredient_df['Fibre_g'].values.astype(np.float32) / 100
        self.protein_g = ingredient_df['Protein_g'].values.astype(np.float32) / 100
        self.salt_g = ingredient_df['Salt_g'].values.astype(np.float32) / 100

    def _initialize_nutrient_targets(self):
        self.target_calories = 530
        self.target_fat_g = 20.6
        self.target_saturates_g = 6.5
        self.target_carbs_g = 70.6
        self.target_sugars_g = 15.5
        self.target_fibre_g = 4.2
        self.target_protein_g = 7.5
        self.target_salt_g = 1  # Temporary target

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

    def _initialize_group_categories(self, ingredient_df):
        self.group_a_veg = ingredient_df['Group A veg'].values.astype(np.float32)
        self.group_a_fruit = ingredient_df['Group A fruit'].values.astype(np.float32)
        self.group_bc = ingredient_df['Group BC'].values.astype(np.float32)
        self.group_d = ingredient_df['Group D'].values.astype(np.float32)
        self.group_e = ingredient_df['Group E'].values.astype(np.float32)
        self.bread = ingredient_df['Bread'].values.astype(np.float32)
        self.confectionary = ingredient_df['Confectionary'].values.astype(np.float32)
    

    def _initialize_group_targets(self):
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
            'confectionary': (0, 50)
        }

        self.group_info = {
            'fruit': {'indexes': np.nonzero(self.group_a_veg)[0], 'probability': 0.8},
            'veg': {'indexes': np.nonzero(self.group_a_fruit)[0], 'probability': 0.8},
            'protein': {'indexes': np.nonzero(self.group_bc)[0], 'probability': 0.8},
            'carbs': {'indexes': np.nonzero(self.group_d)[0], 'probability': 0.8},
            'dairy': {'indexes': np.nonzero(self.group_e)[0], 'probability': 0.8},
            'bread': {'indexes': np.nonzero(self.bread)[0], 'probability': 0.8},
            'confectionary': {'indexes': np.nonzero(self.confectionary)[0], 'probability': 0.01}
        }

    def _initialize_environmental_ratings(self, ingredient_df):
        self.rating_to_int = {'A': 1, 'B': 0.5, 'C': 0, 'D': -0.5, 'E': -1}
        self.animal_welfare_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Animal Welfare Rating'].values], dtype=np.float32)
        self.rainforest_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Rainforest Rating'].values], dtype=np.float32)
        self.water_scarcity_rating = np.array([self.rating_to_int[val] for val in ingredient_df['Water Scarcity Rating'].values], dtype=np.float32)
        self.co2_fu_rating = np.array([self.rating_to_int[val] for val in ingredient_df['CO2 FU Rating'].values], dtype=np.float32)
        self.co2_g = {'co2_g': 0.0}
        self.co2_g_per_1g = ingredient_df['CO2_g_per_100g'].values.astype(np.float32) / 100
        self.target_co2_g_per_meal = 900  # Temporary CO2 target

    def _initialize_consumption_data(self, ingredient_df):
        self.mean_g_per_day = ingredient_df['Mean_g_per_day'].values.astype(np.float32)
        self.standard_deviation = ingredient_df['StandardDeviation'].values.astype(np.float32)
        self.coefficient_of_variation = ingredient_df['Coefficient of Variation'].values.astype(np.float32)

    def _initialize_cost_data(self, ingredient_df):
        self.cost_per_1g = ingredient_df['Cost_100g'].values.astype(np.float32) / 100
        self.menu_cost = {'cost': 0.0}
        self.target_cost_per_meal = 2

    def _initialize_selection_variables(self):
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
        self.target_not_met_counters = Counter()
        self.total_quantity_ratio = self.current_selection / (sum(self.current_selection) + 1e-9)
        
    def _create_index_value_mapping(self):
        """
        Generate evenly spaced values between 0 and 1 for ingredient indices.
        Creates a mapping of original indices to new values.
        """
        new_values = np.linspace(0, 1, self.n_ingredients)
        self.index_value_mapping = {i: new_values[i] for i in range(self.n_ingredients)}

    def _get_reward_calculator(self, reward_calculator_type: str):
        reward_calculator_mapping = {
            'shaped': reward.ShapedRewardCalculator,
            'sparse': reward.SparseRewardCalculator,
            # Add other mappings as needed
        }
        return reward_calculator_mapping[reward_calculator_type]
    
    def _initialize_observation_space(self):
        """
        Define the observation space for the environment.
        The observation space is a dictionary containing various components such as:
        current selection values, indices, time feature, nutrient values, group counts, environment counts, cost, and CO2 emissions.
        """
        self.observation_space = spaces.Dict({
            'current_selection_value': spaces.Box(low=0, high=1000, shape=(self.max_ingredients,), dtype=np.float64),
            'current_selection_index': spaces.Box(low=0, high=1, shape=(self.max_ingredients,), dtype=np.float64),
            'time_feature': spaces.Box(low=0, high=self.max_episode_steps, shape=(1,), dtype=np.float64),
            'nutrients': spaces.Box(low=0, high=3000, shape=(len(self.nutrient_averages),), dtype=np.float64),
            'groups': spaces.Box(low=0, high=self.max_ingredients, shape=(len(self.ingredient_group_count),), dtype=np.float64),
            'environment_counts': spaces.Box(low=-1, high=1, shape=(len(self.ingredient_environment_count),), dtype=np.float64),
            'cost': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float64),
            'co2_g': spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float64),
            'consumption': spaces.Box(low=0, high=100, shape=(len(self.consumption_average),), dtype=np.float64)
        })

    def _calculate_reward(self):
        """
        Calculate the reward for the current state of the environment.
        Rewards are calculated based on multiple factors like nutrient values, ingredient group counts, environmental impact, cost, and consumption.
        The function also checks for termination conditions based on these factors.
        """
        reward = 0
        terminated = False
        
        # Calculate metrics for the current selection
        self._get_metrics()
        # Initialize reward dictionary
        self._initialize_rewards()
        
        # Calculate reward and check for termination conditions every 25 steps
        if self.nsteps % 25 == 0 and self.nsteps > 0:
            # Evaluate rewards for each metric and calculate if the episode is terminated or not
            terminated = self._evaluate_rewards()
            reward += self._sum_reward_values()
            
        return reward, terminated
    
    def _sum_reward_values(self):
        # Sum the reward dictionary values to get the total reward
        return sum(value for subdict in self.reward_dict.values() if isinstance(subdict, dict) for value in subdict.values() if isinstance(value, (int, float))) + \
            sum(value for value in self.reward_dict.values() if isinstance(value, (int, float)))
            
    def _get_metrics(self):
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

    def _calculate_nutrient_averages(self):
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

    def _calculate_non_zero_values(self, non_zero_mask):
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

    def _sum_dict_values(self, data, multiply_with=None):
        """
        Sum values in a dictionary, optionally multiplying each value with a corresponding array.
        If multiply_with is provided, it multiplies each value with the corresponding element in multiply_with before summing.
        Returns a dictionary of summed values.
        """
        if multiply_with is None:
            return {key: sum(values) for key, values in data.items()}
        else:
            return {key: sum(values * multiply_with) for key, values in data.items()}


    def _calculate_ingredient_environment_count(self):
        """
        Calculate the environmental impact counts for the current selection of ingredients.
        Returns a dictionary of rounded environmental ratings for each impact category.
        """
        def _round_to_nearest_rating(value):
            values = list(self.rating_to_int.values())
            closest_value = min(values, key=lambda x: abs(x - value))
            for val in self.rating_to_int.values():
                if val == closest_value:
                    return closest_value
                
        return {
            'animal_welfare': _round_to_nearest_rating(sum(self.animal_welfare_rating * self.total_quantity_ratio)),
            'rainforest': _round_to_nearest_rating(sum(self.rainforest_rating * self.total_quantity_ratio)),
            'water': _round_to_nearest_rating(sum(self.water_scarcity_rating * self.total_quantity_ratio)),
            'co2_rating': _round_to_nearest_rating(sum(self.co2_fu_rating * self.total_quantity_ratio)),
        }

    def _calculate_consumption_average(self):
        """
        Calculate the average consumption values based on the current selection of ingredients.
        Returns a dictionary of average mean consumption and coefficient of variation.
        """
        return {
            'average_mean_consumption': sum(self.mean_g_per_day * self.total_quantity_ratio),
            'average_cv_ingredients': sum(self.coefficient_of_variation * self.total_quantity_ratio)
        }

    def _calculate_cost(self):
        """
        Calculate the total cost based on the current selection of ingredients.
        Returns a dictionary with the total cost.
        """
        return {
            'cost': sum(self.cost_per_1g * self.current_selection),
        }
        
    def _calculate_co2g(self):
        """
        Calculate the total CO2 emissions based on the current selection of ingredients.
        Returns a dictionary with the total CO2 emissions.
        """
        return {
            'co2_g': sum(self.co2_g_per_1g * self.current_selection),
        }

    def _initialize_rewards(self):
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
        self.reward_dict['consumption_reward'] = {'average_mean_consumption': 0, 'cv_penalty': 0}
        self.reward_dict['targets_not_met'] = []
        self.reward_dict['termination_reward'] = 0
        self.reward_dict['step_penalty'] = 0

    def _evaluate_rewards(self):
        """
        Evaluate the rewards based on the current state of the environment.
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
            # Skip step penalty, targets not met, and termination reward rewards
            if reward in ['step_penalty', 'targets_not_met', 'termination_reward']:
                continue
            
            # Get the corresponding reward function from the RewardCalculator
            method = self.reward_to_function_mapping[reward]

            reward_func = getattr(self.reward_calculator, method)
            
            # Calculate the reward, targets met, and termination condition for the current reward
            self.reward_dict[reward], targets_met, terminate = reward_func(self)
            
            # Update the target met flags and termination reasons based on the reward
            self._update_flags_and_reasons(reward, targets_met, terminate, target_flags, termination_reasons)
            
        # Determine if the episode should be terminated based on the overall targets and reasons
        terminated, self.reward_dict['termination_reward'], failed_targets = self.reward_calculator.determine_termination(
            self, target_flags, termination_reasons
        )
        # Store the targets not met in the reward dictionary
        self.reward_dict['targets_not_met'] = failed_targets

        # Set step penalty to zero
        self.reward_dict['step_penalty'] = 0
        
        return terminated

    def _update_flags_and_reasons(self, reward, targets_met, terminate, target_flags, termination_reasons):
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


    def reset(self, seed=None, options=None):
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
        self._initialize_rewards()

        # If render mode is 'human', render the environment
        if self.render_mode == 'human':
            self.render()

        # Return the initial observation and information
        return self._get_obs(), self._get_info()

    def step(self, action):
        pass  # This method should be overridden in subclasses
    
    def _validate_and_process_action(self, action):
        # Validate action shape
        self._validate_action_shape(action)
        
        # Process the action to update the current selection
        self._update_selection(action)
        
    def _update_selection(self, action):
        # Map the binary action to -10, 0, or 10 using the action scaling factor
        action_values = action.reshape((self.n_ingredients, 3))
        mapped_actions = np.dot(action_values, [-1, 0, 1]) * self.action_scaling_factor
        
        # Update the current selection with the mapped actions
        self.current_selection = np.maximum(0, self.current_selection + mapped_actions)
        
        # Ensure that the number of selected ingredients does not exceed the maximum allowed
        num_selected_ingredients = np.sum(self.current_selection > 0)
        if num_selected_ingredients > self.max_ingredients:
            excess_indices = np.argsort(-self.current_selection)
            self.current_selection[excess_indices[self.max_ingredients:]] = 0
            
    def _validate_action_shape(self, action):
        if self.verbose > 1:
            if not isinstance(action, np.ndarray) or action.shape != (self.n_ingredients * 3,):
                raise ValueError(f"Expected action to be an np.ndarray of shape {(self.n_ingredients * 3,)}, but got {type(action)} with shape {action.shape}")
            if action.shape != self.action_space.shape:
                raise ValueError(f"Action shape {action.shape} is not equal to action space shape {self.action_space.shape}")

    def _get_obs(self):
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

    def _get_info(self):
        """
        Get additional information about the current state of the environment.
        Includes nutrient averages, ingredient group counts, environmental impact, cost, CO2 emissions, rewards, group portions, targets not met count, and the current meal plan.
        Returns the info dictionary.
        """
        # Update the counters for targets not met with the current reward dictionary values
        self.target_not_met_counters.update(self.reward_dict['targets_not_met'])
        
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
            'current_meal_plan': current_meal_plan
        }

        return info

    def render(self):
        """
        Render the current state of the environment.
        Displays the current step or episode based on the render mode.
        """
        if self.render_mode == 'step':
            print(f"Step: {self.nsteps}")
        if self.render_mode == "human":
            print(f"Episode: {self.episode_count}")

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass

    def get_current_meal_plan(self):
        """
        Get the current meal plan based on the selected ingredients.
        Ensures that the number of selected ingredients matches the max_ingredients.
        Returns the current meal plan, non-zero indices, and non-zero values.
        """
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

            # Select additional indices from the zero indices to meet max_ingredients
            additional_indices = np.random.choice(
                zero_indices,
                size=additional_indices_needed,
                replace=False
            )
            nonzero_indices = np.concatenate((nonzero_indices, additional_indices))
            nonzero_values = np.concatenate((nonzero_values, self.current_selection[additional_indices]))

        # Create the current meal plan dictionary with category values
        for idx, value in zip(nonzero_indices, nonzero_values):
            category_value = self.ingredient_df['Category7'].iloc[idx]
            current_meal_plan[category_value] = value

        return current_meal_plan, nonzero_indices, nonzero_values
        
    def _initialize_current_selection(self):
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
            print(f"\nInitialized plan: {current_meal_plan}")

        # Calculate initial metrics and update observations
        self._get_metrics()
        self._get_obs()
        
from gymnasium.envs.registration import register
# Register the environment
register(
    id='SchoolMealSelection-v1',
    entry_point='models.envs.env_working:SchoolMealSelection',
)


from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

class SchoolMealSelection(BaseEnvironment):
    def __init__(self, ingredient_df, max_ingredients=6, action_scaling_factor=10, render_mode=None, verbose=0, seed=None, reward_calculator_type='sparse', initialization_strategy='zero', max_episode_steps=100):
        super().__init__(ingredient_df, max_ingredients, action_scaling_factor, render_mode, verbose, seed, reward_calculator_type, initialization_strategy, max_episode_steps)

    def step(self, action):
        # Implement the step method specific to SchoolMealSelection
        terminated = False  # Initialize termination flag

        # Increment the number of steps taken
        self.nsteps += 1
        
        # Validate and process the action
        self._validate_and_process_action(action)
        
        # Calculate the reward and check for termination conditions
        reward, terminated = self._calculate_reward()

        # Return the new observation, reward, termination flag, truncated flag, and additional info
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def valid_action_mask(self, action):
        
        self.ingredient_group_count
        
        self.ingredient_group_count_targets
        
        group_to_data_map = {
            'fruit': self.group_a_fruit,
            'veg': self.group_a_veg,
            'protein': self.group_bc,
            'carbs': self.group_d,
            'dairy': self.group_e,
            'bread': self.bread,
            'confectionary': self.confectionary
        }
        
        self.group_info = {
            'fruit': {'indexes': np.nonzero(self.group_a_veg)[0], 'probability': 0.8},
            'veg': {'indexes': np.nonzero(self.group_a_fruit)[0], 'probability': 0.8},
            'protein': {'indexes': np.nonzero(self.group_bc)[0], 'probability': 0.8},
            'carbs': {'indexes': np.nonzero(self.group_d)[0], 'probability': 0.8},
            'dairy': {'indexes': np.nonzero(self.group_e)[0], 'probability': 0.8},
            'bread': {'indexes': np.nonzero(self.bread)[0], 'probability': 0.8},
            'confectionary': {'indexes': np.nonzero(self.confectionary)[0], 'probability': 0.01}
        }
        
        action_mask = np.ones(self.n_ingredients * 4, dtype=np.int8)
        

                
                
        
        current_meal_plan, nonzero_indices, nonzero_values = self.get_current_meal_plan()
        # Implement the valid_action_mask method specific to SchoolMealSelection
        
        action_mask = np.zeros(self.n_ingredients * 3, dtype=np.int8)
        
        return action_mask
    
    
        for key, value in self.ingredient_group_count.items():
            if value == self.ingredient_group_count_targets[key]: 
                for idx in nonzero_indices:
                    for category, info in group_info.values():
                        if idx in info['indexes']:
                            if category == key:
                                action_mask[idx] = [1, 1, 1, 1]
                                for i in len(action_mask):
                                    if i != idx:
                                        action_mask[i] = [0, 0, 0, 0]
                                action_mask[]
                                break
                            break