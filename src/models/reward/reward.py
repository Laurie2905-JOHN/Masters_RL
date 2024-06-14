import numpy as np

# Function to calculate the reward for the nutrients
def nutrient_reward(self, nutrient_rewards):
    
    # Flag for if all nutrient targets are met
    all_nutrient_targets_met = True
    
    # List to keep track of values far from target
    nutrient_far_flag_list = []
    
    # Loop through values and calculate rewards
    for nutrient, average_value in self.nutrient_averages.items():
        # import pdb
        # pdb.set_trace()
        target_min, target_max = self.nutrient_target_ranges[nutrient]
        
        # If targets are a quarter or quadruple the average value, consider it far off
        far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
        nutrient_far_flag_list.append(far_flag)
        
        # Check if the nutrient value is far from target and adjust rewards and termination accordingly
        if far_flag:
            nutrient_rewards[nutrient] -= 100
            all_nutrient_targets_met = False
        else:
            if target_min <= average_value <= target_max:
                nutrient_rewards[nutrient] += 10
            else:
                distance = max(abs(target_min - average_value), abs(target_max - average_value))
                distance_reward = distance / 10
                
                all_nutrient_targets_met = False
                
                if distance_reward > 500:
                    nutrient_rewards[nutrient] -= 500
                else:
                    nutrient_rewards[nutrient] -= distance / 10
    
    return nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list


def group_count_reward(self, ingredient_group_count_rewards):
    
    def _is_within_portion_range(self, group):
        portion = self.ingredient_group_portion[group]
        min_target, max_target = self.ingredient_group_portion_targets[group]
        return min_target <= portion <= max_target
    
    all_group_targets_met = True

    # Special handling for meat groups
    total_meat_count = self.ingredient_group_count['non_processed_meat'] + self.ingredient_group_count['processed_meat']
    total_meat_target = self.ingredient_group_count_targets['non_processed_meat'] + self.ingredient_group_count_targets['processed_meat']

    if total_meat_count == total_meat_target and _is_within_portion_range(self, 'non_processed_meat') and _is_within_portion_range(self, 'processed_meat'):
        ingredient_group_count_rewards['non_processed_meat'] += 100
        ingredient_group_count_rewards['processed_meat'] += 100
    else:
        all_group_targets_met = False
        meat_distance = abs(total_meat_target - total_meat_count)
        meat_distance_reward = meat_distance * 10
        ingredient_group_count_rewards['non_processed_meat'] += meat_distance_reward
        ingredient_group_count_rewards['processed_meat'] += meat_distance_reward

    # Loop through other groups to calculate rewards
    for group, value in self.ingredient_group_count.items():
        if group in ['non_processed_meat', 'processed_meat']:
            continue

        target = self.ingredient_group_count_targets[group]

        if target == value and _is_within_portion_range(self, group):
            ingredient_group_count_rewards[group] += 100
        else:
            all_group_targets_met = False
            distance = abs(target - value)
            distance_reward = distance * 10
            ingredient_group_count_rewards[group] += distance_reward

    return ingredient_group_count_rewards, all_group_targets_met


def environment_count_reward(self, ingredient_environment_count_rewards):

    # Calculate shaped reward for CO2_g
    CO2_distance = abs(self.target_CO2_g_per_meal - self.ingredient_environment_count['CO2_g'])
    
    environment_target_met = False
    
    if self.ingredient_environment_count['CO2_g'] <= self.target_CO2_g_per_meal:
        ingredient_environment_count_rewards['CO2_g'] = CO2_distance / 10  # Positive reward for lower CO2 emissions
        environment_target_met = True
    else:
        if CO2_distance > 500:
            ingredient_environment_count_rewards['CO2_g'] = -100
        else:
            ingredient_environment_count_rewards['CO2_g'] = -CO2_distance / 50  # Negative reward for higher CO2 emissions
    
    # Define the mapping for ratings (1 gets highest reward, 5 gets lowest)
    rating_to_reward = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    
    # Assign rewards based on ratings for other environmental factors
    for factor in ['animal_welfare', 'rainforest', 'water', 'CO2_rating']:
        rating_value = self.ingredient_environment_count[factor]
        if rating_value in rating_to_reward:
            ingredient_environment_count_rewards[factor] = rating_to_reward[rating_value]
    
    return ingredient_environment_count_rewards, environment_target_met

def cost_reward(self, cost_rewards):
    # Calculate the cost difference
    cost_difference = self.menu_cost - self.target_cost_per_meal
    
    # Determine if cost targets are met
    cost_targets_met = cost_difference <= 0
    
    # Calculate the reward
    if cost_targets_met:
        # Positive reward for staying under or meeting the target cost
        cost_rewards['from_target'] = -cost_difference * 10  # Reward is positive because cost_difference is negative or zero
    else:
        # Negative reward for exceeding the target cost
        cost_rewards['from_target'] = -cost_difference * 10  # Reward is negative because cost_difference is positive

    return cost_rewards, cost_targets_met


def consumption_reward(self, consumption_reward):
    
    consumption_targets_met = True

    # Shape the reward based on the average consumption
    consumption_reward['average_mean_consumption'] = self.consumption_average['average_mean_consumption'] * 3.75  # Adjust the multiplier as needed

    # Shape the reward based on coefficient of variation
    # Assuming a lower CV is better, we use an inverse relation for the reward
    consumption_reward['cv_penalty'] = self.consumption_average['average_cv_ingredients'] * 6.5  # Adjust the multiplier as needed

    return consumption_reward, consumption_targets_met


def termination_reason(
        self,
        nutrition_targets_met,
        group_targets_met,
        environment_targets_met,
        cost_targets_met,
        consumption_targets_met,
        nutrient_far_flag_list,
        reward
    ):

    # Implement termination conditions
    if all([nutrition_targets_met, group_targets_met, environment_targets_met, cost_targets_met, consumption_targets_met]): 
        # If all targets are met terminate the episode
        terminated = True
        self.termination_reason = 2  # All targets met
        reward += 10000
    elif nutrient_far_flag_list.count(True) > 4:
        # If 4 of the metrics are far away or the food wastage is high terminate the episode as no learning opportunity
        terminated = True
        self.termination_reason = -1  # A target is far off
        reward -= 1000
    else:
        terminated = False
        self.termination_reason = 0  # No termination
    
    return terminated, reward


# def estimated_food_waste_percentage(self):
    
#     threshold = 0
#     total_food_waste = 0  # Initialize total food waste

#     # Mask for non-zero values in the current selection
#     non_zero_indices = np.where(self.current_selection > 0)[0]

#     # Generate random values from a normal distribution for each person and each non-zero ingredient
#     consumption_values = np.random.normal(
#         loc=self.Mean_g_per_day[non_zero_indices][:, np.newaxis],
#         scale=self.StandardDeviation[non_zero_indices][:, np.newaxis],
#         size=(len(non_zero_indices), self.num_people)
#     )

#     # Replace negative values with zero for realistic consumption values
#     consumption_values = np.where(consumption_values < threshold, 0, consumption_values)

#     # Calculate the total expected consumption for each ingredient
#     total_expected_consumption = np.sum(consumption_values, axis=1)

#     # Calculate the expected food waste for each ingredient
#     ingredient_food_waste = self.current_selection[non_zero_indices] * self.num_people - total_expected_consumption

#     # Sum up the food waste for all ingredients
#     total_food_waste = np.sum(ingredient_food_waste)
    
#     # Sum up the total selection for ingredients
#     total_selection = np.sum(self.current_selection)
    
#     # Calculate the percentage of food waste
#     food_waste_percentage = total_food_waste / total_selection

#     return food_waste_percentage