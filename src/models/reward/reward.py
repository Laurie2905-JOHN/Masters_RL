

def reward_nutrient(self):
    
    step_penalty = -1  # Negative reward for each step taken
    # Initialising reward as 0
    reward = 0

    terminated = False

    nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list = nutrient_reward(self)
    
    all_group_targets_met = True
                    
    terminated, reward = termination_reason(self, all_nutrient_targets_met, nutrient_far_flag_list, all_group_targets_met, reward)
    
    ingredient_group_count_rewards = 0

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, info, terminated


def reward_nutrient_food_groups(self):
    
    step_penalty = -1  # Negative reward for each step taken
    
    # Initialising reward as 0
    reward = 0

    terminated = False

    nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list = nutrient_reward(self)
    
    ingredient_group_count_rewards, all_group_targets_met = group_count_reward(self)
                    
    terminated, reward = termination_reason(self, all_nutrient_targets_met, nutrient_far_flag_list, all_group_targets_met, reward)

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + sum(ingredient_group_count_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, info, terminated


def reward_nutrient_food_groups_environment(self):
    
    step_penalty = -1  # Negative reward for each step taken
    
    # Initialising reward as 0
    reward = 0

    terminated = False

    nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list = nutrient_reward(self)
    
    ingredient_group_count_rewards, all_group_targets_met = group_count_reward(self)
    
    ingredient_environment_count_rewards = environment_count_reward(self)
                    
    terminated, reward = termination_reason(self, all_nutrient_targets_met, nutrient_far_flag_list, all_group_targets_met, reward)

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + sum(ingredient_group_count_rewards.values()) + sum(ingredient_environment_count_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, info, terminated

# Function to calculate the reward for the nutrients
def nutrient_reward(self):
    
    # Calculate the total values for each nutritional category for the selected ingredients
    self.nutrient_averages = {
        'calories': sum(self.caloric_values * self.current_selection),
        'fat': sum(self.Fat_g * self.current_selection),
        'saturates': sum(self.Saturates_g * self.current_selection),
        'carbs': sum(self.Carbs_g * self.current_selection),
        'sugar': sum(self.Sugars_g * self.current_selection),
        'fibre': sum(self.Fibre_g * self.current_selection),
        'protein': sum(self.Protein_g * self.current_selection),
        'salt': sum(self.Salt_g * self.current_selection)
    }
    
    # Initialising nutrient rewards
    nutrient_rewards = {k: 0 for k in self.nutrient_averages.keys()}
    
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


def termination_reason(self, all_nutrient_targets_met, nutrient_far_flag_list, all_group_targets_met, reward):
    # Implement termination conditions
    if all_nutrient_targets_met and all_group_targets_met:
        # If all targets are met terminate the episode
        terminated = True
        self.termination_reason = 2  # All targets met
        reward += 10000
    elif nutrient_far_flag_list.count(True) > 4:
        # If 4 of the metrics are far away terminate the episode as no learning oppurtunity
        terminated = True
        self.termination_reason = -1  # A target is far off
    else:
        terminated = False
        self.termination_reason = 0  # No termination
    return terminated, reward

def group_count_reward(self):
    non_zero_mask = self.current_selection != 0
    
    # Calculate the total values for each nutritional category for the selected ingredients
    self.ingredient_group_count = {
        'fruit': sum(self.Group_A_fruit * non_zero_mask),
        'veg': sum(self.Group_A_veg * non_zero_mask),
        'non_processed_meat': sum(self.Group_B * non_zero_mask),
        'processed_meat': sum(self.Group_C * non_zero_mask),
        'carbs': sum(self.Group_D * non_zero_mask),
        'dairy': sum(self.Group_E * non_zero_mask),
        'bread': sum(self.Bread * non_zero_mask),
        'confectionary': sum(self.Confectionary * non_zero_mask),
    }
    
    ingredient_group_count_rewards = {k: 0 for k in self.ingredient_group_count.keys()}
    all_group_targets_met = True
    
    # Special handling for meat groups
    total_meat_count = self.ingredient_group_count['non_processed_meat'] + self.ingredient_group_count['processed_meat']
    total_meat_target = self.ingredient_group_count_targets['non_processed_meat'] + self.ingredient_group_count_targets['processed_meat']
    
    if total_meat_count == total_meat_target:
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
        
        if target == value:
            ingredient_group_count_rewards[group] += 100
        else:
            all_group_targets_met = False
            distance = abs(target - value)
            distance_reward = distance * 10
            ingredient_group_count_rewards[group] += distance_reward
    
    return ingredient_group_count_rewards, all_group_targets_met

def environment_count_reward(self):
    non_zero_mask = self.current_selection != 0
    
    # Calculate environmental counts
    self.ingredient_environment_count = {
        'animal_welfare': round(sum(self.Animal_Welfare_Rating * non_zero_mask) / self.max_ingredients),
        'rainforest': round(sum(self.Rainforest_Rating * non_zero_mask) / self.max_ingredients),
        'water': round(sum(self.Water_Scarcity_Rating * non_zero_mask) / self.max_ingredients),
        'CO2_rating': round(sum(self.CO2_FU_Rating * non_zero_mask) / self.max_ingredients),
        'CO2_g': sum(self.CO2_kg_per_100g * self.current_selection),
    }
    
    # Initialize rewards
    ingredient_environment_count_rewards = {k: 0 for k in self.ingredient_environment_count.keys()}
    
    # Calculate shaped reward for CO2_g
    CO2_g_value = self.ingredient_environment_count['CO2_g']
    if CO2_g_value <= self.target_CO2_kg_g:
        ingredient_environment_count_rewards['CO2_g'] = (self.target_CO2_kg_g - CO2_g_value) * 10  # Positive reward for lower CO2 emissions
    else:
        ingredient_environment_count_rewards['CO2_g'] = -(CO2_g_value - self.target_CO2_kg_g) * 10  # Negative reward for higher CO2 emissions
    
    # Define the mapping for ratings (1 gets highest reward, 5 gets lowest)
    rating_to_reward = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    
    # Assign rewards based on ratings for other environmental factors
    for factor in ['animal_welfare', 'rainforest', 'water', 'CO2_rating']:
        rating_value = self.ingredient_environment_count[factor]
        if rating_value in rating_to_reward:
            ingredient_environment_count_rewards[factor] = rating_to_reward[rating_value]
    
    return ingredient_environment_count_rewards


