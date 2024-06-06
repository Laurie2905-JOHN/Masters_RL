

def reward_nutrient_macro(self):
    
    step_penalty = -1  # Negative reward for each step taken
    # Initialising reward as 0
    reward = 0

    terminated = False

    nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list = nutrient_reward(self)
    
    all_group_targets_met = True
                    
    terminated, reward = termination_reason(self, all_nutrient_targets_met, nutrient_far_flag_list, all_group_targets_met, reward)

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, nutrient_rewards, info, terminated


def reward_nutrient_macro_and_groups(self):
    
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

    return reward, nutrient_rewards, info, terminated

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
    
    # import pdb
    # pdb.set_trace()
    
        # Calculate the total values for each nutritional category for the selected ingredients
    self.ingredient_group_count= {
        'fruit': sum(self.Group_A_fruit * self.current_selection),
        'veg': sum(self.Group_A_veg * self.current_selection),
        'non_processed_meat': sum(self.Group_B * self.current_selection),
        'processed_meat': sum(self.Group_C * self.current_selection),
        'carbs': sum(self.Group_D * self.current_selection),
        'dairy': sum(self.Group_E * self.current_selection),
        'bread': sum(self.Bread * self.current_selection),
        'confectionary': sum(self.Confectionary * self.current_selection),
    }
    
    ingredient_group_count_rewards = {k: 0 for k in self.ingredient_group_count.keys()}
    
    all_group_targets_met = True
    
    # Loop through values and calculate rewards
    for group, value in self.ingredient_group_count.items():
        
        target = self.ingredient_group_count_targets[group]
        
        if target == value <= target:
            ingredient_group_count_rewards[group] += 10
        else:
            all_group_targets_met = False
             
            distance = abs(target - value)
            
            distance_reward = distance / 10
            
            all_group_targets_met = False
            
            if distance_reward > 500:
                ingredient_group_count_rewards[group] -= 500
            else:
                ingredient_group_count_rewards[group] -= distance / 10
    
    return ingredient_group_count_rewards, all_group_targets_met