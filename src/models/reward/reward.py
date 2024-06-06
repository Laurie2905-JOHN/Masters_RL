import numpy as np

def reward_nutrient_macro(self):
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
    
    nutrient_rewards = {k: 0 for k in self.nutrient_averages.keys()}
    step_penalty = -1  # Negative reward for each step taken
    reward = 0

    terminated = False
    any_target_far_off = False
    all_targets_met = True
    far_flag_list = []
    for nutrient, average_value in self.nutrient_averages.items():
        
        target_min, target_max = self.target_ranges[nutrient]
        
        # If targets are a quarter or quadruple the average value, consider it far off
        far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
        far_flag_list.append(far_flag)
        
        # Check if the nutrient value is far from target and adjust rewards and termination accordingly
        if far_flag:
            nutrient_rewards[nutrient] -= 100
            any_target_far_off = True
            all_targets_met = False
        else:
            if target_min <= average_value <= target_max:
                nutrient_rewards[nutrient] += 10
            else:
                distance = max(abs(target_min - average_value), abs(target_max - average_value))
                distance_reward = distance / 10
                
                all_targets_met = False
                
                if distance_reward > 500:
                    nutrient_rewards[nutrient] -= 500
                else:
                    nutrient_rewards[nutrient] -= distance / 10  # Adjust this value as necessary
    
    
    
    # Implement termination conditions
    if all_targets_met:
        # If all targets are met terminate the episode
        terminated = True
        self.termination_reason = 2  # All targets met
        reward += 10000
    elif far_flag_list.count(True) > 4:
        # If 4 of the metrics are far away terminate the episode as no learning oppurtunity
        terminated = True
        self.termination_reason = -1  # A target is far off
    else:
        self.termination_reason = 0  # No termination

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, nutrient_rewards, info, terminated




    
def reward_nutrient_macro_and_regulation(self, action):
    # Calculate the total values for each nutritional category for the selected ingredients
    self.average_calories_per_day = sum(self.caloric_values * self.current_selection)
    self.average_fat_per_day = sum(self.Fat_g * self.current_selection)
    self.average_saturates_per_day = sum(self.Saturates_g * self.current_selection)
    self.average_carbs_per_day = sum(self.Carbs_g * self.current_selection)
    self.average_sugar_per_day = sum(self.Sugars_g * self.current_selection)
    self.average_fibre_per_day = sum(self.Fibre_g * self.current_selection)
    self.average_protein_per_day = sum(self.Protein_g * self.current_selection)
    self.average_salt_per_day = sum(self.Salt_g * self.current_selection)


    reward = 0
    nutrient_rewards = {
        'calories': 0,
        'fat': 0,
        'saturates': 0,
        'carbs': 0,
        'sugar': 0,
        'fibre': 0,
        'protein': 0,
        'salt': 0
    }
    step_penalty = -1  # Negative reward for each step taken

    # Define target ranges
    target_ranges = {
        'calories': (self.target_calories * 0.95, self.target_calories * 1.05), # 5% tolerance Min Max
        'fat': (0, self.target_Fat_g), # 0 to max target
        'saturates': (0, self.target_Saturates_g), # 0 to max target
        'carbs': (self.target_Carbs_g, float('inf')), # min target to infinity
        'sugar': (0, self.target_Sugars_g), # 0 to max target
        'fibre': (self.target_Fibre_g, float('inf')), # min target to infinity
        'protein': (self.target_Protein_g, float('inf')), # min target to infinity
        'salt': (0, self.target_Salt_g) # 0 to max target
    }

    # Calculate distances and rewards for each nutrient
    nutrient_values = {
        'calories': self.average_calories_per_day,
        'fat': self.average_fat_per_day,
        'saturates': self.average_saturates_per_day,
        'carbs': self.average_carbs_per_day,
        'sugar': self.average_sugar_per_day,
        'fibre': self.average_fibre_per_day,
        'protein': self.average_protein_per_day,
        'salt': self.average_salt_per_day
    }

    terminated = False
    termination_reason = {}

    any_target_far_off_list = []
    all_targets_met_list = []

    for nutrient, average_value in nutrient_values.items():
        target_min, target_max = target_ranges[nutrient]

        # Calculate distance only if the target is not infinite or zero
        if target_min != 0 and target_max != float('inf'):
            distance = max(abs(average_value - target_min), abs(average_value - target_max))
            far_flag = average_value < 0.5 * target_min or average_value > 5 * target_max
        elif target_min == 0:
            distance = abs(average_value - target_max)
            far_flag = average_value > 5 * target_max
        elif target_max == float('inf'):
            distance = abs(average_value - target_min)
            far_flag = average_value < 0.5 * target_min
        else:
            distance = 0
            far_flag = False

        any_target_far_off_list.append(far_flag)
        all_targets_met_list.append(target_min <= average_value <= target_max)

        if far_flag:
            nutrient_rewards[nutrient] -= 10
            termination_reason[nutrient] = 'too_far_off'
            terminated = True

        if nutrient == 'calories':
            if target_min <= average_value <= target_max:
                nutrient_rewards['calories'] += 10
            else:
                nutrient_rewards['calories'] -= distance / 500
        else:
            if nutrient in ['carbs', 'fibre', 'protein']:
                if average_value >= target_min:
                    nutrient_rewards[nutrient] += 10
                else:
                    nutrient_rewards[nutrient] -= distance / 500
            else:
                if average_value <= target_max:
                    nutrient_rewards[nutrient] += 10
                else:
                    nutrient_rewards[nutrient] -= distance / 500

    # Implement termination conditions
    all_targets_met = all(all_targets_met_list)
    any_target_far_off = any(any_target_far_off_list)

    if all_targets_met:
        terminated = True
        termination_reason['reason'] = 'all_targets_met'
    elif any_target_far_off:
        terminated = True
        termination_reason['reason'] = 'any_target_far_off'

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, nutrient_rewards, info, terminated, termination_reason