import numpy as np

def calculate_simple_reward(self, action):

    # Set small values to zero to disregard them in the selection
    threshold = 0.1
    raw_action = np.where(action > threshold, action, 0)

    selected_flag = action > threshold
    # Evaluate the number of selected ingredients based on the adjusted actions
    total_selection = np.sum(selected_flag)

    # Separate the scaling factor from the rest of the action
    scaling_factor = (550 * self.num_people) / total_selection  # Scaling to try enforce approximately 550 grams per person for a meal / num of ingredients selected
    
    scaled_action = raw_action * scaling_factor
    
    # print(scaled_action)
    
    # Calculate calories only from the selected ingredients
    calories_selected_ingredients = self.caloric_values * scaled_action
    
    # Calculate average calories per day per person
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people
    

    # Initialize reward
    reward = 0

    # Define target ranges
    target_selection_min = 5
    target_selection_max = 10
    target_calories_min = self.target_calories - 50
    target_calories_max = self.target_calories + 50

    # Check if the action meets both conditions for a massive reward
    if target_selection_min <= total_selection <= target_selection_max and target_calories_min <= average_calories_per_day <= target_calories_max:
        reward += 10  # Massive reward for meeting both conditions
    else:
        # Calculate the distance to the target ranges and provide smaller incremental rewards or penalties
        if target_selection_min <= total_selection <= target_selection_max:
            reward += 5
        else:
            selection_distance = min(abs(total_selection - target_selection_min), abs(total_selection - target_selection_max))
            reward -= 0.1 * selection_distance  # Penalize based on how far it is from the target range

        if target_calories_min <= average_calories_per_day <= target_calories_max:
            reward += 5
        else:
            calories_distance = min(abs(average_calories_per_day - target_calories_min), abs(average_calories_per_day - target_calories_max))
            reward -= 0.1 * calories_distance  # Penalize based on how far it is from the target range

    # Since it's a single step, always terminate after giving the reward
    terminated = True

    info = self._get_info(total_selection, average_calories_per_day, scaled_action, calories_selected_ingredients)

    return reward, info, terminated



def calculate_simple_reward3(self, action):
    total_selection = np.sum(self.current_selection > 0)
    action_selection = np.sum(action > 0)
    calories_selected_ingredients = self.caloric_values * self.current_selection  
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

    reward = 0
    selection_reward = 0
    calorie_reward = 0

    target_calories_min = self.target_calories - 50
    target_calories_max = self.target_calories + 50

    # Calorie reward calculation
    if target_calories_min <= average_calories_per_day <= target_calories_max:
        calorie_reward += 1000  # Moderate reward for meeting calorie criteria
    else:
        calories_distance = min(abs(average_calories_per_day - target_calories_min), abs(average_calories_per_day - target_calories_max))
        calorie_reward -= (calories_distance ** 1) / 500

    # Selection reward calculation
    if total_selection < 10:
        selection_reward += (10 - total_selection) * 100  # Increased weight for selection reward
    elif total_selection > 10:
        selection_reward -= (total_selection - 10) * 100

    # Large bonus for meeting both criteria
    if target_calories_min <= average_calories_per_day <= target_calories_max and total_selection == 10:
        reward += 1e6  # Large reward for meeting both criteria
        terminated = False
    else:
        terminated = False

    reward += calorie_reward + selection_reward

    info = self._get_info(total_selection, average_calories_per_day, calories_selected_ingredients)
    return reward, selection_reward, calorie_reward, info, terminated



def reward_no_selection(self, action):
    calories_selected_ingredients = self.caloric_values * self.current_selection
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

    reward = 0
    selection_reward = 0
    calorie_reward = 0
    step_penalty = -1  # Negative reward for each step taken

    target_calories_min = self.target_calories - 50
    target_calories_max = self.target_calories + 50
    
    calories_distance = min(abs(average_calories_per_day - target_calories_min), abs(average_calories_per_day - target_calories_max))
    
    # Additional penalty for being too far from the target calories
    if calories_distance > self.max_possible_calories:
        calorie_reward -= 10
        terminated = True
    else:
        # Calorie reward calculation
        if target_calories_min <= average_calories_per_day <= target_calories_max:
            calorie_reward += 10  # Moderate reward for meeting calorie criteria
            terminated = True
        else:
            calorie_reward -= (calories_distance ** 1) / 500
            terminated = False
    
    # Include the step penalty in the reward calculation
    reward += calorie_reward + selection_reward + step_penalty

    info = self._get_info(average_calories_per_day, calories_selected_ingredients)
    
    return reward, selection_reward, calorie_reward, info, terminated


def reward_nutrient_macro(self):
    # Calculate the total values for each nutritional category for the selected ingredients
    nutrient_totals = {
        'calories': sum(self.caloric_values * self.current_selection),
        'fat': sum(self.Fat_g * self.current_selection),
        'saturates': sum(self.Saturates_g * self.current_selection),
        'carbs': sum(self.Carbs_g * self.current_selection),
        'sugar': sum(self.Sugars_g * self.current_selection),
        'fibre': sum(self.Fibre_g * self.current_selection),
        'protein': sum(self.Protein_g * self.current_selection),
        'salt': sum(self.Salt_g * self.current_selection)
    }

    # Calculate the average values per day for each nutritional category
    nutrient_averages = {k: v / self.num_people for k, v in nutrient_totals.items()}

    # Define target ranges and initialize rewards
    target_ranges = {
        'calories': (self.target_calories * 0.95, self.target_calories * 1.05),
        'fat': (0, self.target_Fat_g),
        'saturates': (0, self.target_Saturates_g),
        'carbs': (self.target_Carbs_g, float('inf')),
        'sugar': (0, self.target_Sugars_g),
        'fibre': (self.target_Fibre_g, float('inf')),
        'protein': (self.target_Protein_g, float('inf')),
        'salt': (0, self.target_Salt_g)
    }
    nutrient_rewards = {k: 0 for k in nutrient_totals.keys()}
    step_penalty = -1  # Negative reward for each step taken
    reward = 0

    terminated = False
    any_target_far_off = False
    all_targets_met = True

    for nutrient, average_value in nutrient_averages.items():
        target_min, target_max = target_ranges[nutrient]
        far_flag = False

        # Calculate distance and determine if target is far off
        if target_max == float('inf'):
            distance = abs(average_value - target_min)
            far_flag = average_value < 0.5 * target_min
        elif target_min == 0:
            distance = abs(average_value - target_max)
            far_flag = average_value > 2 * target_max
        else:
            distance = max(abs(average_value - target_min), abs(average_value - target_max)) * 2
            far_flag = average_value < 0.5 * target_min or average_value > 2 * target_max

        if far_flag:
            nutrient_rewards[nutrient] -= 100
            any_target_far_off = True
        else:
            if target_min <= average_value <= target_max:
                nutrient_rewards[nutrient] += 10
            else:
                nutrient_rewards[nutrient] -= distance / 500
                all_targets_met = False

        if nutrient == 'calories' and not (target_min <= average_value <= target_max):
            nutrient_rewards['calories'] -= distance / 500

    # Implement termination conditions
    if all_targets_met:
        terminated = True
        self.termination_reason = 2  # All targets met
        reward += 1000
    elif any_target_far_off:
        terminated = True
        self.termination_reason = -1  # Any target is far off
    else:
        self.termination_reason = 0  # No termination

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, nutrient_rewards, info, terminated


    
def reward_nutrient_macro_and_regulation(self, action):
    # Calculate the total values for each nutritional category for the selected ingredients
    total_calories = sum(self.caloric_values * self.current_selection)
    total_fat = sum(self.Fat_g * self.current_selection)
    total_saturates = sum(self.Saturates_g * self.current_selection)
    total_carbs = sum(self.Carbs_g * self.current_selection)
    total_sugar = sum(self.Sugars_g * self.current_selection)
    total_fibre = sum(self.Fibre_g * self.current_selection)
    total_protein = sum(self.Protein_g * self.current_selection)
    total_salt = sum(self.Salt_g * self.current_selection)

    # Calculate the average values per day for each nutritional category
    self.average_calories_per_day = total_calories / self.num_people
    self.average_fat_per_day = total_fat / self.num_people
    self.average_saturates_per_day = total_saturates / self.num_people
    self.average_carbs_per_day = total_carbs / self.num_people
    self.average_sugar_per_day = total_sugar / self.num_people
    self.average_fibre_per_day = total_fibre / self.num_people
    self.average_protein_per_day = total_protein / self.num_people
    self.average_salt_per_day = total_salt / self.num_people

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