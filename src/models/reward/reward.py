import numpy as np
    
def calculate_complex_reward(self, action):
    reward = 0

    # Set small values to zero to disregard them in the selection
    threshold = 0.1
    raw_action = np.where(action > threshold, action, 0)

    selected_flag = action > threshold
    # Evaluate the number of selected ingredients based on the adjusted actions
    total_selection = np.sum(selected_flag)

    # Separate the scaling factor from the rest of the action
    scaling_factor = (550 * self.num_people) / total_selection  # Reduced scaling factor for more reasonable values
    scaled_action = raw_action * scaling_factor
    
    # Calculate calories only from the selected ingredients
    selected_ingredients = scaled_action[selected_flag]
    selected_calories = self.caloric_values[selected_flag]
    calories_selected_ingredients = selected_ingredients * selected_calories / 100  # Divide by 100 as caloric values are per 100g

    # Calculate average calories per day per person
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

    # Check if both conditions are met
    if 5 <= total_selection <= 10 and 2000 <= average_calories_per_day <= 3000:
        reward += 500  # Massive reward for meeting both conditions
        terminated = True
    else:
        terminated = False

        # Apply shaped negative rewards for selecting more than 10 ingredients
        if total_selection > 10:
            reward -= 20 * (total_selection - 10)  # Increasing penalty for selecting more than 10 ingredients
        elif total_selection < 5:
            reward -= 10  # Penalty for selecting less than 5 ingredients
        else:
            reward += 20  # Reward for selecting between 5 and 10 ingredients

        # Reward based on the average calories per day
        if 2000 <= average_calories_per_day <= 3000:
            reward += 100  # Additional reward for meeting the calorie constraint
        else:
            reward += max(0, 50 - abs(average_calories_per_day - 2500) * 0.1)  # Small reward for getting close to the target
            reward -= 10 * abs((average_calories_per_day - 2500) / 2500)  # Shaped penalty for not meeting the calorie constraint

        # Penalty for extreme ingredient quantities
        reward -= np.sum(np.maximum(0, scaled_action - 500)) * 0.1

        # Reward for zero quantities explicitly
        reward += np.sum(scaled_action == 0) * 0.2

    info = self._get_info(total_selection, average_calories_per_day)

    return reward, info, terminated

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


def calculate_simple_reward2(self, action):
    # Calculate the current state values based on actions taken
    total_selection = np.sum(self.current_selection > 0)

    # Calculate calories only from the selected ingredients
    calories_selected_ingredients = self.caloric_values * self.current_selection
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

    # Initialize reward incrementally
    reward = 0

    # Define target ranges
    target_selection_min = 5
    target_selection_max = 10
    target_calories_min = self.target_calories - 50
    target_calories_max = self.target_calories + 50

    # Calculate the distance to the target ranges and provide incremental rewards or penalties
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

    # Additional reward if both conditions are met
    if target_selection_min <= total_selection <= target_selection_max and target_calories_min <= average_calories_per_day <= target_calories_max:
        reward += 10  # Extra reward for meeting both conditions

    # Add a penalty of -1 for each step taken
    reward -= 1

    # Determine if the episode should terminate based on both conditions being met
    total_calories = np.sum(calories_selected_ingredients)
    terminated = (target_selection_min <= total_selection <= target_selection_max and
                  target_calories_min <= total_calories <= target_calories_max)

    info = self._get_info(total_selection, average_calories_per_day, calories_selected_ingredients)

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

def reward_complex(self, action):
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

def reward_nutrient_env4(self, action):
    
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
        'calorie_reward': 0,
        'fat_reward': 0,
        'saturates_reward': 0,
        'carbs_reward': 0,
        'sugar_reward': 0,
        'fibre_reward': 0,
        'protein_reward': 0,
        'salt_reward': 0
    }
    step_penalty = -1  # Negative reward for each step taken

    # Define target ranges
    target_ranges = {
        'calories': (self.target_calories - 50, self.target_calories + 50),
        'fat': (0, self.target_Fat_g),
        'saturates': (0, self.target_Saturates_g),
        'carbs': (self.target_Carbs_g, float('inf')),
        'sugar': (0, self.target_Sugars_g),
        'fibre': (self.target_Fibre_g, float('inf')),
        'protein': (self.target_Protein_g, float('inf')),
        'salt': (0, self.target_Salt_g)
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

    for nutrient, average_value in nutrient_values.items():
        target_min, target_max = target_ranges[nutrient]
        distance = min(abs(average_value - target_min), abs(average_value - target_max))

        if nutrient == 'calories':
            if distance > self.max_possible_calories:
                nutrient_rewards['calorie_reward'] -= 10
                terminated = True
            elif target_min <= average_value <= target_max:
                nutrient_rewards['calorie_reward'] += 10
                terminated = True
            else:
                nutrient_rewards['calorie_reward'] -= (distance ** 1) / 500
                terminated = False
        else:
            if nutrient in ['carbs', 'fibre', 'protein']:
                if average_value >= target_min:
                    nutrient_rewards[f'{nutrient}_reward'] += 10
                else:
                    nutrient_rewards[f'{nutrient}_reward'] -= (distance ** 1) / 500
            else:
                if average_value <= target_max:
                    nutrient_rewards[f'{nutrient}_reward'] += 10
                else:
                    nutrient_rewards[f'{nutrient}_reward'] -= (distance ** 1) / 500

    # Include the step penalty in the reward calculation
    reward += sum(nutrient_rewards.values()) + step_penalty

    info = self._get_info()

    return reward, nutrient_rewards, info, terminated

