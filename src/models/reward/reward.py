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
    # Calculate the current number of ingredients
    total_selection = np.sum(self.current_selection > 0)
    
    # Calculate number of ingredients which are selected in the action
    action_selection = np.sum(action > 0)
    
    # Calculate calories only from the selected ingredients
    calories_selected_ingredients = self.caloric_values * self.current_selection  
    average_calories_per_day = sum(calories_selected_ingredients) / self.num_people

    # Initialize reward
    reward = 0

    # Define target ranges
    target_calories_min = self.target_calories - 50
    target_calories_max = self.target_calories + 50

    # Caloric intake reward
    if target_calories_min <= average_calories_per_day <= target_calories_max:
        reward += 10  # Reward for being within the target range
        terminated = False # Continue the episode if within the target range as want it to learn this is optimal
    else:
        # Penalize based on how far it is from the target range
        calories_distance = min(abs(average_calories_per_day - target_calories_min), abs(average_calories_per_day - target_calories_max))
        reward -= 0.1 * calories_distance
        terminated = False  # Continue the episode if not within target range

    # Ingredient selection reward - Encourage moving towards 10 ingredients
    if total_selection < 10:
        reward += (10 - total_selection)  # Reward for reducing the number of ingredients towards 10
    elif total_selection > 10:
        reward -= (total_selection - 10)  # Penalize for selecting more than 10 ingredients

    # Penalize if the current action increases the selection excessively
    if action_selection > 5:
        reward -= 10   # Penalize for selecting too many ingredients in the current action

    # Add a penalty of -1 for each step taken
    reward -= 1

    # Create the info dictionary
    info = self._get_info(total_selection, average_calories_per_day, calories_selected_ingredients)

    return reward, info, terminated

