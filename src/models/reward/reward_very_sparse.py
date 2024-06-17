import numpy as np

# Function to calculate the reward for the nutrients
def nutrient_reward(self, nutrient_rewards, nutrient_far_flag_list):
    
    # Flag for if all nutrient targets are met
    all_nutrient_targets_met = True
    nutrient_name_and_reason = []
    # Loop through values and calculate rewards
    for nutrient, average_value in self.nutrient_averages.items():

        target_min, target_max = self.nutrient_target_ranges[nutrient]
        
        # If targets are a quarter or quadruple the average value, consider it far off
        far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
        nutrient_far_flag_list.append(far_flag)
        if far_flag and self.verbose > 0:
            nutrient_name_and_reason.append(f"{nutrient}: {average_value} is far off from {target_min} to {target_max}")

        # Check if the nutrient value is far from target and adjust rewards and termination accordingly
        if far_flag and self.nsteps > 200:
            if self.verbose > 0:
                if len(nutrient_far_flag_list) > 7 and nutrient_far_flag_list.count(True) > 4:
                    info = self._get_info()
                    print(f"At episode {self.episode_count} step {self.nsteps}: \n Four Nutrients are Far off: {nutrient_name_and_reason} with Selected Ingredients and Quantities: {info['current_meal_plan']}.")

                    
            all_nutrient_targets_met = False
        else:
            if target_min <= average_value <= target_max:
                nutrient_rewards[nutrient] += 1
            else:
                nutrient_rewards[nutrient] -= 1
    
    return nutrient_rewards, all_nutrient_targets_met, nutrient_far_flag_list

def group_count_reward(self, ingredient_group_count_rewards):

    def _is_within_portion_range(group):
        
        if self.ingredient_group_count[group] != 0:
            # Normalize by the number of ingredients in the group
            portion = self.ingredient_group_portion[group] / self.ingredient_group_count[group]
            min_target, max_target = self.ingredient_group_portion_targets[group]
            
            return min_target <= portion <= max_target
        
        else:
            
            return True

    all_group_targets_met = True
    any_group_exceeded = False
    group_exceeded = []

    # Special handling for protein groups
    protein_groups = ['non_processed_protein', 'processed_protein']
    total_protein_count = sum(self.ingredient_group_count[group] for group in protein_groups)
    total_protein_target = sum(self.ingredient_group_count_targets[group] for group in protein_groups)

    if total_protein_count >= total_protein_target:
        if not all(_is_within_portion_range(group) for group in protein_groups):
            all_group_targets_met = False
            for group in protein_groups:
                ingredient_group_count_rewards[group] -= 0

        if total_protein_count > total_protein_target:
            any_group_exceeded = True
            group_exceeded.append('protein')
    else:
        all_group_targets_met = False
        for group in protein_groups:
            ingredient_group_count_rewards[group] -= 0

    # Handle confectionary group
    if self.ingredient_group_count['confectionary'] == self.ingredient_group_count_targets['confectionary']:
        ingredient_group_count_rewards['confectionary'] += 0
    else:
        ingredient_group_count_rewards['confectionary'] -= 0

    # Loop through other groups to calculate rewards
    for group, value in self.ingredient_group_count.items():
        if group in protein_groups + ['confectionary']:
            continue

        target = self.ingredient_group_count_targets[group]

        if value >= target and _is_within_portion_range(group):
            ingredient_group_count_rewards[group] += 0
            if value > target:
                any_group_exceeded = True
                group_exceeded.append(group) 
        else:
            all_group_targets_met = False
            ingredient_group_count_rewards[group] -= 0

    # Apply negative reward if any group exceeded the target and not all targets are met
    if not all_group_targets_met:
        for group in ingredient_group_count_rewards:
            ingredient_group_count_rewards[group] -= 1
        
        if any_group_exceeded:
            for group in group_exceeded:
                if 'protein' == group:
                    ingredient_group_count_rewards['non_processed_protein'] -= 0.5
                    ingredient_group_count_rewards['processed_protein'] -= -0.5
                else:
                    ingredient_group_count_rewards[group] -= 1
    
    return ingredient_group_count_rewards, all_group_targets_met




def environment_count_reward(self, ingredient_environment_count_rewards):
    
    environment_target_met = False
    
    if self.ingredient_environment_count['CO2_g'] <= self.target_CO2_g_per_meal:
        ingredient_environment_count_rewards['CO2_g'] = 1
        environment_target_met = True
    else:
        ingredient_environment_count_rewards['CO2_g'] = -1
    
    # # Define the mapping for ratings (1 gets highest reward, 5 gets lowest)
    # rating_to_reward = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    
    # # Assign rewards based on ratings for other environmental factors
    # for factor in ['animal_welfare', 'rainforest', 'water', 'CO2_rating']:
    #     rating_value = self.ingredient_environment_count[factor]
    #     if rating_value in rating_to_reward:
    #         ingredient_environment_count_rewards[factor] = rating_to_reward[rating_value]
    
    return ingredient_environment_count_rewards, environment_target_met

def cost_reward(self, cost_rewards):
    # Calculate the cost difference
    cost_difference = self.menu_cost - self.target_cost_per_meal
    
    # Determine if cost targets are met
    cost_targets_met = cost_difference <= 0
    
    # Calculate the reward
    if cost_targets_met:
        # Positive reward for staying under or meeting the target cost
        cost_rewards['from_target'] = -1  # Reward is positive because cost_difference is negative or zero
    else:
        # Negative reward for exceeding the target cost
        cost_rewards['from_target'] = -1  # Reward is negative because cost_difference is positive

    return cost_rewards, cost_targets_met


def consumption_reward(self, consumption_reward):
    
    consumption_targets_met = True

    # Shape the reward based on the average consumption
    consumption_reward['average_mean_consumption'] = 0

    # Shape the reward based on coefficient of variation
    # Assuming a lower CV is better, we use an inverse relation for the reward
    consumption_reward['cv_penalty'] = 0

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
    
    # Initialize variables
    terminated = False
    reward = 0
    self.termination = 0  # Default to no termination
    targets_not_met = []  # List to hold targets not met

    # Check each target and add to the list if not met
    if not nutrition_targets_met:
        targets_not_met.append("Nutrition")
    if not group_targets_met:
        targets_not_met.append("Group")
    if not environment_targets_met:
        targets_not_met.append("Environment")
    if not cost_targets_met:
        targets_not_met.append("Cost")
    if not consumption_targets_met:
        targets_not_met.append("Consumption")
        

    # Check if all targets are met
    if not targets_not_met:
        if self.verbose:
            print(f"All targets met at episode {self.episode_count} step {self.nsteps}.")
        terminated = True
        self.termination = 2  # All targets met
        reward += 1000

    # Check if the step count exceeds 100
    if self.nsteps > 200:
        # Check if more than 4 nutrient metrics are far off
        if nutrient_far_flag_list and nutrient_far_flag_list.count(True) > 4:
            terminated = True
            self.termination = -1  # Indicate a target is far off
            reward -= 1000
            
    return terminated, reward, targets_not_met
