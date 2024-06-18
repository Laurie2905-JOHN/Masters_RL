import numpy as np

class AdvancedRewardCalculator:
    
    @staticmethod
    def nutrient_reward(nutrient_averages, nutrient_target_ranges, nutrient_rewards, nutrient_far_flag_list, verbose=0):
        all_nutrient_targets_met = True
        
        for nutrient, average_value in nutrient_averages.items():
            target_min, target_max = nutrient_target_ranges[nutrient]
            far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
            nutrient_far_flag_list.append(far_flag)
            
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

    @staticmethod
    def group_count_reward(ingredient_group_count, ingredient_group_portion, ingredient_group_count_rewards, ingredient_group_count_targets, ingredient_group_portion_targets):
        def _is_within_portion_range(group):
            if ingredient_group_count[group] != 0:
                portion = ingredient_group_portion[group] / ingredient_group_count[group]
                min_target, max_target = ingredient_group_portion_targets[group]
                return min_target <= portion <= max_target
            return True

        all_group_targets_met = True
        any_group_exceeded = False
        group_exceeded = []

        protein_groups = ['non_processed_protein', 'processed_protein']
        total_protein_count = sum(ingredient_group_count[group] for group in protein_groups)
        total_protein_target = sum(ingredient_group_count_targets[group] for group in protein_groups)

        if total_protein_count >= total_protein_target:
            if all(_is_within_portion_range(group) for group in protein_groups):
                for group in protein_groups:
                    ingredient_group_count_rewards[group] += 50
            else:
                all_group_targets_met = False
                for group in protein_groups:
                    ingredient_group_count_rewards[group] -= 25

            if total_protein_count > total_protein_target:
                any_group_exceeded = True
                group_exceeded.append('protein')
        else:
            all_group_targets_met = False
            for group in protein_groups:
                ingredient_group_count_rewards[group] -= 50

        if ingredient_group_count['confectionary'] == ingredient_group_count_targets['confectionary']:
            ingredient_group_count_rewards['confectionary'] += 100
        else:
            ingredient_group_count_rewards['confectionary'] -= 100

        for group, value in ingredient_group_count.items():
            if group in protein_groups + ['confectionary']:
                continue

            target = ingredient_group_count_targets[group]

            if value >= target:
                if value > target:
                    any_group_exceeded = True
                    group_exceeded.append(group)
                
                if _is_within_portion_range(group):
                    ingredient_group_count_rewards[group] += 100
                else:
                    all_group_targets_met = False
                    ingredient_group_count_rewards[group] -= 50
            else:
                all_group_targets_met = False
                ingredient_group_count_rewards[group] -= 110

        if not all_group_targets_met and any_group_exceeded:
            for group in group_exceeded:
                if group == 'protein':
                    for protein_group in protein_groups:
                        ingredient_group_count_rewards[protein_group] -= 50
                else:
                    ingredient_group_count_rewards[group] -= 100

        return ingredient_group_count_rewards, all_group_targets_met

    @staticmethod
    def environment_count_reward(ingredient_environment_count_rewards, ingredient_environment_count, target_CO2_g_per_meal):
        CO2_distance = abs(target_CO2_g_per_meal - ingredient_environment_count['CO2_g'])
        environment_target_met = False
        
        if ingredient_environment_count['CO2_g'] <= target_CO2_g_per_meal:
            ingredient_environment_count_rewards['CO2_g'] = CO2_distance / 10
            environment_target_met = True
        else:
            if CO2_distance > 500:
                ingredient_environment_count_rewards['CO2_g'] = -100
            else:
                ingredient_environment_count_rewards['CO2_g'] = -CO2_distance / 50

        rating_to_reward = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
        
        for factor in ['animal_welfare', 'rainforest', 'water', 'CO2_rating']:
            rating_value = ingredient_environment_count[factor]
            if rating_value in rating_to_reward:
                ingredient_environment_count_rewards[factor] = rating_to_reward[rating_value]
        
        return ingredient_environment_count_rewards, environment_target_met

    @staticmethod
    def cost_reward(cost_rewards, menu_cost, target_cost_per_meal):
        cost_difference = menu_cost - target_cost_per_meal
        cost_targets_met = cost_difference <= 0
        
        if cost_targets_met:
            cost_rewards['from_target'] = -cost_difference * 10
        else:
            cost_rewards['from_target'] = -cost_difference * 10

        return cost_rewards, cost_targets_met

    @staticmethod
    def consumption_reward(consumption_reward, consumption_average):
        consumption_targets_met = True

        consumption_reward['average_mean_consumption'] = consumption_average['average_mean_consumption'] * 3.75
        consumption_reward['cv_penalty'] = consumption_average['average_cv_ingredients'] * 6.5

        return consumption_reward, consumption_targets_met

    @staticmethod
    def termination_reason(nutrition_targets_met, group_targets_met, environment_targets_met, cost_targets_met, consumption_targets_met, nutrient_far_flag_list, reward):
        terminated = False
        reward = 0
        termination = 0
        targets_not_met = []

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

        if not targets_not_met:
            terminated = True
            termination = 2
            reward += 1e8
        elif nutrient_far_flag_list:
            if nutrient_far_flag_list.count(True) > 4:
                terminated = True
                termination = -1
                reward -= 1000

        return terminated, reward, targets_not_met, termination
