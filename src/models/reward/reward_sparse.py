class RewardCalculator:
    @staticmethod
    def nutrient_reward(main_class):
        all_nutrient_targets_met = True
        far_flag_list = []
        
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            flag = target_min <= average_value <= target_max
            main_class.reward_dict['nutrient_reward'][nutrient] += 1 if flag else -1
            if not flag:
                all_nutrient_targets_met = False
                far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
                
                if far_flag:
                    nutrient_name_and_reason = f"{nutrient}: {average_value} is far off from {target_min} to {target_max}"
                    far_flag_list.append(nutrient_name_and_reason)
        
        if main_class.verbose > 2 and len(far_flag_list) > 4:  
            current_meal_plan = main_class._get_info()['current_meal_plan']
            print(f"\nAt episode {main_class.episode_count} step {main_class.nsteps}: Four Nutrients Far off: {nutrient_name_and_reason} with Selected Ingredients and Quantities: {current_meal_plan}.") 

        return main_class.reward_dict['nutrient_reward'], all_nutrient_targets_met

    @staticmethod
    def group_count_reward(main_class):
        def _is_within_portion_range(group):
            if main_class.ingredient_group_count[group] != 0:
                portion = main_class.ingredient_group_portion[group] / main_class.ingredient_group_count[group]
                min_target, max_target = main_class.ingredient_group_portion_targets[group]
                return min_target <= portion <= max_target
            return True

        all_group_targets_met = True
        any_group_exceeded = False
        group_exceeded = []

        protein_groups = ['non_processed_protein', 'processed_protein']
        total_protein_count = sum(main_class.ingredient_group_count[group] for group in protein_groups)
        total_protein_target = sum(main_class.ingredient_group_count_targets[group] for group in protein_groups)

        if total_protein_count >= total_protein_target:
            if not all(_is_within_portion_range(group) for group in protein_groups):
                all_group_targets_met = False
                for group in protein_groups:
                    main_class.reward_dict['ingredient_group_count_reward'][group] -= 0
            if total_protein_count > total_protein_target:
                any_group_exceeded = True
                group_exceeded.append('protein')
        else:
            all_group_targets_met = False
            for group in protein_groups:
                main_class.reward_dict['ingredient_group_count_reward'][group] -= 0

        if main_class.ingredient_group_count['confectionary'] == main_class.ingredient_group_count_targets['confectionary']:
            main_class.reward_dict['ingredient_group_count_reward']['confectionary'] += 0
        else:
            main_class.reward_dict['ingredient_group_count_reward']['confectionary'] -= 0

        for group, value in main_class.ingredient_group_count.items():
            if group in protein_groups + ['confectionary']:
                continue
            target = main_class.ingredient_group_count_targets[group]
            if value >= target and _is_within_portion_range(group):
                main_class.reward_dict['ingredient_group_count_reward'][group] += 0
                if value > target:
                    any_group_exceeded = True
                    group_exceeded.append(group)
            else:
                all_group_targets_met = False
                main_class.reward_dict['ingredient_group_count_reward'][group] -= 0

        if not all_group_targets_met:
            for group in main_class.reward_dict['ingredient_group_count_reward']:
                main_class.reward_dict['ingredient_group_count_reward'][group] -= 1
            if any_group_exceeded:
                for group in group_exceeded:
                    if 'protein' == group:
                        main_class.reward_dict['ingredient_group_count_reward']['non_processed_protein'] -= 0.5
                        main_class.reward_dict['ingredient_group_count_reward']['processed_protein'] -= 0.5
                    else:
                        main_class.reward_dict['ingredient_group_count_reward'][group] -= 1

        
        return main_class.reward_dict['ingredient_group_count_reward'], all_group_targets_met

    @staticmethod
    def environment_count_reward(main_class):
        environment_target_met = main_class.ingredient_environment_count['CO2_g'] <= main_class.target_CO2_g_per_meal
        main_class.reward_dict['ingredient_environment_count_reward']['CO2_g'] = 1 if environment_target_met else -1

        
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met

    @staticmethod
    def cost_reward(main_class):
        cost_difference = main_class.menu_cost - main_class.target_cost_per_meal
        cost_targets_met = cost_difference <= 0
        main_class.reward_dict['cost_reward']['from_target'] = -1 if not cost_targets_met else 1

        
        return main_class.reward_dict['cost_reward'], cost_targets_met

    @staticmethod
    def consumption_reward(main_class):
        consumption_targets_met = True
        main_class.reward_dict['consumption_reward'][0]['average_mean_consumption'] = 0

        
        return main_class.reward_dict['consumption_reward'][0], consumption_targets_met

    @staticmethod
    def termination_reason(main_class, nutrition_targets_met, group_targets_met, environment_targets_met, cost_targets_met, consumption_targets_met):
        terminated = False
        targets_not_met = []
        termination_reward = 0
        
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
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 1000

        # List of all targets
        targets_met = [nutrition_targets_met, group_targets_met, environment_targets_met, cost_targets_met, consumption_targets_met]

        # Count the number of failed targets
        failed_targets = sum([not target for target in targets_met])

        # Check if more than half of the targets are failed
        if main_class.nsteps > 100 and failed_targets > 4:
            terminated = True
            termination_reward -= 1000

        return terminated, termination_reward, targets_not_met
