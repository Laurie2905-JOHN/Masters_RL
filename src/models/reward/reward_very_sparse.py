class RewardCalculator:
    @staticmethod
    def nutrient_reward(main_class):
        all_nutrient_targets_met = True
        terminate = False
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            true_flag = target_min <= average_value <= target_max
            main_class.reward_dict['nutrient_reward'][nutrient] += 1 if true_flag else -1
            if not true_flag:
                all_nutrient_targets_met = False
                far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
                if far_flag:
                    terminate = True
        
        return main_class.reward_dict['nutrient_reward'], all_nutrient_targets_met, terminate

    @staticmethod
    def group_count_reward(main_class):
        def _is_within_portion_range(group):
            if main_class.ingredient_group_count[group] != 0:
                portion = main_class.ingredient_group_portion[group] / main_class.ingredient_group_count[group]
            else:
                portion = 0
            min_target, max_target = main_class.ingredient_group_portion_targets[group]
            terminate = portion > 200
            return min_target <= portion <= max_target, terminate

        all_group_targets_met = True
        terminate = False

        # Handle confectionary separately as it has a different target
        if main_class.ingredient_group_count['confectionary'] == main_class.ingredient_group_count_targets['confectionary']:
            main_class.reward_dict['ingredient_group_count_reward']['confectionary'] += 1
        else:
            all_group_targets_met = False

        # Handle other groups
        for group, value in main_class.ingredient_group_count.items():
            if group != 'confectionary':
                portion_flag, group_terminate = _is_within_portion_range(group)
                target_met = value >= main_class.ingredient_group_count_targets[group] and portion_flag
                if target_met:
                    main_class.reward_dict['ingredient_group_count_reward'][group] += 1
                else:
                    all_group_targets_met = False

                # Track if any group causes termination
                if group_terminate:
                    terminate = True

        return main_class.reward_dict['ingredient_group_count_reward'], all_group_targets_met, terminate


    @staticmethod
    def environment_count_reward(main_class):
        environment_target_met = main_class.ingredient_environment_count['CO2_g'] <= main_class.target_CO2_g_per_meal
        main_class.reward_dict['ingredient_environment_count_reward']['CO2_g'] = 1 if environment_target_met else 0
        
        if main_class.ingredient_environment_count['CO2_g'] > 1000:
            terminate = True
        else:
            terminate = False
        
        
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def cost_reward(main_class):
        cost_difference = main_class.menu_cost - main_class.target_cost_per_meal
        cost_targets_met = cost_difference <= 0
        main_class.reward_dict['cost_reward']['from_target'] = 0 if not cost_targets_met else 1
        
        
        if cost_difference > 4:
            terminate = True
        else:
            terminate = False
        
        return main_class.reward_dict['cost_reward'], cost_targets_met, terminate

    @staticmethod
    def consumption_reward(main_class):
        consumption_targets_met = True
        main_class.reward_dict['consumption_reward'][0]['average_mean_consumption'] = 0

        terminate = False
            
        return main_class.reward_dict['consumption_reward'][0], consumption_targets_met, terminate

    @staticmethod
    def termination_reason(main_class, nutrition_targets_met, group_targets_met, environment_targets_met, cost_targets_met, consumption_targets_met, termination_reasons):
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
        if main_class.nsteps > 50 and failed_targets > 4:
            if main_class.verbose > 1:
                print('Terminated as half the targets are not met')
                print('Failed targets:', targets_not_met)
            terminated = True
            termination_reward -= 1000
            
        if len(termination_reasons) > 0:
            if main_class.verbose > 1:
                print('Terminated as metrics are out of bounds')
                print("Termination triggered due to:", ", ".join(termination_reasons))
            
            terminated = True
            termination_reward -= 1000

        return terminated, termination_reward, targets_not_met
