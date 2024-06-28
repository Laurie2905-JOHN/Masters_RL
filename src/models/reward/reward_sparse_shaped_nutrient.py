class RewardCalculator:
    @staticmethod
    def nutrient_reward(main_class):
        all_nutrient_targets_met = True
        terminate = False
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            true_flag = target_min <= average_value <= target_max
            if not true_flag:
                all_nutrient_targets_met = False
                far_flag = average_value < 0.25 * target_min or average_value > 4 * target_max
                average_target = (target_min + target_max) / 2
                distance = max(abs(target_min - average_value), abs(target_max - average_value))
                distance_reward = 1 - (distance / average_target)
                
                if distance_reward < 0.1:
                    main_class.reward_dict['nutrient_reward'][nutrient] = 0.1
                else:
                    main_class.reward_dict['nutrient_reward'][nutrient] = distance_reward
                    
                if far_flag:
                    main_class.reward_dict['nutrient_reward'][nutrient] -= -0.1
                    terminate = True
            else:
                main_class.reward_dict['nutrient_reward'][nutrient] += 1.5
        
        return main_class.reward_dict['nutrient_reward'], all_nutrient_targets_met, terminate

    @staticmethod
    def group_count_reward(main_class):
        def _is_within_portion_range(group):
            if main_class.ingredient_group_count[group] != 0:
                portion = main_class.ingredient_group_portion[group] / main_class.ingredient_group_count[group]
            else:
                portion = 0
            min_target, max_target = main_class.ingredient_group_portion_targets[group]
            terminate = portion > 400
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
                portion_flag, portion_terminate = _is_within_portion_range(group)
                target_met = value >= main_class.ingredient_group_count_targets[group] and portion_flag
                if target_met:
                    main_class.reward_dict['ingredient_group_count_reward'][group] += 1
                else:
                    all_group_targets_met = False

                # Track if any group causes termination by choosing too large of a portion
                if portion_terminate:
                    terminate = True

        return main_class.reward_dict['ingredient_group_count_reward'], all_group_targets_met, terminate


    @staticmethod
    def environment_count_reward(main_class):
        environment_target_met = True
        # Terminate always set to false as there is no termination condition for environment count
        terminate = False
        for metric, value in main_class.ingredient_environment_count.items():
            main_class.reward_dict['ingredient_environment_count_reward'][metric] = value
        
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def cost_reward(main_class):
        terminate = False
        cost_targets_met = main_class.menu_cost['cost'] <= main_class.target_cost_per_meal
        
        if cost_targets_met:
            main_class.reward_dict['cost_reward']['cost'] += 1.5
        else:
            far_flag = main_class.menu_cost['cost'] > main_class.target_cost_per_meal * 3
            distance_reward = 1 - ( (main_class.menu_cost['cost'] - main_class.target_cost_per_meal) / main_class.target_cost_per_meal)
            
            if distance_reward < 0.1:
                main_class.reward_dict['cost_reward']['cost'] = 0.1
            else:
                main_class.reward_dict['cost_reward']['cost'] = distance_reward

            if far_flag:
                main_class.reward_dict['cost_reward']['cost'] -= 0.1
                terminate = True

        return main_class.reward_dict['cost_reward'], cost_targets_met, terminate
    
    @staticmethod
    def co2g_reward(main_class):
        terminate = False
        co2_targets_met = main_class.co2g['co2_g'] <= main_class.target_CO2_g_per_meal
        main_class.reward_dict['co2g_reward']['co2_g'] = 1 if co2_targets_met else 0
        
        if main_class.co2g['co2_g'] > 2000:
            terminate = True
        
        return main_class.reward_dict['co2g_reward'], co2_targets_met, terminate

    @staticmethod
    def consumption_reward(main_class):
        # Cosumption target always met as there is no target, currently not being used
        consumption_targets_met = True
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0

        terminate = False
            
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate

    @staticmethod
    def termination_reason(main_class, targets, termination_reasons):
        terminated = False
        targets_not_met = []
        termination_reward = 0
        failed_targets = 0
        for key, value in targets.items():
            if not value:
                targets_not_met.append(key)
                # Count the number of failed targets
                failed_targets += 1

        if not targets_not_met:
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 100

        

        # Check if more than half of the targets are failed
        if main_class.nsteps > 50 and failed_targets > 4:
            if main_class.verbose > 1:
                print('Terminated as half the targets are not met')
                print('Failed targets:', targets_not_met)
            terminated = True
            termination_reward -= 100
            
        if main_class.nsteps > 50 and len(termination_reasons) > 0:
            if main_class.verbose > 1:
                print('Terminated as metrics are out of bounds')
                print("Termination triggered due to:", ", ".join(termination_reasons))
            
            terminated = True
            termination_reward -= 100

        return terminated, termination_reward, targets_not_met
