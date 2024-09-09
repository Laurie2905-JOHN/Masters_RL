from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

class BaseRewardCalculator(ABC):

    @staticmethod
    @abstractmethod
    def calculate_cost_reward(main_class: Any) -> Tuple[dict, bool, bool]:
        # Abstract method to calculate cost reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_nutrient_reward(main_class: Any) -> Tuple[dict, bool, bool]:
        # Abstract method to calculate nutrient reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_group_count_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        # Abstract method to calculate group count reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_environment_count_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        # Abstract method to calculate environment count reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_co2g_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        # Abstract method to calculate CO2 emissions reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_consumption_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        # Abstract method to calculate consumption reward
        pass

    @staticmethod
    @abstractmethod
    def calculate_step_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        # Abstract method to calculate step reward
        pass

    @staticmethod
    def cost_target(main_class: Any) -> bool:
        # Check if cost target is met
        if not main_class.total_average_portion < sum(main_class.current_selection):
            cost_target_met = False
        else:
            cost_target_met = main_class.menu_cost['cost'] <= main_class.target_cost_per_meal
        return cost_target_met

    @staticmethod
    def environment_count_target(main_class: Any) -> bool:
        # Check if environment count target is met
        environment_count_target_met = sum(main_class.ingredient_environment_count.values()) > 0.5
        return environment_count_target_met

    @staticmethod
    def co2g_target(main_class: Any) -> bool:
        # Check if CO2 emissions target is met
        if main_class.total_average_portion < sum(main_class.current_selection):
            co2g_target = main_class.co2_g['co2e_g'] <= main_class.target_co2_g_per_meal
        else:
            co2g_target = False
        return co2g_target

    @staticmethod
    def consumption_target(main_class: Any) -> bool:
        # Check if consumption target is met
        consumption_targets_met = True
        return consumption_targets_met

    @staticmethod
    def is_confectionary_target_met(main_class) -> bool:
        # Check if the confectionary target is met
        return main_class.ingredient_group_count['confectionary'] == main_class.ingredient_group_count_targets['confectionary']

    @staticmethod
    def is_within_portion_range(main_class, group: str) -> Tuple[bool, bool]:
        # Check if the portion range for a group is within the target range
        portion = (
            main_class.ingredient_group_portion[group] / main_class.ingredient_group_count[group]
            if main_class.ingredient_group_count[group] != 0 else 0
        )
        min_target, max_target = main_class.ingredient_group_portion_targets[group]
        terminate = portion > 300
        return min_target <= portion <= max_target, terminate

    @staticmethod
    def calculate_distance_reward(value: float, target_min: float, target_max: float, distance_factor: float = 1.0) -> float:
        # Calculate the distance reward based on how far the value is from the target range
        average_target = (target_min + target_max) / 2
        distance = max(abs(target_min - value), abs(target_max - value)) * distance_factor
        distance_reward = 1 - (distance / average_target)
        return max(0.1, distance_reward)

    @staticmethod
    def determine_final_termination(main_class: Any, targets: Dict[str, bool], main_terminate) -> Tuple[bool, float, List[str]]:
        # Determine final termination condition
        terminated = False
        failed_targets = [key for key, met in targets.items() if not met]
        termination_reward = 0

        if not failed_targets:
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 500
        elif main_terminate:
            reward_per_target_met = 500 / len(targets)
            termination_reward += reward_per_target_met * (len(targets) - len(failed_targets))
            if main_class.verbose > 0:
                print(f"{len(failed_targets)} targets failed at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True

        main_class.reward_dict['targets_not_met'] = failed_targets

        return terminated, termination_reward, failed_targets

    @staticmethod
    def common_calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Common logic to calculate group count reward
        all_targets_met = True
        terminate = False
        for group, count in main_class.ingredient_group_count.items():
            if group != 'confectionary':
                portion_flag, terminate = BaseRewardCalculator.is_within_portion_range(main_class, group)
                target_met = count >= main_class.ingredient_group_count_targets[group] and portion_flag
                if not target_met:
                    all_targets_met = False
            else:
                if not BaseRewardCalculator.is_confectionary_target_met(main_class):
                    all_targets_met = False
        return main_class.reward_dict['ingredient_group_count_reward'], all_targets_met, terminate


    @staticmethod
    def calculate_step_reward(main_class):
        # Initialize step reward
        main_class.reward_dict['step_reward'] = 0
    
class ShapedRewardCalculator(BaseRewardCalculator):
    
    @staticmethod
    def calculate_cost_reward(main_class) -> Tuple[dict, bool, bool]:
        # Initialize termination flag
        terminate = False
        # Check if cost target is met
        cost_target_met = BaseRewardCalculator.cost_target(main_class)
        if cost_target_met:
            # If cost target is met, increment cost reward
            main_class.reward_dict['cost_reward']['cost'] += 1.5
        else:
            # If cost target is not met, calculate distance reward
            target_min = main_class.target_cost_per_meal
            target_max = main_class.target_cost_per_meal
            distance_reward = BaseRewardCalculator.calculate_distance_reward(
                main_class.menu_cost['cost'], target_min, target_max
            )
            main_class.reward_dict['cost_reward']['cost'] = distance_reward
            # Calculate cost difference and check if it exceeds threshold
            cost_difference = main_class.menu_cost['cost'] - main_class.target_cost_per_meal
            far_flag = cost_difference > 3
            if far_flag:
                # Penalize and terminate if cost difference is too large
                main_class.reward_dict['cost_reward']['cost'] -= 0.1
                terminate = True

        return main_class.reward_dict['cost_reward'], cost_target_met, terminate

    @staticmethod
    def calculate_nutrient_reward(main_class) -> Tuple[dict, bool, bool]:
        # Initialize flags
        all_targets_met = True
        terminate = False
        # Iterate through nutrient averages
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            if target_min <= average_value <= target_max:
                # If nutrient target is met, increment nutrient reward
                main_class.reward_dict['nutrient_reward'][nutrient] += 1.5
            else:
                # If nutrient target is not met, calculate distance reward
                all_targets_met = False
                distance_reward = BaseRewardCalculator.calculate_distance_reward(
                    average_value, target_min, target_max
                )
                main_class.reward_dict['nutrient_reward'][nutrient] = distance_reward
                # Check if nutrient value is far from the target range
                far_flag = average_value < 0.5 * target_min or average_value > target_max * 2
                if far_flag:
                    # Penalize and terminate if nutrient value is too far
                    main_class.reward_dict['nutrient_reward'][nutrient] -= 0.1
                    terminate = True

        return main_class.reward_dict['nutrient_reward'], all_targets_met, terminate

    @staticmethod
    def calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Initialize flags
        all_targets_met = True
        terminate = False
        if main_class.algo != "MASKED_PPO":
            # Check if confectionary target is met
            if BaseRewardCalculator.is_confectionary_target_met(main_class):
                main_class.reward_dict['ingredient_group_count_reward']['confectionary'] += 1
            else:
                all_targets_met = False
            # Iterate through ingredient group counts
            for group, count in main_class.ingredient_group_count.items():
                if group != 'confectionary':
                    portion_flag, terminate = BaseRewardCalculator.is_within_portion_range(main_class, group)
                    target_met = count >= main_class.ingredient_group_count_targets[group] and portion_flag
                    if target_met:
                        # Increment reward if group target is met
                        main_class.reward_dict['ingredient_group_count_reward'][group] += 1
                    else:
                        all_targets_met = False
        return main_class.reward_dict['ingredient_group_count_reward'], all_targets_met, terminate

    @staticmethod
    def calculate_environment_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Check if environment count target is met
        environment_target_met = BaseRewardCalculator.environment_count_target(main_class)
        terminate = False
        # Iterate through environment counts
        for metric, value in main_class.ingredient_environment_count.items():
            if value < 0:
                value = 0
            # Normalize and assign rewards
            main_class.reward_dict['ingredient_environment_count_reward'][metric] = value / 4
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def calculate_co2g_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Initialize termination flag
        terminate = False
        # Check if CO2 target is met
        co2_target_met = BaseRewardCalculator.co2g_target(main_class)
        if co2_target_met:
            # If CO2 target is met, assign high reward
            main_class.reward_dict['co2g_reward']['co2e_g'] = 2
        else:
            # If CO2 target is not met, calculate distance reward
            target_min = main_class.target_co2_g_per_meal
            target_max = main_class.target_co2_g_per_meal
            distance_reward = BaseRewardCalculator.calculate_distance_reward(
                main_class.co2_g['co2e_g'], target_min, target_max
            )
            main_class.reward_dict['co2g_reward']['co2e_g'] = distance_reward
            # Check if CO2 value is far from the target range
            far_flag = main_class.co2_g['co2e_g'] > 1500
            if far_flag:
                # Penalize and terminate if CO2 value is too high
                main_class.reward_dict['co2g_reward']['co2e_g'] -= 0.1
                terminate = True
        return main_class.reward_dict['co2g_reward'], co2_target_met, terminate

    @staticmethod
    def calculate_consumption_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Check if consumption target is met
        consumption_targets_met = BaseRewardCalculator.consumption_target(main_class)
        # Initialize consumption reward
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0
        terminate = False
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate

   
    
class SemiSparseRewardCalculator(BaseRewardCalculator):
    @staticmethod
    def calculate_cost_reward(main_class) -> Tuple[dict, bool, bool]:
        # Check if the cost target is met
        terminate = False
        cost_target_met = BaseRewardCalculator.cost_target(main_class)
        if not cost_target_met:
            # Check if the cost difference is far from the target
            cost_difference = main_class.menu_cost['cost'] - main_class.target_cost_per_meal
            far_flag = cost_difference > 3
            if far_flag:
                terminate = True
        return main_class.reward_dict['cost_reward'], cost_target_met, terminate

    @staticmethod
    def calculate_nutrient_reward(main_class) -> Tuple[dict, bool, bool]:
        # Check if all nutrient targets are met
        all_targets_met = True
        terminate = False
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            if not target_min <= average_value <= target_max:
                all_targets_met = False
                far_flag = average_value < 0.5 * target_min or average_value > target_max * 2
                if far_flag:
                    terminate = True
        return main_class.reward_dict['nutrient_reward'], all_targets_met, terminate

    @staticmethod
    def calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Use the common logic to calculate group count reward
        return BaseRewardCalculator.common_calculate_group_count_reward(main_class)

    @staticmethod
    def calculate_environment_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Check if the environment count target is met
        environment_target_met = BaseRewardCalculator.environment_count_target(main_class)
        terminate = False
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def calculate_co2g_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Check if the CO2 target is met
        terminate = False
        co2_target_met = BaseRewardCalculator.co2g_target(main_class)
        far_flag = main_class.co2_g['co2e_g'] > 2000
        if far_flag:
            terminate = True
        return main_class.reward_dict['co2g_reward'], co2_target_met, terminate

    @staticmethod
    def calculate_consumption_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        # Check if the consumption target is met
        consumption_targets_met = BaseRewardCalculator.consumption_target(main_class)
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0
        terminate = False
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate

    @staticmethod
    def determine_final_termination(main_class: Any, targets: Dict[str, bool], main_terminate) -> Tuple[bool, float, List[str]]:
        # Determine final termination condition and assign rewards for each met target
        terminated = False
        failed_targets = [key for key, met in targets.items() if not met]
        termination_reward = 0

        if not failed_targets:
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 10  # Total reward for meeting all targets

        for target, met in targets.items():
            if met:
                termination_reward += 1

        if main_terminate and failed_targets:
            if main_class.verbose > 0:
                print(f"{len(failed_targets)} targets failed at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True

        main_class.reward_dict['targets_not_met'] = failed_targets

        return terminated, termination_reward, failed_targets


class SparseRewardCalculator(BaseRewardCalculator):
    @staticmethod
    def calculate_cost_reward(main_class) -> Tuple[dict, bool, bool]:
        terminate = False
        cost_target_met = BaseRewardCalculator.cost_target(main_class)
        return main_class.reward_dict['cost_reward'], cost_target_met, terminate

    @staticmethod
    def calculate_nutrient_reward(main_class) -> Tuple[dict, bool, bool]:
        all_targets_met = True
        terminate = False
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            if not target_min <= average_value <= target_max:
                all_targets_met = False
                far_flag = average_value < 0.5 * target_min or average_value > target_max * 2
                if far_flag:
                    terminate = True
        return main_class.reward_dict['nutrient_reward'], all_targets_met, terminate

    @staticmethod
    def calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        return BaseRewardCalculator.common_calculate_group_count_reward(main_class)

    @staticmethod
    def calculate_environment_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        environment_target_met = BaseRewardCalculator.environment_count_target(main_class)
        terminate = False
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def calculate_co2g_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        terminate = False
        co2_target_met = BaseRewardCalculator.co2g_target(main_class)
        far_flag = main_class.co2_g['co2e_g'] > 2000
        if far_flag:
            terminate = True
        return main_class.reward_dict['co2g_reward'], co2_target_met, terminate

    @staticmethod
    def calculate_consumption_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        consumption_targets_met = BaseRewardCalculator.consumption_target(main_class)
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0
        terminate = False
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate

    @staticmethod
    def calculate_step_reward(main_class):
        main_class.reward_dict['step_reward'] = 0
        return main_class.reward_dict['step_reward'], True, False
    
    @staticmethod
    def determine_final_termination(main_class: Any, targets: Dict[str, bool], main_terminate) -> Tuple[bool, float, List[str]]:
        # Determine final termination condition, no reward unless all targets met
        terminated = False
        failed_targets = [key for key, met in targets.items() if not met]
        termination_reward = 0

        if not failed_targets:
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 10 # Reward for selecting all targets
        else:
            if main_class.verbose > 0:
                print(f"{len(failed_targets)} targets failed at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True

        main_class.reward_dict['targets_not_met'] = failed_targets

        return terminated, termination_reward, failed_targets
