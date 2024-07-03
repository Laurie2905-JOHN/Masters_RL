from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

class BaseRewardCalculator(ABC):

    @staticmethod
    @abstractmethod
    def calculate_cost_reward(main_class: Any) -> Tuple[dict, bool, bool]:
        pass

    @staticmethod
    @abstractmethod
    def calculate_nutrient_reward(main_class: Any) -> Tuple[dict, bool, bool]:
        pass

    @staticmethod
    @abstractmethod
    def calculate_group_count_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        pass

    @staticmethod
    @abstractmethod
    def calculate_environment_count_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        pass

    @staticmethod
    @abstractmethod
    def calculate_co2g_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        pass

    @staticmethod
    @abstractmethod
    def calculate_consumption_reward(main_class: Any) -> Tuple[Dict[str, float], bool, bool]:
        pass
    
    @staticmethod
    def is_confectionary_target_met(main_class) -> bool:
        """
        Check if the confectionary target is met.
        
        Args:
            main_class: The main class instance containing confectionary data.
        
        Returns:
            A boolean indicating if the confectionary target is met.
        """
        return main_class.ingredient_group_count['confectionary'] == main_class.ingredient_group_count_targets['confectionary']

    @staticmethod
    def is_within_portion_range(main_class, group: str) -> Tuple[bool, bool]:
        """
        Check if the portion range for a group is within the target range.
        
        Args:
            main_class: The main class instance containing portion data.
            group: The group to check.
        
        Returns:
            A tuple containing:
                - A boolean indicating if the portion is within the target range.
                - A boolean indicating if the process should terminate.
        """
        portion = (
            main_class.ingredient_group_portion[group] / main_class.ingredient_group_count[group]
            if main_class.ingredient_group_count[group] != 0 else 0
        )
        min_target, max_target = main_class.ingredient_group_portion_targets[group]
        # if any one portion greater than 400, terminate
        terminate = portion > 400
        return min_target <= portion <= max_target, terminate
    
    @staticmethod
    def calculate_distance_reward(value: float, target_min: float, target_max: float, distance_factor: float = 1.0) -> Tuple[float, bool]:
        """
        Calculate the distance reward based on how far the value is from the target range.
        
        Args:
            value: The value to evaluate (can be nutrient, cost, etc.).
            target_min: The minimum target value.
            target_max: The maximum target value.
            distance_factor: A factor to adjust the distance calculation.
            far_threshold: A factor to determine if the value is far from the target range.
        
        Returns:
            A tuple containing:
                - The calculated distance reward.
                - A boolean indicating if the value is far from the target range.
        """
        average_target = (target_min + target_max) / 2
        distance = max(abs(target_min - value), abs(target_max - value)) * distance_factor
        distance_reward = 1 - (distance / average_target)
        
        return distance_reward
    
    @staticmethod
    def determine_termination(main_class: Any, targets: Dict[str, bool], termination_reasons: List[str]) -> Tuple[bool, float, List[str]]:
        terminated = False
        failed_targets = [key for key, met in targets.items() if not met]
        termination_reward = 0

        if not failed_targets:
            if main_class.verbose > 0:
                print(f"All targets met at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True
            termination_reward += 100
        else:
            reward_per_target_met = 100 / len(targets)
            termination_reward += reward_per_target_met * (len(targets) - len(failed_targets))
            if main_class.verbose > 0:
                print(f"{len(failed_targets)} targets failed at episode {main_class.episode_count} step {main_class.nsteps}.")
            terminated = True

        if termination_reasons:
            if main_class.verbose > 1:
                print("Termination triggered due to:", ", ".join(termination_reasons))
            terminated = True

        if main_class.verbose > 1 and terminated:
            print('Terminated as metrics are out of bounds')
            print('Failed targets:', failed_targets)

        return terminated, termination_reward, failed_targets

class ShapedRewardCalculator(BaseRewardCalculator):
    @staticmethod
    def calculate_cost_reward(main_class) -> Tuple[dict, bool, bool]:
        """
        Calculate the cost reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing cost data.
        
        Returns:
            A tuple containing:
                - A dictionary with cost rewards.
                - A boolean indicating if the cost target was met.
                - A boolean indicating if the process should terminate.
        """
        terminate = False
        cost_target_met = main_class.menu_cost['cost'] <= main_class.target_cost_per_meal
        
        if cost_target_met:
            main_class.reward_dict['cost_reward']['cost'] += 1.5
        else:
            target_min = main_class.target_cost_per_meal
            target_max = main_class.target_cost_per_meal
            distance_reward = BaseRewardCalculator.calculate_distance_reward(
                main_class.menu_cost['cost'], target_min, target_max
            )
            main_class.reward_dict['cost_reward']['cost'] = max(0.1, distance_reward)
            cost_difference = main_class.menu_cost['cost'] - main_class.target_cost_per_meal
            
            # If the cost is far from the target range, penalize the reward and terminate  
            far_flag = cost_difference > 3
            if far_flag:
                main_class.reward_dict['cost_reward']['cost'] -= 0.1
                terminate = True

        return main_class.reward_dict['cost_reward'], cost_target_met, terminate

    @staticmethod
    def calculate_nutrient_reward(main_class) -> Tuple[dict, bool, bool]:
        """
        Calculate the nutrient reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing nutrient data.
        
        Returns:
            A tuple containing:
                - A dictionary with nutrient rewards.
                - A boolean indicating if all nutrient targets were met.
                - A boolean indicating if the process should terminate.
        """
        all_targets_met = True
        terminate = False

        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            if target_min <= average_value <= target_max:
                main_class.reward_dict['nutrient_reward'][nutrient] += 1.5
            else:
                all_targets_met = False
                distance_reward = BaseRewardCalculator.calculate_distance_reward(
                    average_value, target_min, target_max
                )
                main_class.reward_dict['nutrient_reward'][nutrient] = max(0.1, distance_reward)
                
                far_flag = average_value < 0.25 * target_min or average_value > target_max * 4
                if far_flag:
                    main_class.reward_dict['nutrient_reward'][nutrient] -= 0.1
                    terminate = True

        return main_class.reward_dict['nutrient_reward'], all_targets_met, terminate


    @staticmethod
    def calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the group count reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing group count data.
        
        Returns:
            A tuple containing:
                - A dictionary with group count rewards.
                - A boolean indicating if all group targets were met.
                - A boolean indicating if the process should terminate.
        """
        all_targets_met = True
        terminate = False

        if BaseRewardCalculator.is_confectionary_target_met(main_class):
            main_class.reward_dict['ingredient_group_count_reward']['confectionary'] += 1
        else:
            all_targets_met = False

        for group, count in main_class.ingredient_group_count.items():
            if group != 'confectionary':
                portion_flag, terminate = BaseRewardCalculator.is_within_portion_range(main_class, group)
                target_met = count >= main_class.ingredient_group_count_targets[group] and portion_flag
                if target_met:
                    main_class.reward_dict['ingredient_group_count_reward'][group] += 1
                else:
                    all_targets_met = False

        return main_class.reward_dict['ingredient_group_count_reward'], all_targets_met, terminate

    @staticmethod
    def calculate_environment_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the environment count reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing environment count data.
        
        Returns:
            A tuple containing:
                - A dictionary with environment count rewards.
                - A boolean indicating if all environment targets were met.
                - A boolean indicating if the process should terminate.
        """
        environment_target_met = True
        terminate = False
        for metric, value in main_class.ingredient_environment_count.items():
            main_class.reward_dict['ingredient_environment_count_reward'][metric] = value
        
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def calculate_co2g_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the CO2 emissions reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing CO2 emissions data.
        
        Returns:
            A tuple containing:
                - A dictionary with CO2 emissions rewards.
                - A boolean indicating if the CO2 target was met.
                - A boolean indicating if the process should terminate.
        """
        terminate = False
        co2_target_met = main_class.co2_g['co2_g'] <= main_class.target_co2_g_per_meal
        
        if co2_target_met:
            main_class.reward_dict['co2g_reward']['co2_g'] = 0
        else:
            target_min = main_class.target_co2_g_per_meal
            target_max = main_class.target_co2_g_per_meal
            distance_reward = BaseRewardCalculator.calculate_distance_reward(
                main_class.co2_g['co2_g'], target_min, target_max
            )
            main_class.reward_dict['co2g_reward']['co2_g'] = max(0.1, distance_reward)
            far_flag = main_class.co2_g['co2_g'] > 2000
            
            if far_flag:
                main_class.reward_dict['co2g_reward']['co2_g'] -= 0.1
                terminate = True

        return main_class.reward_dict['co2g_reward'], co2_target_met, terminate

    @staticmethod
    def calculate_consumption_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the consumption reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing consumption data.
        
        Returns:
            A tuple containing:
                - A dictionary with consumption rewards.
                - A boolean indicating if all consumption targets were met.
                - A boolean indicating if the process should terminate.
        """
        consumption_targets_met = True
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0

        terminate = False
            
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate

class SparseRewardCalculator(BaseRewardCalculator):
    @staticmethod
    def calculate_cost_reward(main_class) -> Tuple[dict, bool, bool]:
        """
        Calculate the cost reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing cost data.
        
        Returns:
            A tuple containing:
                - A dictionary with cost rewards.
                - A boolean indicating if the cost target was met.
                - A boolean indicating if the process should terminate.
        """
        terminate = False
        cost_target_met = main_class.menu_cost['cost'] <= main_class.target_cost_per_meal
        
        if not cost_target_met:
            cost_difference = main_class.menu_cost['cost'] - main_class.target_cost_per_meal
            # If the cost is far from the target range, terminate  
            far_flag = cost_difference > 3
            if far_flag:
                terminate = True

        return main_class.reward_dict['cost_reward'], cost_target_met, terminate

    @staticmethod
    def calculate_nutrient_reward(main_class) -> Tuple[dict, bool, bool]:
        """
        Calculate the nutrient reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing nutrient data.
        
        Returns:
            A tuple containing:
                - A dictionary with nutrient rewards.
                - A boolean indicating if all nutrient targets were met.
                - A boolean indicating if the process should terminate.
        """
        all_targets_met = True
        terminate = False
        for nutrient, average_value in main_class.nutrient_averages.items():
            target_min, target_max = main_class.nutrient_target_ranges[nutrient]
            if not target_min <= average_value <= target_max:
                all_targets_met = False
                
                far_flag = average_value < 0.25 * target_min or average_value > target_max * 4
                if far_flag:
                    terminate = True

        return main_class.reward_dict['nutrient_reward'], all_targets_met, terminate


    @staticmethod
    def calculate_group_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the group count reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing group count data.
        
        Returns:
            A tuple containing:
                - A dictionary with group count rewards.
                - A boolean indicating if all group targets were met.
                - A boolean indicating if the process should terminate.
        """
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
    def calculate_environment_count_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the environment count reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing environment count data.
        
        Returns:
            A tuple containing:
                - A dictionary with environment count rewards.
                - A boolean indicating if all environment targets were met.
                - A boolean indicating if the process should terminate.
        """
        environment_target_met = True
        terminate = False
        
        return main_class.reward_dict['ingredient_environment_count_reward'], environment_target_met, terminate

    @staticmethod
    def calculate_co2g_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the CO2 emissions reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing CO2 emissions data.
        
        Returns:
            A tuple containing:
                - A dictionary with CO2 emissions rewards.
                - A boolean indicating if the CO2 target was met.
                - A boolean indicating if the process should terminate.
        """
        
        terminate = False
        co2_target_met = main_class.co2_g['co2_g'] <= main_class.target_co2_g_per_meal
        far_flag = main_class.co2_g['co2_g'] > 2000
        
        if far_flag:
            terminate = True

        return main_class.reward_dict['co2g_reward'], co2_target_met, terminate

    @staticmethod
    def calculate_consumption_reward(main_class) -> Tuple[Dict[str, float], bool, bool]:
        """
        Calculate the consumption reward for the given main_class instance.
        
        Args:
            main_class: The main class instance containing consumption data.
        
        Returns:
            A tuple containing:
                - A dictionary with consumption rewards.
                - A boolean indicating if all consumption targets were met.
                - A boolean indicating if the process should terminate.
        """
        consumption_targets_met = True
        main_class.reward_dict['consumption_reward']['average_mean_consumption'] = 0

        terminate = False
            
        return main_class.reward_dict['consumption_reward'], consumption_targets_met, terminate