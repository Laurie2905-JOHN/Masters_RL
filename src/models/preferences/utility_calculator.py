import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MenuUtilityCalculator:
    def __init__(self, true_preferences: Dict[str, Dict[str, List[str]]], 
                 child_feature_data: Dict[str, Any], menu_plan_length: int, save_to_json: str) -> None:
        self.true_preferences = self._convert_true_preferences_to_dict(true_preferences)
        self.child_feature_data = child_feature_data

        self.raw_utility: Dict[int, Dict[str, float]] = {}
        self.true_raw_utility: Dict[int, Dict[str, float]] = {}
        self.predicted_raw_utility: Dict[int, Dict[str, float]] = {}
        self.true_utility: Dict[int, Dict[str, float]] = {}
        self.predicted_utility: Dict[int, Dict[str, float]] = {}
        self.previous_day_utilities: Dict[str, Optional[float]] = {child: None for child in child_feature_data.keys()}
        self.previous_day_predicted_utilities: Dict[str, Optional[float]] = {child: None for child in child_feature_data.keys()}
        self.previous_day_true_utilities: Dict[str, Optional[float]] = {child: None for child in child_feature_data.keys()}
        self.min_raw_utility: float = 8.0
        self.max_raw_utility: float = 40.0
        self.day: int = 1
        self.week: int = 1
        self.menu_plan_length = menu_plan_length
        self.save_to_json = save_to_json
        
        self.daily_gini_coefficients: Dict[int, Dict[str, float]] = {}
        self.cumulative_gini_coefficients: List[Dict[str, float]] = []

        # Ensure the file path exists
        if not os.path.exists(os.path.dirname(self.save_to_json)):
            os.makedirs(os.path.dirname(self.save_to_json))
        
        # Initialize the generated count
        self.generated_count = 0

    @staticmethod
    def _convert_true_preferences_to_dict(true_preferences: Dict[str, Dict[str, Dict[str, List[str]]]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Convert the true preferences list to a dictionary for easier access.
        """
        preferences = {}
        for child in true_preferences.keys():
            preferences[child] = {
                "likes": true_preferences[child]['known']["likes"] + true_preferences[child]['unknown']["likes"],
                "dislikes": true_preferences[child]['known']["dislikes"] + true_preferences[child]['unknown']["dislikes"],
                "neutral": true_preferences[child]['known']["neutral"] + true_preferences[child]['unknown']["neutral"],
            }
        return preferences
    
    def get_ingredient_util(self, preferences: Dict[str, Dict[str, List[str]]], 
                            child: str, ingredient: str) -> int:
        """
        Get the current preference list (likes, dislikes, neutral) or raise an error if unknown.
        """
        if ingredient in preferences[child]["likes"]:
            return 5
        elif ingredient in preferences[child]["dislikes"]:
            return -5
        elif ingredient in preferences[child]["neutral"]:
            return 2
        else: 
            raise ValueError(f"Ingredient {ingredient} not found in preferences for child {child}")

    def get_scaling_factor(self, previous_day_utility: Optional[float]) -> float:
        """
        Calculate the scaling factor based on the previous day's utility.
        """
        if previous_day_utility is None:
            return 1.0  # No penalty or bonus for the first day
        if previous_day_utility <= self.min_raw_utility:
            return 1  # Full penalty
        elif previous_day_utility >= self.max_raw_utility:
            return 1  # Full bonus
        else:
            return 1
            # # Linear scaling between self.min_raw_utility and self.max_raw_utility
            # return 0.5 + ((previous_day_utility - self.min_raw_utility) / 
            #               (self.max_raw_utility - self.min_raw_utility)) * 1.5

    def calculate_day_menu_utility(self, predicted_preferences: Dict[str, Dict[str, List[str]]], 
                                   menu_plan: list) -> Dict[str, float]:
        """
        Calculate the utility of the day's menu for each child.
        """
        logging.info(f"Calculating utilities for meal plan #{self.generated_count + 1}: {menu_plan}")
        
        if self.generated_count >= self.menu_plan_length:
            logging.info(f"Reached {self.menu_plan_length} meal plans. Resetting utilities and saving data.")
            self.reset_utilities_and_save()
            self.generated_count = 0  # Reset the count after saving
            
        self.raw_utility[self.day] = {child: 0 for child in self.child_feature_data.keys()}
        self.true_raw_utility[self.day] = {child: 0 for child in self.child_feature_data.keys()}
        self.predicted_raw_utility[self.day] = {child: 0 for child in self.child_feature_data.keys()}
        self.true_utility[self.day] = {child: 0 for child in self.child_feature_data.keys()}
        self.predicted_utility[self.day] = {child: 0 for child in self.child_feature_data.keys()}

        for child in self.child_feature_data.keys():
            for ingredient in menu_plan:
                self.true_raw_utility[self.day][child] += self.get_ingredient_util(self.true_preferences, child, ingredient)
                self.predicted_raw_utility[self.day][child] += self.get_ingredient_util(predicted_preferences, child, ingredient)
            
            # Apply scaling factor based on the previous day's utility
            previous_day_predicted_utilities = self.previous_day_predicted_utilities.get(child)
            previous_day_true_utilities = self.previous_day_true_utilities.get(child)
            predicted_scaling_factor = self.get_scaling_factor(previous_day_predicted_utilities)
            true_scaling_factor = self.get_scaling_factor(previous_day_true_utilities)
            self.predicted_utility[self.day][child] = self.predicted_raw_utility[self.day][child] * predicted_scaling_factor
            self.true_utility[self.day][child] = self.true_raw_utility[self.day][child] * true_scaling_factor

        # Calculate daily Gini coefficients
        self.daily_gini_coefficients[self.day] = {
            "true_gini": self.calculate_gini_coefficient(self.true_utility[self.day]),
            "predicted_gini": self.calculate_gini_coefficient(self.predicted_utility[self.day])
        }

        logging.info(f"Daily Gini coefficients for day {self.day}: {self.daily_gini_coefficients[self.day]}")

        # Update previous day's utilities
        for child in self.child_feature_data.keys():
            self.previous_day_predicted_utilities[child] = self.predicted_raw_utility[self.day][child]
            self.previous_day_true_utilities[child] = self.true_raw_utility[self.day][child]
            
        self.day += 1  # Move to the next day after calculation
        self.generated_count += 1
        
        return self.predicted_utility[self.day - 1]  # Return the predicted utility for the current day (before increment)

    def reset_utilities_and_save(self) -> None:
        """
        Save the current utilities and Gini coefficients to a JSON file and reset them.
        """
        cumulative_true_utilities = self.sum_and_sort_utilities_across_days(self.true_utility)
        cumulative_predicted_utilities = self.sum_and_sort_utilities_across_days(self.predicted_utility)

        cumulative_gini = {
            "true_cumulative_gini": self.calculate_gini_coefficient(cumulative_true_utilities),
            "predicted_cumulative_gini": self.calculate_gini_coefficient(cumulative_predicted_utilities)
        }
        self.cumulative_gini_coefficients.append(cumulative_gini)

        # Calculate the sum of utilities
        sum_true_raw_utility = {day: sum(utilities.values()) for day, utilities in self.true_raw_utility.items()}
        sum_predicted_raw_utility = {day: sum(utilities.values()) for day, utilities in self.predicted_raw_utility.items()}
        sum_true_utility = {day: sum(utilities.values()) for day, utilities in self.true_utility.items()}
        sum_predicted_utility = {day: sum(utilities.values()) for day, utilities in self.predicted_utility.items()}

        data_to_save = {
            "week": self.week,
            "true_raw_utility": self.true_raw_utility,
            "predicted_raw_utility": self.predicted_raw_utility,
            "true_utility": self.true_utility,
            "predicted_utility": self.predicted_utility,
            "sum_true_raw_utility": sum_true_raw_utility,
            "sum_predicted_raw_utility": sum_predicted_raw_utility,
            "sum_true_utility": sum_true_utility,
            "sum_predicted_utility": sum_predicted_utility,
            "daily_gini_coefficients": self.daily_gini_coefficients,
            "cumulative_gini_coefficients": self.cumulative_gini_coefficients
        }

        if os.path.exists(self.save_to_json):
            with open(self.save_to_json, 'r+') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []

                existing_data.append(data_to_save)
                json_file.seek(0)
                json.dump(existing_data, json_file, indent=4)
                json_file.truncate()
        else:
            with open(self.save_to_json, 'w') as json_file:
                json.dump([data_to_save], json_file, indent=4)

        logging.info("Utilities and Gini coefficients have been saved and reset.")

        # Increment the week number
        self.week += 1

        # Reset utilities and Gini coefficients
        self.true_raw_utility = {}
        self.predicted_raw_utility = {}
        self.true_utility = {}
        self.predicted_utility = {}
        self.daily_gini_coefficients = {}

        # Restore previous day utilities
        self.previous_day_predicted_utilities = {child: self.previous_day_predicted_utilities[child] for child in self.child_feature_data.keys()}
        self.previous_day_true_utilities = {child: self.previous_day_true_utilities[child] for child in self.child_feature_data.keys()}
        
        return data_to_save
    
    def sum_and_sort_utilities_across_days(self, utilities: Dict[int, Dict[str, float]]) -> List[Tuple[str, float]]:
        """
        Sum utilities across days and sort them.
        """
        cumulative_utilities: Dict[str, float] = {}

        for day_utility in utilities.values():
            for child, util in day_utility.items():
                if child not in cumulative_utilities:
                    cumulative_utilities[child] = 0
                cumulative_utilities[child] += util

        # Sort by utility value in descending order
        sorted_utilities = sorted(cumulative_utilities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_utilities

    def calculate_gini_coefficient(self, utilities: Dict[str, float]) -> float:
        """
        Calculate the Gini coefficient from the sorted utilities.
        """
        
        if isinstance(utilities, list):
            values = []
            for util in utilities:
                values.append(util[1])
        else:
            values = list(utilities.values())
        
        n = len(values)
        if n == 0:
            return 0
        
        sorted_values = sorted(values)
        gini_numerator = sum((i + 1) * sorted_values[i] for i in range(n))
        gini_denominator = n * sum(sorted_values)
        
        gini_coefficient = (2 * gini_numerator) / gini_denominator - (n + 1) / n
        return gini_coefficient

    def close(self) -> Dict[str, Any]:
        """
        Close the calculator, saving the data to a JSON file if specified.
        Returns utilities.
        """
        data_to_save = self.reset_utilities_and_save()

        return data_to_save
