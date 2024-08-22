
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Callable, Union, Set
import pandas as pd
import json 
import os 

class IngredientNegotiator:
    def __init__(self, seed: int, ingredient_df: pd.DataFrame, preferences: Dict[str, Dict[str, List[str]]], complex_weight_func_args: Dict[str, bool] = {},
                 previous_feedback: Dict[str, Union[int, float]] = {}, previous_utility: Dict[str, Union[int, float]]  =  {}):
        """
        Initialize the IngredientNegotiator class.

        :param seed: Random seed for reproducibility.
        :param ingredient_df: DataFrame containing ingredient data.
        :param preferences: Dictionary containing preferences for each child.
        :param previous_feedback: Dictionary containing previous feedback for each child.
        :param previous_fairness_index: Dictionary containing previous fairness index for each child.
        :param previous_utility: Dictionary containing previous utility values for each child.
        """
        self.seed = seed
        self.ingredient_df = ingredient_df
        self.preferences = preferences
        self.feedback = previous_feedback
        self.previous_utility = previous_utility
        self.average_utility = self._calculate_average_utility(previous_utility)
        
        if self.previous_utility:
            # Calculate all compensatory factors for normalization
            all_factors = [self.previous_utility[c] - self.average_utility for c in self.previous_utility.keys()]
            
            # Determine the minimum and maximum compensatory factors
            self.min_factor = min(all_factors)
            self.max_factor = max(all_factors)
        
        # Map ingredients to their respective groups
        self.ingredient_groups = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E', 'Bread', 'Confectionary']
        self.ingredient_to_groups = {
            ingredient: group for group in self.ingredient_groups 
            for ingredient in self.ingredient_df[self.ingredient_df[group] == 1]['Category7']
        }
        
        
        self.supplier_availability = self.get_supplier_availability(mean_unavailable=0, std_dev_unavailable=0)
        self.vote_gini_dict = {}
        
        if complex_weight_func_args == {}:
            raise ValueError("Complex weight function arguments must be provided.")
        
        self.complex_weight_func_args = complex_weight_func_args
        self.children_dislikes_in_top_n = {}
        self.children_dislikes_in_top_n_complex = {}
        self.vote_gini_dict_complex = {}
        
    @staticmethod
    def _calculate_average_utility(previous_utility: Dict[str, Union[int, float]]) -> float:
        """
        Calculate the average utility from previous utility values.

        :param previous_utility: Dictionary of previous utility values.
        :return: Average utility.
        """
        if previous_utility:
            return np.mean(list(previous_utility.values()))
        return 1

    @staticmethod
    def generate_ordered_list(ingredients_list: List[str]) -> Dict[str, int]:
        """
        Generate an ordered list of ingredients with their positions.

        :param ingredients_list: List of ingredients.
        :return: Dictionary of ingredients with their positions.
        """
        return {ingredient: index + 1 for index, ingredient in enumerate(ingredients_list)}

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize the weights so that their sum is 1.

        :param weights: Dictionary of weights.
        :return: Normalized dictionary of weights.
        """
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        return weights

    @staticmethod
    def _multiply_weights_by_factor(weights: Dict[str, float], factor: float) -> Dict[str, float]:
        """
        Multiply weights by a given factor.

        :param weights: Dictionary of weights.
        :param factor: Factor to multiply weights by.
        :return: Updated dictionary of weights.
        """
        for key in weights:
            weights[key] *= factor
        return weights

    def compare_negotiated_ingredients(self, old_ingredients: Dict[str, List[str]], new_ingredients: Dict[str, List[str]], 
                                       old_unavailable: Dict[str, List[str]], new_unavailable: Dict[str, List[str]]) -> None:
        """
        Compare old and new negotiated ingredients and print changes.

        :param old_ingredients: Dictionary of old ingredients.
        :param new_ingredients: Dictionary of new ingredients.
        :param old_unavailable: Dictionary of old unavailable ingredients.
        :param new_unavailable: Dictionary of new unavailable ingredients.
        """
        changes = {}
        all_ingredient_types = set(old_ingredients.keys()).union(new_ingredients.keys()).union(old_unavailable.keys()).union(new_unavailable.keys())

        for ingredient_type in all_ingredient_types:
            old_list = old_ingredients.get(ingredient_type, [])
            new_list = new_ingredients.get(ingredient_type, [])
            old_unavail_list = old_unavailable.get(ingredient_type, [])
            new_unavail_list = new_unavailable.get(ingredient_type, [])

            old_order = self.generate_ordered_list(old_list)
            new_order = self.generate_ordered_list(new_list)

            order_changes = []
            for ingredient in set(old_order.keys()).union(new_order.keys()).union(old_unavail_list).union(new_unavail_list):
                old_pos = old_order.get(ingredient, None)
                new_pos = new_order.get(ingredient, None)

                if ingredient in old_unavail_list:
                    old_pos = 'Unavailable'
                if ingredient in new_unavail_list:
                    new_pos = 'Unavailable'

                if old_pos != new_pos:
                    order_changes.append((ingredient, old_pos, new_pos))

            if order_changes:
                changes[ingredient_type] = order_changes

        for ingredient_type, order_changes in changes.items():
            print(f"Changes in {ingredient_type}:")
            for ingredient, old_pos, new_pos in order_changes:
                print(f"{ingredient}: Pos: {old_pos} -> Pos: {new_pos}")

    def collect_weighted_votes(self, weight_function: Callable[[], Dict[str, Dict[str, float]]]) -> Tuple[Dict[str, int], set]:
        """
        Collect weighted votes for ingredients based on preferences.

        :param weight_function: Function to calculate weights.
        :return: Tuple containing votes and a set of unavailable ingredients.
        """
        votes = {ingredient: 0 for ingredient in self.supplier_availability}
        unavailable_ingredients = set()
        weights = weight_function()

        for child, pref in self.preferences.items():
            likes = set(pref["likes"])
            neutrals = set(pref["neutral"])
            dislikes = set(pref["dislikes"])

            for ingredient in likes | neutrals | dislikes:
                if sum([ingredient in likes, ingredient in neutrals, ingredient in dislikes]) > 1:
                    raise ValueError(f"Ingredient {ingredient} is found in multiple categories")

                if self.supplier_availability.get(ingredient, 0) > 0:
                    if ingredient in likes:
                        votes[ingredient] += 5 * weights[child]['likes']
                    elif ingredient in neutrals:
                        votes[ingredient] += 1 * weights[child]['neutral']
                    elif ingredient in dislikes:
                        votes[ingredient] += -5 * weights[child]['dislikes']
                else:
                    unavailable_ingredients.add(ingredient)

        return votes, unavailable_ingredients

    # Function to populate negotiated ingredients based on the votes
    def populate_negotiated_ingredients(self, votes, negotiated_ingredients, unavailable_ingredients):
        
        for ingredient, vote in votes.items():
            if ingredient not in unavailable_ingredients:
                group = self.ingredient_to_groups.get(ingredient)
                if group:
                    negotiated_ingredients[group][ingredient] = vote
        
        # Handle miscellaneous ingredients
        misc_df = self.ingredient_df[(self.ingredient_df[self.ingredient_groups].sum(axis=1) == 0)]
        misc_votes = {
            ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() 
            if ingredient in votes and ingredient not in unavailable_ingredients
        }
        negotiated_ingredients['Misc'] = misc_votes

    def calculate_child_weight_simple(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate simple weights for each child's preferences.

        :return: Dictionary of weights for each child.
        """
        return {child: {'likes': 1, 'neutral': 1, 'dislikes': 1} for child in self.preferences.keys()}

    def negotiate_ingredients(self, simple_only = True) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Set[str]]:
        """
        Negotiate ingredients using both simple and complex child weight calculations.

        :return: Tuple containing two dictionaries of negotiated ingredients (for simple and complex votes) 
                 and a set of unavailable ingredients.
        """
        
        # Collect votes using both simple and complex strategies
        votes_simple, unavailable_ingredients = self.collect_weighted_votes(self.calculate_child_weight_simple)
        if simple_only == False:
            votes_complex, _ = self.collect_weighted_votes(self.calculate_child_weight_complex)
        
        # Initialize negotiated ingredients for both voting strategies
        negotiated_ingredients_simple = {group: {} for group in self.ingredient_groups}
        negotiated_ingredients_complex = {group: {} for group in self.ingredient_groups}
    
        # Populate negotiated ingredients for both simple and complex votes
        self.populate_negotiated_ingredients(votes_simple, negotiated_ingredients_simple, unavailable_ingredients)
        if simple_only == False:
            self.populate_negotiated_ingredients(votes_complex, negotiated_ingredients_complex, unavailable_ingredients)
        
        # Identify dislikes in top N ingredients for both sets
        self.children_dislikes_in_top_n_simple = self.identify_dislikes_in_top_n(negotiated_ingredients_simple, n=10)
        if simple_only == False:
            self.children_dislikes_in_top_n_complex = self.identify_dislikes_in_top_n(negotiated_ingredients_complex, n=10)
        
        # Calculate and store Gini coefficients for the simple method
        ginis_simple, _ = self._calculate_all_gini(self.calculate_child_weight_simple())
        self.vote_gini_dict_simple = ginis_simple
        if simple_only == True:
            self.vote_gini_dict_simple = None
            
        # Return both sets of negotiated ingredients and the set of unavailable ingredients
        return negotiated_ingredients_simple, negotiated_ingredients_complex, unavailable_ingredients


    def identify_dislikes_in_top_n(self, negotiated_ingredients: Dict[str, Dict[str, int]], n: int = 5) -> Dict[str, Dict[str, List[str]]]:
        """
        Identify which children have all their dislikes in the top n ingredients for each group.

        :param negotiated_ingredients: Updated list of negotiated ingredients.
        :param n: Number of top ingredients to consider.
        :return: Dictionary with children and the groups they have all dislikes in the top n ingredients.
        """
        # Dictionary to store the results
        children_dislikes_in_top_n = {}

        # Iterate over each group to process the ingredients
        for group, group_votes in negotiated_ingredients.items():
            # Skip Misc, Confectionary, and Bread groups as either too few ingredients in the database or not relevant
            if group in ['Misc', 'Confectionary', 'Bread']:
                continue

            # Sort ingredients in the group by their votes in descending order
            sorted_ingredients = sorted(group_votes.items(), key=lambda item: item[1], reverse=True)
            top_n_ingredients = sorted_ingredients[:n]

            # Check each child's preferences
            for child, pref in self.preferences.items():
                if child not in children_dislikes_in_top_n:
                    children_dislikes_in_top_n[child] = {}
                    
                # Find the dislikes that are in the top n ingredients
                dislikes_in_top_n = set(pref['dislikes']).intersection(dict(top_n_ingredients).keys())

                # If all top n ingredients in this group are in the child's dislikes, record this information
                if len(dislikes_in_top_n) == n:
                    children_dislikes_in_top_n[child][group] = {'number': 'ALL', 'ingredients': list(dislikes_in_top_n)}
                else:
                    children_dislikes_in_top_n[child][group] = {'number': len(dislikes_in_top_n), 'ingredients': list(dislikes_in_top_n)}
          
                    
        self.children_dislikes_in_top_n = children_dislikes_in_top_n



    def _normalize_for_vote_categories(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights based on the number of preferences for each category.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Normalized weights.
        """
        for category in weights.keys():
            weights[category] = weights[category] / len(self.preferences[child][category]) if len(self.preferences[child][category]) > 0 else 0
        return weights
    
    def _normalize_for_total_preference_count(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights based on the total number of preferences.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Normalized weights.
        """
        total_preferences = sum([len(pref) for pref in self.preferences[child].values()])
        for category in weights.keys():
            weights[category] = weights[category] / total_preferences if total_preferences > 0 else 0
        return weights

    def _use_compensatory_weight_update(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using a compensatory factor based on previous utility, normalized between 0 and 2,
        where the most negative factor gets 2 and the most positive gets 0.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Updated weights.
        """
        if child in self.previous_utility.keys():
            previous_utility = self.previous_utility[child]
            distance_from_avg = previous_utility - self.average_utility
            compensatory_factor = distance_from_avg 

            # Normalize factors between 0 and 2, where the highest factor becomes 0 and the lowest becomes 2
            if self.max_factor != self.min_factor:  # Avoid division by zero
                normalized_factor = 2 * (self.max_factor - compensatory_factor) / (self.max_factor - self.min_factor)
            else:
                normalized_factor = 1  # If all factors are the same, apply no change

            # Apply the normalized compensatory factor to the weights
            weights = self._multiply_weights_by_factor(weights, factor=normalized_factor)

        return weights


    def _use_feedback_weight_update(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using feedback.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Updated weights.
        """
        if child in self.feedback.keys():
            weights = self._multiply_weights_by_factor(weights, factor=2)
        return weights

    def calculate_child_weight_complex(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate complex weights for each child's preferences using various factors.
        
        :return: Dictionary of weights for each child.
        """
        
        use_compensatory = self.complex_weight_func_args['use_compensatory']
        use_feedback = self.complex_weight_func_args['use_feedback']
        use_fairness = self.complex_weight_func_args['use_fairness']
        target_gini = self.complex_weight_func_args['target_gini']
        
        raw_weights = self.calculate_child_weight_simple()
        weights = copy.deepcopy(raw_weights)

        for child in self.preferences.keys():
            if use_compensatory:
                weights[child] = self._use_compensatory_weight_update(child, weights[child])
            if use_feedback:
                weights[child] = self._use_feedback_weight_update(child, weights[child])

        ginis, weight_category_total = self._calculate_all_gini(weights)
        current_gini = ginis['total']
        print("Current gini", current_gini)
        if use_fairness and current_gini > target_gini:
            updated_weights, final_ginis = self.scaling_to_adjust_weights_for_target_gini(weights, weight_category_total, current_gini, target_gini)
            print(final_ginis)
        else:
            updated_weights = weights
            final_ginis = ginis

        self.vote_gini_dict_complex = final_ginis  # Store the Gini coefficients for the complex method
        return updated_weights

    def scaling_to_adjust_weights_for_target_gini(self, weights: Dict[str, Dict[str, float]], weight_category_total: List[float], 
                                                  current_gini: float, target_gini: float) -> Dict[str, Dict[str, float]]:
        """
        Adjust weights to achieve the target Gini coefficient.

        :param weights: Dictionary of weights.
        :param weight_category_total: List of total weights for each category.
        :param current_gini: Current Gini coefficient.
        :param target_gini: Target Gini coefficient.
        :return: Updated weights dictionary.
        """
        mean_weight = np.mean(weight_category_total)

        def adjust(weight_category_total: List[float], current_gini: float, target_gini: float) -> List[float]:
            original_weights = weight_category_total.copy()
            adjustment_factor = 0.01
            iteration = 0

            while current_gini > target_gini and iteration < 10000:
                iteration += 1
                for i in range(len(weight_category_total)):
                    if weight_category_total[i] < mean_weight:
                        weight_category_total[i] += adjustment_factor * (mean_weight - weight_category_total[i]) / mean_weight
                    else:
                        weight_category_total[i] -= adjustment_factor * (weight_category_total[i] - mean_weight) / mean_weight

                current_gini = self._calculate_gini(weight_category_total)

            scaling_factors = [new / old if old != 0 else 1 for new, old in zip(weight_category_total, original_weights)]
            return scaling_factors

        scaling_factors = adjust(weight_category_total, current_gini, target_gini)
        updated_weights = self._change_original_weight_dict(weights, scaling_factors)
        final_ginis, _ = self._calculate_all_gini(updated_weights)
        
        return updated_weights, final_ginis

    @staticmethod
    def _change_original_weight_dict(weights: Dict[str, Dict[str, float]], scaling_factors: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Change the original weight dictionary based on scaling factors.

        :param weights: Dictionary of weights.
        :param scaling_factors: List of scaling factors.
        :return: Updated weights dictionary.
        """
        # Create a deep copy of the weights dictionary
        updated_weights = copy.deepcopy(weights)
        
        child_counter = 0
        for child, weight in updated_weights.items():
            for category in weight.keys():
                updated_weights[child][category] *= scaling_factors[child_counter]
            child_counter += 1
            
        return updated_weights

    @staticmethod
    def _calculate_gini(weights_category: List[float]) -> float:
        """
        Calculate the Gini coefficient for a list of weights.

        :param weights_category: List of weights.
        :return: Gini coefficient.
        """
        mean_weight = sum(weights_category) / len(weights_category)
        differences_sum = sum(abs(i - j) for i in weights_category for j in weights_category)
        gini = differences_sum / (2 * len(weights_category) ** 2 * mean_weight)
        return gini

    def _calculate_all_gini(self, weights: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], List[float]]:
        """
        Calculate Gini coefficients for all categories.

        :param weights: Dictionary of weights.
        :return: Tuple containing Gini coefficients and total weights.
        """
        categories = {'likes': [], 'neutral': [], 'dislikes': [], 'total': []}

        for value in weights.values():
            categories['likes'].append(value['likes'])
            categories['neutral'].append(value['neutral'])
            categories['dislikes'].append(value['dislikes'])
            categories['total'].append(value['likes'] + value['neutral'] + value['dislikes'])

        ginis = {category: self._calculate_gini(values) for category, values in categories.items()}
        return ginis, categories['total']

    def get_supplier_availability(self, mean_unavailable: int = 10, std_dev_unavailable: int = 5) -> Dict[str, bool]:
        """
        Generate supplier availability for ingredients.

        Ensures that no group is left without any available ingredients.

        :param mean_unavailable: Mean number of unavailable ingredients.
        :param std_dev_unavailable: Standard deviation of unavailable ingredients.
        :return: Dictionary of supplier availability.
        """
        ingredients = self.ingredient_df['Category7'].tolist()
        random.seed(self.seed)
        
        while True:
            num_unavailable = max(0, int(np.random.normal(mean_unavailable, std_dev_unavailable)))
            unavailable_ingredients = random.sample(ingredients, num_unavailable)
            supplier_availability = {ingredient: ingredient not in unavailable_ingredients for ingredient in ingredients}
            
            # Track available ingredients in each group
            groups_availability = {}
            for ingredient, group in self.ingredient_to_groups.items():
                if group not in groups_availability:
                    groups_availability[group] = 0
                if supplier_availability[ingredient]:
                    groups_availability[group] += 1
            
            # Check if any group has all ingredients unavailable
            all_groups_have_available_ingredients = all(count > 0 for count in groups_availability.values())

            # If all groups have at least one available ingredient, break the loop
            if all_groups_have_available_ingredients:
                break

        return supplier_availability
    
    def log_data(self, log_file: str, week: int, day: int) -> None:
        """
        Log the Gini coefficients and other relevant data for both simple and complex methods to a JSON file.

        :param log_file: Path to the log file.
        :param week: Week number.
        :param day: Day number.
        """
        data_to_log = {
            'Week': week,
            'Day': day,
            'Simple Method': {
                'Gini Votes': self.vote_gini_dict_simple,
                'Children Dislikes in Top n': self.children_dislikes_in_top_n_simple
            },
            'Complex Method': {
                'Gini Votes': self.vote_gini_dict_complex,
                'Children Dislikes in Top n': self.children_dislikes_in_top_n_complex
            }
        }

        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        else:
            existing_data = []

        existing_data.append(data_to_log)

        with open(log_file, 'w') as file:
            json.dump(existing_data, file, indent=4)
        print(f"Data successfully logged to {log_file}")

        
    def close(self, log_file: str, week: int, day: int) -> None:
        """
        Save the current state for both simple and complex methods to a file and perform any necessary cleanup.

        :param log_file: Path to the log file.
        :param week: Week number.
        :param day: Day number.
        """
        self.log_data(log_file, week, day)
        print("Negotiator closed and data saved.")
