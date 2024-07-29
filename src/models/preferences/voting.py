from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Callable, Union
import pandas as pd
import json 

class IngredientNegotiator:
    def __init__(self, seed: int, ingredient_df: pd.DataFrame, preferences: Dict[str, Dict[str, List[str]]], 
                 previous_feedback: Dict[str, Union[int, float]], previous_utility: Dict[str, Union[int, float]]):
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
        self.ingredient_groups = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E', 'Bread', 'Confectionary']
        self.supplier_availability = self.get_supplier_availability()
        self.vote_gini_dict = {}
        
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
    def create_preference_score_function(votes: Dict[str, Dict[str, float]]) -> Callable[[str], float]:
        """
        Create a preference score function based on votes.

        :param votes: Dictionary of votes.
        :return: Function that returns the score for a given ingredient.
        """
        ingredient_scores = {}
        if votes:
            for group, ingredients in votes.items():
                scores = np.array(list(ingredients.values())).reshape(-1, 1)
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(scores).flatten()
                for ingredient, norm_score in zip(ingredients.keys(), normalized_scores):
                    ingredient_scores[ingredient] = norm_score

        def score_function(ingredient: str) -> float:
            return ingredient_scores.get(ingredient, 0)

        return score_function

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
                        votes[ingredient] += 3 * weights[child]['neutral']
                    elif ingredient in dislikes:
                        votes[ingredient] += 1 * weights[child]['dislikes']
                else:
                    unavailable_ingredients.add(ingredient)

        return votes, unavailable_ingredients
    
    def negotiate_ingredients(self, weight_function: str) -> Tuple[Dict[str, Dict[str, int]], set]:
        """
        Negotiate ingredients based on the specified weight function.

        Parameters:
        weight_function (str): The weight function to use for negotiation. 
                            It can be either 'simple' or 'complex'.

        Returns:
        Tuple[Dict[str, Dict[str, int]], set]: 
            - negotiated (Dict[str, Dict[str, int]]): A dictionary containing the negotiated ingredients.
            - unavailable (set): A set of ingredients that are unavailable.
        """
        
        if weight_function == "simple":
            # Use the simple negotiation method
            negotiated, unavailable = self.negotiate_ingredients_simple()
        elif weight_function == "complex":
            # Use the complex negotiation method
            negotiated, unavailable = self.negotiate_ingredients_complex()
        else:
            # Raise an error if the weight function is invalid
            raise ValueError("Invalid weight function")

        return negotiated, unavailable

    
    def negotiate_ingredients_simple(self) -> Tuple[Dict[str, Dict[str, int]], set]:
        """
        Negotiate ingredients using a simple weight calculation.

        :return: Tuple containing negotiated ingredients and a set of unavailable ingredients.
        """
        votes, unavailable_ingredients = self.collect_weighted_votes(self.calculate_child_weight_simple)
        negotiated_ingredients = {group: {} for group in self.ingredient_groups}
        ingredient_to_groups = {ingredient: group for group in self.ingredient_groups 
                                for ingredient in self.ingredient_df[self.ingredient_df[group] == 1]['Category7']}

        for ingredient, vote in votes.items():
            if ingredient not in unavailable_ingredients:
                group = ingredient_to_groups.get(ingredient)
                if group:
                    negotiated_ingredients[group][ingredient] = vote

        misc_df = self.ingredient_df[(self.ingredient_df[self.ingredient_groups].sum(axis=1) == 0)]
        misc_votes = {ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() if ingredient in votes and ingredient not in unavailable_ingredients}
        negotiated_ingredients['Misc'] = misc_votes

        return negotiated_ingredients, unavailable_ingredients

    def calculate_child_weight_simple(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate simple weights for each child's preferences.

        :return: Dictionary of weights for each child.
        """
        return {child: {'likes': 1, 'neutral': 1, 'dislikes': 1} for child in self.preferences.keys()}

    def negotiate_ingredients_complex(self) -> Tuple[Dict[str, Dict[str, int]], set]:
        """
        Negotiate ingredients using a complex weight calculation.

        :return: Tuple containing negotiated ingredients and a set of unavailable ingredients.
        """
        votes, unavailable_ingredients = self.collect_weighted_votes(self.calculate_child_weight_complex)
        negotiated_ingredients = {group: {} for group in self.ingredient_groups}
        ingredient_to_groups = {ingredient: group for group in self.ingredient_groups 
                                for ingredient in self.ingredient_df[self.ingredient_df[group] == 1]['Category7']}

        for ingredient, vote in votes.items():
            if ingredient not in unavailable_ingredients:
                group = ingredient_to_groups.get(ingredient)
                if group:
                    negotiated_ingredients[group][ingredient] = vote

        misc_df = self.ingredient_df[(self.ingredient_df[self.ingredient_groups].sum(axis=1) == 0)]
        misc_votes = {ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() if ingredient in votes and ingredient not in unavailable_ingredients}
        negotiated_ingredients['Misc'] = misc_votes
        
        self.identify_dislikes_in_top_n(negotiated_ingredients, n=10)

        return negotiated_ingredients, unavailable_ingredients


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



    def _normalize_for_preference_count(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights based on the number of preferences for each category.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Normalized weights.
        """
        for category in weights.keys():
            weights[category] = weights[category] / len(self.preferences[child][category]) if len(self.preferences[child][category]) > 0 else 0
        return weights

    def _use_compensatory_weight_update(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using a compensatory factor based on previous utility.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Updated weights.
        """
        if child in self.previous_utility.keys():
            previous_utility = self.previous_utility[child]
            distance_from_avg = previous_utility - self.average_utility
            compensatory_factor = 1 + (1 - abs(distance_from_avg / self.average_utility))
            weights = self._multiply_weights_by_factor(weights, factor=compensatory_factor)
        return weights

    def _use_feedback_weight_update(self, child: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights using feedback.

        :param child: Child identifier.
        :param weights: Dictionary of weights for the child.
        :return: Updated weights.
        """
        if child in self.feedback.keys():
            weights = self._multiply_weights_by_factor(weights, factor=1.1)
        return weights

    def calculate_child_weight_complex(self, use_normalize_weights: bool = True, use_compensatory: bool = True, 
                                       use_feedback: bool = True, use_fairness: bool = True, target_gini: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Calculate complex weights for each child's preferences using various factors.

        :param use_normalize_weights: Whether to normalize weights.
        :param use_compensatory: Whether to use compensatory weight update.
        :param use_feedback: Whether to use feedback weight update.
        :param use_fairness: Whether to use fairness adjustment.
        :param target_gini: Target Gini coefficient for fairness.
        :return: Dictionary of weights for each child.
        """
        raw_weights = self.calculate_child_weight_simple()
        weights = copy.deepcopy(raw_weights)

        for child in self.preferences.keys():
            if use_normalize_weights:
                weights[child] = self._normalize_for_preference_count(child, weights[child])
            if use_compensatory:
                weights[child] = self._use_compensatory_weight_update(child, weights[child])
            if use_feedback:
                weights[child] = self._use_feedback_weight_update(child, weights[child])

        ginis, weight_category_total = self._calculate_all_gini(weights)
        current_gini = ginis['total']

        if use_fairness and current_gini > target_gini:
            updated_weights, final_ginis = self.scaling_to_adjust_weights_for_target_gini(weights, weight_category_total, current_gini, target_gini)
        else:
            updated_weights = weights
            final_ginis = ginis

        self.vote_gini_dict = final_ginis  # Store the Gini coefficients
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
            adjustment_factor = 0.001
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
        child_counter = 0
        for child, weight in weights.items():
            for category in weight.keys():
                weights[child][category] += scaling_factors[child_counter]
            child_counter += 1
        return weights

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

    def get_supplier_availability(self, mean_unavailable: int = 5, std_dev_unavailable: int = 2) -> Dict[str, bool]:
        """
        Generate supplier availability for ingredients.

        :param mean_unavailable: Mean number of unavailable ingredients.
        :param std_dev_unavailable: Standard deviation of unavailable ingredients.
        :return: Dictionary of supplier availability.
        """
        ingredients = self.ingredient_df['Category7'].tolist()
        random.seed(self.seed)
        num_unavailable = max(0, int(np.random.normal(mean_unavailable, std_dev_unavailable)))
        unavailable_ingredients = random.sample(ingredients, num_unavailable)
        supplier_availability = {ingredient: ingredient not in unavailable_ingredients for ingredient in ingredients}
        return supplier_availability
    
    def log_data(self, log_file: str) -> None:
        """
        Log the Gini coefficients and other relevant data to a JSON file.

        :param log_file: Path to the log file.
        """
        data_to_log = {
            'Gini Votes': self.vote_gini_dict,
            # 'Children Dislikes in Top n': self.children_dislikes_in_top_n
        }

        with open(log_file, 'w') as file:
            json.dump(data_to_log, file, indent=4)
        print(f"Data successfully logged to {log_file}")

    def close(self, log_file: str) -> None:
        """
        Save the current state to a file and perform any necessary cleanup.

        :param log_file: Path to the log file.
        """
        self.log_data(log_file)
        # Perform any other necessary cleanup here
        print("Negotiator closed and data saved.")