import random
from typing import Dict, List, Set, Optional

class RandomMenuGenerator:
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the RandomMenuGenerator with an optional seed for randomization.

        :param seed: An optional seed for random number generation.
        """
        self.seed = seed
        self.random = random.Random(seed)
        self.generated_count = 1
        # Only remove from these groups as other groups do not have the count number to remove from
        self.groups_to_remove_from = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E']

    def initialize_ingredient_in_groups(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]]) -> None:
        """
        Initializes the ingredients and their scores for each group, excluding unavailable ingredients,
        and normalizes the scores. Also makes a copy of the original ingredients and scores to reset after 10 generations.

        :param negotiated: A dictionary where keys are ingredient groups and values are dictionaries of ingredients with their scores.
        :param unavailable: A set of unavailable ingredients.
        """
        self.negotiated = negotiated
        self.unavailable = unavailable or set()
        self.ingredient_groups = list(negotiated.keys())

        # Initialize dictionaries
        self.ingredients_in_groups = {group: [] for group in self.ingredient_groups}
        self.ingredients_scores = {group: [] for group in self.ingredient_groups}
        self.total_scores = {group: 0 for group in self.ingredient_groups}

        # Populate dictionaries and normalize scores
        for group, items in self.negotiated.items():
            for ingredient, score in items.items():
                if ingredient not in self.unavailable:
                    self.ingredients_in_groups[group].append(ingredient)
                    self.ingredients_scores[group].append(score)
                    self.total_scores[group] += score

        # Normalize scores initially
        self.normalize_scores()

        # Make a copy of the original ingredients list and scores to reset after 10 generations
        self.original_ingredients_in_groups = {group: items.copy() for group, items in self.ingredients_in_groups.items()}
        self.original_ingredients_scores = {group: scores.copy() for group, scores in self.ingredients_scores.items()}
        self.original_total_scores = self.total_scores.copy()

    def normalize_scores(self) -> None:
        """
        Normalizes the scores of the ingredients within each group to ensure they sum up to 1.
        """
        self.normalized_scores = {group: [] for group in self.ingredient_groups}
        for group in self.ingredient_groups:
            total_score = self.total_scores[group]
            if total_score > 0:
                for score in self.ingredients_scores[group]:
                    self.normalized_scores[group].append(score / total_score)
            else:
                if (len(self.ingredients_scores[group]) * len(self.ingredients_scores[group])) == 0:
                    pass
                self.normalized_scores[group] = [1 / len(self.ingredients_scores[group])] * len(self.ingredients_scores[group])

    def generate_random_item(self, group: str, num_items: int = 1) -> List[str]:
        """
        Generates a list of random items from a specified group based on the normalized scores.

        :param group: The ingredient group to sample from.
        :param num_items: The number of items to sample.
        :return: A list of randomly selected ingredients.
        :raises ValueError: If there are not enough ingredients in the group to sample the requested number of items.
        """
        if len(self.ingredients_in_groups[group]) < num_items:
            raise ValueError(f"Not enough ingredients in group {group} to sample {num_items} items")
        items = self.ingredients_in_groups[group]
        weights = self.normalized_scores[group]
        item = self.random.choices(items, weights=weights, k=num_items)
        return item
    
    def generate_random_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, groups_to_remove_from: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generates a random menu by selecting one random item from each ingredient group. If 10 menus have been generated,
        it resets the ingredients.

        :param negotiated: A dictionary where keys are ingredient groups and values are dictionaries of ingredients with their scores.
        :param unavailable: A set of unavailable ingredients.
        :param groups_to_remove_from: List of groups to remove selected items from.
        :return: A dictionary representing the generated menu.
        """
        self.initialize_ingredient_in_groups(negotiated, unavailable)
        
        if self.generated_count >= 11:
            print("\nGenerated 10 meal plans, resetting ingredients.")
            self.reset_ingredients()
        
        if groups_to_remove_from is None:
            groups_to_remove_from = self.groups_to_remove_from
        
        menu = {}
        for group in self.ingredient_groups:
            try:
                item = self.generate_random_item(group, 1)[0]
                menu[group] = item
                # Remove chosen ingredients from the specified groups
                if group in groups_to_remove_from:
                    index = self.ingredients_in_groups[group].index(item)
                    self.total_scores[group] -= self.ingredients_scores[group][index]
                    del self.ingredients_in_groups[group][index]
                    del self.ingredients_scores[group][index]
                    self.normalize_scores()
            except ValueError as e:
                print(f"Error generating item for group {group}: {e}")
                self.reset_ingredients()
                return self.generate_random_menu(negotiated, unavailable, groups_to_remove_from)
        
        print("\nGenerated meal plan number", self.generated_count)
        print(list(menu.values()))
        self.generated_count += 1
        return menu
    
    def reset_ingredients(self) -> None:
        """
        Resets the ingredients and their scores to the original state and normalizes the scores.
        """
        self.ingredients_in_groups = {group: items.copy() for group, items in self.original_ingredients_in_groups.items()}
        self.ingredients_scores = {group: scores.copy() for group, scores in self.original_ingredients_scores.items()}
        self.total_scores = self.original_total_scores.copy()
        self.normalize_scores()
        self.generated_count = 1
