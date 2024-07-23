from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import copy

class IngredientNegotiator:
    def __init__(self, seed, ingredient_df, preferences, previous_feedback, previous_fairness_index, previous_utility):
        
        self.seed = seed
        self.ingredient_df = ingredient_df
        
        self.preferences = preferences
        self.feedback = previous_feedback
        self.fairness_index = previous_fairness_index
        
        self.previous_utility = previous_utility
        
        self.average_utility = self._calculate_average_utility(previous_utility)
        
        self.ingredient_groups = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E', 'Bread', 'Confectionary']
        self.supplier_availability = self.get_supplier_availability()
    
    @staticmethod
    def _calculate_average_utility(previous_utility):
        if previous_utility:
            return np.mean(list(previous_utility.values()))
        else:
            return 1
        
    @staticmethod
    def generate_ordered_list(ingredients_list):
        return {ingredient: index + 1 for index, ingredient in enumerate(ingredients_list)}

    @staticmethod
    def normalize_weights(weights):
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        return weights



    @staticmethod
    def create_preference_score_function(votes):
        if votes:
            ingredient_scores = {}
            for group, ingredients in votes.items():
                scores = np.array(list(ingredients.values())).reshape(-1, 1)
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(scores).flatten()
                for ingredient, norm_score in zip(ingredients.keys(), normalized_scores):
                    ingredient_scores[ingredient] = norm_score
        else:
            ingredient_scores = {}

        def score_function(ingredient):
            return ingredient_scores.get(ingredient, 0)

        return score_function
    
    @staticmethod
    def _multiply_weights_by_factor(weights, factor):
        for key in weights:
            weights[key] *= factor
        return weights

    def compare_negotiated_ingredients(self, old_ingredients, new_ingredients, old_unavailable, new_unavailable):
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

    def collect_weighted_votes(self, weight_function):
        
        votes = {ingredient: 0 for ingredient in self.supplier_availability}
        
        unavailable_ingredients = set()
        
        weights = weight_function()

        for child, pref in self.preferences.items():
            likes = set(pref["likes"])
            neutrals = set(pref["neutral"])
            dislikes = set(pref["dislikes"])

            for ingredient in likes | neutrals | dislikes:
                in_likes = ingredient in likes
                in_neutrals = ingredient in neutrals
                in_dislikes = ingredient in dislikes

                if sum([in_likes, in_neutrals, in_dislikes]) > 1:
                    raise ValueError(f"Ingredient {ingredient} is found in multiple categories")

                if ingredient in self.supplier_availability and self.supplier_availability[ingredient] > 0:
                    if in_likes:
                        votes[ingredient] += 5 * weights['likes']
                    elif in_neutrals:
                        votes[ingredient] += 3 * weights['neutral']
                    elif in_dislikes:
                        votes[ingredient] += 1 * weights['dislikes']
                else:
                    unavailable_ingredients.add(ingredient)

        return votes, unavailable_ingredients

    def negotiate_ingredients_simple(self):
        votes, unavailable_ingredients = self.collect_weighted_votes(self.calculate_child_weight_simple)
        negotiated_ingredients = {group: {} for group in self.ingredient_groups}
        ingredient_to_groups = {ingredient: group for group in self.ingredient_groups for ingredient in self.ingredient_df[self.ingredient_df[group] == 1]['Category7']}

        for ingredient, vote in votes.items():
            if ingredient not in unavailable_ingredients:
                group = ingredient_to_groups.get(ingredient)
                if group:
                    negotiated_ingredients[group][ingredient] = vote

        misc_df = self.ingredient_df[(self.ingredient_df[self.ingredient_groups].sum(axis=1) == 0)]
        misc_votes = {ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() if ingredient in votes and ingredient not in unavailable_ingredients}
        negotiated_ingredients['Misc'] = misc_votes

        return negotiated_ingredients, unavailable_ingredients

    def calculate_child_weight_simple(self):
        return {'likes': 1, 'neutral': 1, 'dislikes': 1}

    def negotiate_ingredients_complex(self):
        
        votes, unavailable_ingredients = self.collect_weighted_votes(self.calculate_child_weight_complex)
        negotiated_ingredients = {group: {} for group in self.ingredient_groups}
        ingredient_to_groups = {ingredient: group for group in self.ingredient_groups for ingredient in self.ingredient_df[self.ingredient_df[group] == 1]['Category7']}

        for ingredient, vote in votes.items():
            if ingredient not in unavailable_ingredients:
                group = ingredient_to_groups.get(ingredient)
                if group:
                    negotiated_ingredients[group][ingredient] = vote

        misc_df = self.ingredient_df[(self.ingredient_df[self.ingredient_groups].sum(axis=1) == 0)]
        misc_votes = {ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() if ingredient in votes and ingredient not in unavailable_ingredients}
        negotiated_ingredients['Misc'] = misc_votes

        for group, group_votes in negotiated_ingredients.items():
            sorted_ingredients = sorted(group_votes.items(), key=lambda item: item[1], reverse=True)
            top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]

            for child, pref in self.preferences.items():
                dislikes_in_top_10 = set(pref['dislikes']).intersection(dict(top_10_percent).keys())
                if len(dislikes_in_top_10) == len(top_10_percent):
                    for dislike in dislikes_in_top_10:
                        group_votes[dislike] -= 1
                        sorted_ingredients = sorted(group_votes.items(), key=lambda item: item[1], reverse=True)
                        top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]

            negotiated_ingredients[group] = dict(sorted_ingredients)

        return negotiated_ingredients, unavailable_ingredients
    
    def _normalize_for_preference_count(self, child, weights):
        
        for key in weights:
            weights[key] = weights[key] / len(self.preferences[child][key]) if len(self.preferences[child][key]) > 0 else 0
        
        return weights

    def _use_compensatory_weight_update(self, child, weights):
        
        if child in self.previous_utility.keys(): 
            previous_utility = self.previous_utility[child]
            distance_from_avg = previous_utility - self.average_utility
            compensatory_factor = 1 + (1 - abs(distance_from_avg / self.average_utility))
            weights = self._multiply_weights_by_factor(weights, factor = compensatory_factor)
            
        return weights
    
    def _use_feedback_weight_update(self, child, weights):
        if child in self.feedback.keys():
            weights = self._multiply_weights_by_factor(weights, factor = 1.1)
        return weights

    # 0.0 - 0.2: High equality
    # 0.2 - 0.3: Moderate equality
    # 0.3 - 0.4: Moderate inequality
    # 0.4 - 0.6: High inequality
    # 0.6 - 1.0: Extreme inequality
            
    def calculate_child_weight_complex(self, use_normalize_weights=True, use_compensatory=True, use_feedback=True, use_fairness=True, target_gini=0.1):
        
        # Get raw weights
        raw_weights = self.calculate_child_weight_simple()
        
        weights = {}
        
        for child in self.preferences.keys():
            
            # Initialize raw weights
            weights[child] = copy.deepcopy(raw_weights)
            
            # Normalize for number of likes, neutral, dislikes so each child within each category has the same voting power 
            if use_normalize_weights:
                weights[child] = self._normalize_for_preference_count(child, weights[child])

            # Use compensation factor to adjust weights if utility was less than the average last round
            if use_compensatory:
                weights[child] = self._use_compensatory_weight_update(child, weights[child])
            
            # If child provided feedback reward for contribution
            if use_feedback:
                weights[child] = self._use_feedback_weight_update(child, weights[child])
                
        # Convert weights dictionary to list for Gini calculation
        weight_values = list(weights.values())

        # Calculate current Gini coefficient
        current_gini = self.calculate_gini(weight_values)

        # Adjust weights to achieve the target Gini coefficient only if current Gini is below target
        if use_fairness and current_gini < target_gini:
            weight_values = self.adjust_weights_for_target_gini(weight_values, target_gini)
            # Update the weights dictionary with adjusted values
            for i, child in enumerate(weights.keys()):
                weights[child] = weight_values[i]

        return weights
    
    def adjust_weights_for_target_gini(self, weights, target_gini):
        mean_weight = np.mean(weights)
        
        def adjust(weights, target_gini):
            adjustment_factor = 0.01  # Small factor to increment/decrement weights
            current_gini = self.calculate_gini(weights)
            iteration = 0

            while current_gini > target_gini and iteration < 10000:  # Limiting to 10000 iterations to avoid infinite loop
                iteration += 1
                for i in range(len(weights)):
                    if weights[i] < mean_weight:
                        weights[i] += adjustment_factor * (mean_weight - weights[i]) / mean_weight
                    else:
                        weights[i] -= adjustment_factor * (weights[i] - mean_weight) / mean_weight

                current_gini = self.calculate_gini(weights)

            return weights

        adjusted_weights = adjust(weights, target_gini)
        return adjusted_weights
    
    @staticmethod
    def calculate_gini(weights):
        mean_weight = sum(weights) / len(weights)
        differences_sum = sum(abs(i - j) for i in weights for j in weights)
        gini = differences_sum / (2 * len(weights) ** 2 * mean_weight)
        return gini
    
    def get_supplier_availability(self, mean_unavailable=5, std_dev_unavailable=2):
        
        ingredients = self.ingredient_df['Category7'].tolist()
        
        # Function to randomly generate supplier availability for ingredients
        random.seed(self.seed)
        
        # Determine the number of unavailable ingredients
        num_unavailable = max(0, int(np.random.normal(mean_unavailable, std_dev_unavailable)))
        unavailable_ingredients = random.sample(ingredients, num_unavailable)
        
        # Generate supplier availability
        supplier_availability = {ingredient: ingredient not in unavailable_ingredients for ingredient in ingredients}
        
        return supplier_availability