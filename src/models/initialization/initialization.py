import random
import numpy as np

class IngredientSelectionInitializer:
    
    @staticmethod
    def initialize_selection(main_class):
        if main_class.initialization_strategy == 'perfect':
            return IngredientSelectionInitializer._initialize_current_selection_perfect(main_class)
        elif main_class.initialization_strategy == 'zero':
            return IngredientSelectionInitializer._initialize_current_selection_zero(main_class)
        elif main_class.initialization_strategy == 'probabilistic':
            return IngredientSelectionInitializer._initialize_current_selection_probabilistic(main_class)
        else:
            raise ValueError(f"Invalid value for initialization strategy: {main_class.initialization_strategy}")

    @staticmethod
    def _initialize_current_selection_perfect(main_class):
        rng = random.Random(None)
        selected_indices = []
        ingredient_group_count_targets = main_class.ingredient_group_count_targets
        max_ingredients = main_class.max_ingredients
        n_ingredients = main_class.n_ingredients
        all_indices = main_class.all_indices
        group_info = main_class.group_info

        selected_counts = {category: 0 for category in ingredient_group_count_targets}
        num_indices_to_select = min(max_ingredients, n_ingredients)
        total_target_count = sum(ingredient_group_count_targets.values())
        
        if total_target_count > num_indices_to_select:
            raise ValueError(f"Total target counts {total_target_count} exceed max ingredients {num_indices_to_select}")
        
        for key, value in ingredient_group_count_targets.items():
            for _ in range(value):
                selected_index = rng.choice(group_info[key]['indexes'])
                selected_indices.append(selected_index)
                selected_counts[key] += 1
                
        if num_indices_to_select < max_ingredients:
            remaining_indices = set(all_indices) - set(selected_indices)
            selected_indices.extend(rng.sample(remaining_indices, num_indices_to_select - total_target_count))
        
        current_selection = IngredientSelectionInitializer._assign_values_to_selected_indices(main_class, selected_indices, selected_counts)
        
        if main_class.verbose > 1:
            IngredientSelectionInitializer._print_counts(selected_counts)
            
        return current_selection

    @staticmethod
    def _initialize_current_selection_zero(main_class):
        current_selection = np.zeros(main_class.n_ingredients)
        ingredient_group_count_targets = main_class.ingredient_group_count_targets
        selected_counts = {category: 0 for category in ingredient_group_count_targets}
        
        if main_class.verbose > 1:
            IngredientSelectionInitializer._print_counts(selected_counts)
            
        return current_selection

    @staticmethod
    def _initialize_current_selection_probabilistic(main_class):
        rng = random.Random(None)
        max_ingredients = main_class.max_ingredients
        n_ingredients = main_class.n_ingredients
        all_indices = main_class.all_indices
        verbose = main_class.verbose
        ingredient_group_count_targets = main_class.ingredient_group_count_targets
        group_info = main_class.group_info
        
        selected_indices = rng.sample(all_indices, min(max_ingredients, n_ingredients))
        
        if verbose > 0 and len(selected_indices) > max_ingredients:
            raise ValueError(f"Selected indices: {selected_indices} exceed max ingredients {max_ingredients}")
        
        selected_counts = {category: 0 for category in ingredient_group_count_targets}
        
        for idx in selected_indices:
            for category, info in group_info.items():
                if idx in info['indexes']:
                    selected_counts[category] += 1
                    break
        
        current_selection = IngredientSelectionInitializer._assign_values_to_selected_indices(main_class, selected_indices, selected_counts)
        
        if main_class.verbose > 1:
            IngredientSelectionInitializer._print_counts(selected_counts)
            
        return current_selection

    @staticmethod
    def _assign_values_to_selected_indices(main_class, selected_indices, selected_counts):
        rng = random.Random(None)
        ingredient_group_portion_targets = main_class.ingredient_group_portion_targets
        current_selection = main_class.current_selection
        
        values_to_assign = []
        
        for group, count in selected_counts.items():
            if count > 0:
                for _ in range(count):
                    value = rng.randint(*ingredient_group_portion_targets[group])
                    while value == 0:
                        value = rng.randint(*ingredient_group_portion_targets[group])
                    values_to_assign.append(value)
        
        for idx, value in zip(selected_indices, values_to_assign):
            current_selection[idx] = value
        
        if len(values_to_assign) > main_class.max_ingredients:
            raise ValueError(f"Values to assign greater than max ingredients: {values_to_assign}")

        return current_selection

    def _print_counts(selected_counts):
        for key, value in selected_counts.items():
            print(f"Category: {key}, Count: {value}")