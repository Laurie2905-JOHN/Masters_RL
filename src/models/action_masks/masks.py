import numpy as np

def mask_fn1(self) -> np.ndarray:
    """
    Generate an action mask indicating valid actions.
    """
    
    # Cache necessary attributes
    n_ingredients = self.env.get_wrapper_attr('n_ingredients')  # Number of ingredients
    nsteps = self.env.get_wrapper_attr('nsteps')  # Number of steps taken so far
    self.get_metrics()  # Ensure metrics are up-to-date
    
    # Get current state attributes
    ingredient_group_count = self.env.get_wrapper_attr('ingredient_group_count')  # Current count of each ingredient group
    ingredient_group_count_targets = self.env.get_wrapper_attr('ingredient_group_count_targets')  # Target count for each ingredient group
    group_info = self.env.get_wrapper_attr('group_info')  # Information about each group
    current_selection = self.env.get_wrapper_attr('current_selection')  # Current ingredient selection
    
    # Initialize action mask with zeros
    action_mask = np.zeros(3 + n_ingredients, dtype=np.int8)  # First 3 for actions, rest for ingredients
    
    # Determine if all group targets have been met
    all_group_target_met = all(
        ingredient_group_count[key] == target
        for key, target in ingredient_group_count_targets.items()
    )
    
    # Iterate through each ingredient group to set the action mask
    for key, target in ingredient_group_count_targets.items():
        value = ingredient_group_count[key]  # Current count of the ingredient group
        indexes = group_info[key]['indexes']  # Indexes of ingredients in this group
        selected = [idx for idx in indexes if current_selection[idx] > 0]  # Selected ingredients in this group
        
        if target == 0:
            # If the target is zero, block selection for all these ingredients
            action_mask[indexes + 3] = 0
        elif value >= target:
            # If the group target is met or exceeded
            if all_group_target_met:
                # Allow only selected ingredients to be acted upon in this group if all the group targets are met
                for idx in indexes:
                    action_mask[idx + 3] = 1 if idx in selected else 0
            else:
                # Block selection for all ingredients in this group until all group targets are met
                action_mask[indexes + 3] = 0
        else:
            # If the group target is not met, allow selection for all ingredients in this group
            action_mask[indexes + 3] = 1
    
    # ALlow all actions to be taken on an ingredient if all group targets are met and nsteps is greater than 25 [zero, decrease, increase]
    if all_group_target_met and nsteps > 25:
        action_mask[:3] = [1, 1, 1]  # Allow all actions if all group targets are met and steps > 25
    else:
        # If group targets are not met, only allow the increase action as only ingredients which are needed for group target will be selected
        action_mask[:3] = [0, 0, 1]  # Only allow the "increase" action otherwise
        
    return action_mask


