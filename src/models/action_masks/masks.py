import numpy as np

def convert_unavailable_to_index(unavailable_ingredients, ingredient_df):
    # Get the indexes of rows where 'Category7' is in the list of unavailable ingredients
    indexes = ingredient_df.index[ingredient_df['Category7'].isin(unavailable_ingredients)].tolist()
    return indexes

def mask_fn(self, unavailable=None) -> np.ndarray:
    """
    Generate an action mask indicating valid actions.

    Args:
        unavailable (list): List of unavailable ingredients. If none, all ingredients are available.

    Returns:
        np.ndarray: The action mask indicating valid actions.
    """
    # Cache necessary attributes
    n_ingredients = self.env.get_wrapper_attr('n_ingredients')
    get_metrics = self.env.get_wrapper_attr('get_metrics')
    
    # Ensure metrics are up-to-date
    get_metrics()

    # Get current state attributes
    ingredient_group_count = self.env.get_wrapper_attr('ingredient_group_count')
    ingredient_group_count_targets = self.env.get_wrapper_attr('ingredient_group_count_targets')
    ingredient_group_portion = self.env.get_wrapper_attr('ingredient_group_portion')
    ingredient_group_portion_targets = self.env.get_wrapper_attr('ingredient_group_portion_targets')
    group_info = self.env.get_wrapper_attr('group_info')
    current_selection = self.env.get_wrapper_attr('current_selection')
    ingredient_df = self.env.get_wrapper_attr('ingredient_df')

    extra_action = 2  # First 2 for actions [do nothing, increase], rest for ingredients

    # Initialize action mask with zeros
    action_mask = np.zeros(extra_action + n_ingredients, dtype=np.int8)

    # Determine if all targets are met
    all_group_count_target_met = all(
        ingredient_group_count[key] >= count_target
        for key, count_target in ingredient_group_count_targets.items()
    )

    all_group_portion_target_met = all(
        portion_target[0] <= ingredient_group_portion[key] <= portion_target[1]
        for key, portion_target in ingredient_group_portion_targets.items()
    )
    
    # If the calorie target isn't met one has to increase the ingredients to meet this
    calorie_target_met = self.env.get_wrapper_attr('nutrient_averages')['calories'] >= self.env.get_wrapper_attr('nutrient_target_ranges')['calories'][0]

    target_flag = all_group_count_target_met and all_group_portion_target_met and calorie_target_met

    # If all the group count targets are met, allow selection for all ingredients in that list
    if target_flag:
        selected_indexes = np.where(current_selection > 0)[0]
        for idx in selected_indexes:
            action_mask[idx + extra_action] = 1
        action_mask[:extra_action] = [1, 1]
        
        return action_mask
        
    else:
        # Iterate through each ingredient group to set the action mask
        for key, target in ingredient_group_count_targets.items():
            value = ingredient_group_count[key]
            indexes = group_info[key]['indexes']
            selected = [idx for idx in indexes if current_selection[idx] > 0]
            
            portion_value = ingredient_group_portion[key]
            portion_target = ingredient_group_portion_targets[key]

            if target == 0:
                # If the target is zero, block selection for all these ingredients
                action_mask[indexes + extra_action] = 0
                
            elif value == target:
                # If the group portion limit is met or exceeded block actions to that group
                if portion_value >= portion_target[1]:
                    # Block all actions to that ingredient group
                    action_mask[indexes + extra_action] = 0
                
                # If portion is met for one category but not for all groups block actions to that group until all groups are met
                elif portion_target[0] <= portion_value <= portion_target[1] and not all_group_portion_target_met:
                    # Block all actions to that ingredient group until all group portion targets are met
                    action_mask[indexes + extra_action] = 0
                        
                else:
                    # If the count target is met, but portion is not allow selection for selected ingredient
                    for idx in selected:
                        if idx in indexes:
                            action_mask[idx + extra_action] = 1
                        else:
                            # Block selection for non selected
                            action_mask[idx + extra_action] = 0
                        
            else:
                # If the group count target is not met, allow selection for all ingredients in this group
                action_mask[indexes + extra_action] = 1

    # If unavailable ingredients are specified, block selection for these ingredients
    action_mask = block_unavailable_ingredients(unavailable, ingredient_df, action_mask, extra_action)
       
    # Apply any additional action constraints
    action_mask = ingredient_action(self, ingredient_df, all_group_portion_target_met, calorie_target_met, action_mask, extra_action)

    return action_mask

def block_unavailable_ingredients(unavailable, ingredient_df, action_mask, extra_action):
    """
    Blocks the selection of unavailable ingredients by updating the action mask.

    Parameters:
    unavailable (list): List of unavailable ingredient names.
    ingredient_df (DataFrame): DataFrame containing ingredient information.
    action_mask (list): List representing the action mask for ingredient selection.
    extra_action (int): Offset to apply to the indices in the action mask.
    
    Returns:
    action_mask
    """
    if not unavailable:
        return

    try:
        # Convert unavailable ingredients to their respective indexes
        unavailable_indexes = convert_unavailable_to_index(unavailable, ingredient_df)

        # Update the action mask to block these indexes
        for idx in unavailable_indexes:
            action_mask[idx + extra_action] = 0

        return action_mask
    except Exception as e:
        print(f"Error blocking unavailable ingredients: {e}")

def ingredient_action(self, ingredient_df, all_group_portion_target_met, calorie_target_met, action_mask, extra_action):
    """
    Update action mask based on whether count and portion targets for all groups are met.

    Args:
        self: Reference to the environment wrapper.
        all_group_portion_target_met (bool): Indicator if all targets are met.
        action_mask (np.ndarray): Current action mask.
        extra_action (int): Number of extra actions (do nothing, increase).

    Returns:
        np.ndarray: Updated action mask.
    """
        
    if all_group_portion_target_met and calorie_target_met:
        # Allow all actions if both targets are met, including the do nothing signal
        action_mask[:extra_action] = [1, 1]
    else:
        # If the targets are not met, only allow the increase action
        action_mask[:extra_action] = [0, 1]

    verbose = self.env.get_wrapper_attr('verbose')
    if verbose > 1:
        print_action_mask(action_mask, ingredient_df, extra_action)

    return action_mask

def print_action_mask(action_mask, ingredient_df, extra_action):
    """
    Print the action mask with indications of allowed actions.

    Args:
        action_mask (np.ndarray): Current action mask.
        ingredient_df (pd.DataFrame): DataFrame of ingredient information.
        extra_action (int): Number of extra actions (do nothing, increase).
    """
    action_list = ['do nothing', 'increase']
    print_action_list = [f"{action} (allowed)" if action_mask[i] == 1 else f"{action} (disallowed)" for i, action in enumerate(action_list)]

    # Print allowed actions
    for action in print_action_list:
        print(action)

    # Get non-zero indices for ingredient actions
    nonzero_indices = np.nonzero(action_mask[extra_action:])[0]
    print(f"To selected ingredients:")
    for idx in nonzero_indices:
        category_value = ingredient_df['Category7'].iloc[idx]
        print(f"{category_value}")
