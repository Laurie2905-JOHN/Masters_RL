import numpy as np

def mask_fn1(self) -> np.ndarray:
    """
    Generate an action mask indicating valid actions.
    """
    # Cache necessary attributes
    n_ingredients = self.env.get_wrapper_attr('n_ingredients')  # Number of ingredients
    
    get_metrics = self.env.get_wrapper_attr('get_metrics')  # Metrics
    
    step_to_reward = self.env.get_wrapper_attr('step_to_reward')  # Number of ingredients
    
    get_metrics()  # Ensure metrics are up-to-date
    
    # Get current state attributes
    ingredient_group_count = self.env.get_wrapper_attr('ingredient_group_count')  # Current count of each ingredient group
    ingredient_group_count_targets = self.env.get_wrapper_attr('ingredient_group_count_targets')  # Target count for each ingredient group 
    group_info = self.env.get_wrapper_attr('group_info')  # Information about each group
    current_selection = self.env.get_wrapper_attr('current_selection')  # Current ingredient selection
    
    if get_env_name_from_class(self) != 'SchoolMealSelectionDiscreteDone':
        extra_action = 3 # First 3 for actions, rest for ingredients
    else:
        extra_action = 6 # First 6 for actions on ingredients, rest for ingredient selection
    
    # Initialize action mask with zeros
    action_mask = np.zeros(extra_action + n_ingredients, dtype=np.int8)  
    
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
            action_mask[indexes + extra_action] = 0
        elif value == target:
            # If the group target is met or exceeded
            if all_group_target_met:
                # Allow only selected ingredients to be acted upon in this group if all the group targets are met
                for idx in indexes:
                    action_mask[idx + extra_action] = 1 if idx in selected else 0
            else:
                # Block selection for all ingredients in this group until all group targets are met
                action_mask[indexes + extra_action] = 0
        else:
            # If the group target is not met, allow selection for all ingredients in this group
            action_mask[indexes + extra_action] = 1
        
    # Create mask for actions on selected ingredients
    action_mask = ingredient_action_1(self, all_group_target_met, action_mask, step_to_reward)

    return action_mask

def ingredient_action_1(self, all_group_target_met, action_mask, step_to_reward):
    
    nsteps = self.env.get_wrapper_attr('nsteps')  # Number of steps taken so far
    if get_env_name_from_class(self) != 'SchoolMealSelectionDiscreteDone':
        # Allow all actions to be taken on an ingredient if all group targets are met and nsteps is greater than step_to_reward
        if all_group_target_met and nsteps > step_to_reward:
            action_mask[:3] = [1, 1, 1] 
        else:
            # If group targets are not met, only allow the increase action as only ingredients which are needed for group target will be selected
            action_mask[:3] = [0, 0, 1]  # Only allow the "increase" action or "zero" action
    else:
        # Extra done action if this env
        # Allow all actions to be taken on an ingredient if all group targets are met and nsteps is greater than step_to_reward
        if all_group_target_met and nsteps > step_to_reward:
            action_mask[:6] = [1, 1, 1, 1, 1, 1]  # Allow all actions if all group targets are met and steps > step_to_reward
        else:
            # If group targets are not met, only allow the increase action as only ingredients which are needed for group target will be selected
            action_mask[:6] = [0, 0, 0, 1, 1, 0]  # Only allow the "increase" action and no done signal
    return action_mask

def mask_fn2(self) -> np.ndarray:
    """
    Generate an action mask indicating valid actions.
    """
    # Cache necessary attributes
    n_ingredients = self.env.get_wrapper_attr('n_ingredients')  # Number of ingredients
    
    get_metrics = self.env.get_wrapper_attr('get_metrics')  # Metrics
    
    step_to_reward = self.env.get_wrapper_attr('step_to_reward')  # Number of ingredients
    
    get_metrics()  # Ensure metrics are up-to-date
    
    # Get current state attributes
    ingredient_group_count = self.env.get_wrapper_attr('ingredient_group_count')  # Current count of each ingredient group
    ingredient_group_count_targets = self.env.get_wrapper_attr('ingredient_group_count_targets')  # Target count for each ingredient group
    
    ingredient_group_portion = self.env.get_wrapper_attr('ingredient_group_portion')  # Portion sizes for each ingredient group
    ingredient_group_portion_targets = self.env.get_wrapper_attr('ingredient_group_portion_targets')  # Target portion size range for each ingredient group
    
    group_info = self.env.get_wrapper_attr('group_info')  # Information about each group
    current_selection = self.env.get_wrapper_attr('current_selection')  # Current ingredient selection
    
    if get_env_name_from_class(self) != 'SchoolMealSelectionDiscreteDone':
        extra_action = 3 # First 3 for actions, rest for ingredients
    else:
        extra_action = 6 # First 6 for actions on ingredients, rest for ingredient selection
    
    # Initialize action mask with zeros
    action_mask = np.zeros(extra_action + n_ingredients, dtype=np.int8)  
    
    # Determine if all group count targets have been met
    all_group_count_target_met = all(
        ingredient_group_count[key] >= count_target
        for key, count_target in ingredient_group_count_targets.items()
    )
    
    # Determine if all group portion targets have been met
    all_group_portion_target_met = all(
        portion_target[0] <= ingredient_group_portion[key] <= portion_target[1]
        for key, portion_target in ingredient_group_portion_targets.items()
    )
    
    # Iterate through each ingredient group to set the action mask
    for key, count_target in ingredient_group_count_targets.items():
        value = ingredient_group_count[key]  # Current count of the ingredient group
        portion = ingredient_group_portion[key]  # Current portion of the ingredient group
        portion_target = ingredient_group_portion_targets[key]  # Target portion range for the ingredient group
        indexes = group_info[key]['indexes']  # Indexes of ingredients in this group
        selected = [idx for idx in indexes if current_selection[idx] > 0]  # Selected ingredients in this group
        
        if count_target == 0 and portion_target == (0, 0):
            # If the target is zero for both count and portion, block selection for all these ingredients
            action_mask[indexes + extra_action] = 0
        elif value >= count_target:
            if portion_target[0] <= portion <= portion_target[1]:
                # If both group targets are met or exceeded
                if all_group_count_target_met and all_group_portion_target_met:
                    # Allow only selected ingredients to be acted upon in this group if all the group targets are met
                    for idx in indexes:
                        action_mask[idx + extra_action] = 1 if idx in selected else 0
                else:
                    # Block selection for all ingredients in this group until all group targets are met
                    action_mask[indexes + extra_action] = 0
            else: # If portion not met only allow action to that ingredient selected
                for idx in indexes:
                    action_mask[idx + extra_action] = 1 if idx in selected else 0           
        else:
            # If the group targets are not met
            if value < count_target:
                # Allow selection if count is not met or portion is below the minimum target
                action_mask[indexes + extra_action] = 1
            else:
                # Block selection if count is met and portion is within the target range
                action_mask[indexes + extra_action] = 0
        
    # Create mask for actions on selected ingredients
    action_mask = ingredient_action_2(self, all_group_count_target_met, all_group_portion_target_met, action_mask, step_to_reward)

    return action_mask

def ingredient_action_2(self, all_group_count_target_met, all_group_portion_target_met, action_mask, step_to_reward):
    """
    Update action mask based on whether count and portion targets for all groups are met.
    """
    nsteps = self.env.get_wrapper_attr('nsteps')  # Number of steps taken so far

    if get_env_name_from_class(self) != 'SchoolMealSelectionDiscreteDone':
        # For environments other than 'SchoolMealSelectionDiscreteDone'
        if all_group_count_target_met and all_group_portion_target_met and nsteps > step_to_reward:
            # Allow all actions if both targets are met and steps > step_to_reward
            action_mask[:3] = [1, 1, 1]
        else:
            if not all_group_count_target_met:
                # Allow only the "increase" action if count targets are not met
                action_mask[:3] = [0, 0, 1]
            elif not all_group_portion_target_met:
                if all_group_count_target_met:
                    # Allow "increase" and "decrease" actions if count targets are met but portion targets are not
                    action_mask[:3] = [0, 1, 1]
                else:
                    # Allow "decrease", and "increase" actions if neither count nor portion targets are met
                    action_mask[:3] = [0, 1, 1]
    else:
        # For 'SchoolMealSelectionDiscreteDone' environment
        if all_group_count_target_met and all_group_portion_target_met and nsteps > step_to_reward:
            # Allow all actions if both targets are met and steps > step_to_reward
            action_mask[:6] = [1, 1, 1, 1, 1, 1]
        else:
            if not all_group_count_target_met:
                # Allow only the "increase" action if count targets are not met
                action_mask[:6] = [0, 0, 0, 1, 1, 0]
            else:
                # Allow "increase" and "decrease" actions with no done signal
                action_mask[:6] = [0, 0, 1, 1, 1, 0]

    return action_mask



    
def get_env_name_from_class(self):
    class_name = self.env.unwrapped.__class__.__name__
    return class_name
