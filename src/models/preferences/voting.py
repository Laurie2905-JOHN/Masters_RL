from models.preferences.data_utils import get_supplier_availability
import pandas as pd

def generate_ordered_list(ingredients_list):
    return {ingredient: index + 1 for index, ingredient in enumerate(ingredients_list)}

def compare_negotiated_ingredients(old_ingredients, new_ingredients, old_unavailable, new_unavailable):
    changes = {}

    all_ingredient_types = set(old_ingredients.keys()).union(new_ingredients.keys()).union(old_unavailable.keys()).union(new_unavailable.keys())

    for ingredient_type in all_ingredient_types:
        old_list = old_ingredients.get(ingredient_type, [])
        new_list = new_ingredients.get(ingredient_type, [])
        old_unavail_list = old_unavailable.get(ingredient_type, [])
        new_unavail_list = new_unavailable.get(ingredient_type, [])

        old_order = generate_ordered_list(old_list)
        new_order = generate_ordered_list(new_list)

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
            
    # Display the changes in a readable format
    for ingredient_type, order_changes in changes.items():
        print(f"Changes in {ingredient_type}:")
        for ingredient, old_pos, new_pos in order_changes:
            print(f"{ingredient}: Pos: {old_pos} -> Pos: {new_pos}")

def calculate_child_weight_complex(child, preferences, feedback):
    
    # Normalize likes, neutrals, and dislikes so each child has equal contribution
    total_likes = len(preferences["likes"])
    total_neutrals = len(preferences["neutral"])
    total_dislikes = len(preferences["dislikes"])
    
    # Make vote proportional to like/dislike ratio
    like_dislike_ratio = total_likes / total_dislikes
    
    child_merit_weight = 1  # Placeholder for child-specific merit
    
    if str(child) in feedback.keys():
        feedback_provided_weight = 1
    else:
        feedback_provided_weight = 0.9
    
    # Ensure each category contributes equally
    normalized_likes = 1 / max(1, total_likes)
    normalized_neutrals = 1 / max(1, total_neutrals)
    normalized_dislikes = 1 / max(1, total_dislikes)
    
    return {
        'likes': normalized_likes * like_dislike_ratio * child_merit_weight * feedback_provided_weight,
        'neutral': normalized_neutrals * like_dislike_ratio * child_merit_weight * feedback_provided_weight,
        'dislikes': normalized_dislikes * like_dislike_ratio * child_merit_weight * feedback_provided_weight
    }

def calculate_child_weight_simple(child, preferences):
    return {
        'likes': 1,
        'neutral': 1,
        'dislikes': 1
    }

def collect_weighted_votes(preferences, ingredient_df, weight_function, seed):
    # Initialize the availability and votes dictionaries
    supplier_availability = get_supplier_availability(ingredient_df, mean_unavailable=5, std_dev_unavailable=2, seed=seed)
    votes = {ingredient: 0 for ingredient in supplier_availability}  # Initialize votes for each ingredient
    unavailable_ingredients = set()
    
    for child, pref in preferences.items():
        # Calculate weights based on preferences and stability for each child
        weights = weight_function(child, pref)
        
        # Convert preference lists to sets for efficient lookups
        likes = set(pref["likes"])
        neutrals = set(pref["neutral"])
        dislikes = set(pref["dislikes"])
        
        # Iterate over all unique ingredients in preferences
        for ingredient in likes | neutrals | dislikes:
            # Check which category the ingredient belongs to
            in_likes = ingredient in likes
            in_neutrals = ingredient in neutrals
            in_dislikes = ingredient in dislikes

            # Raise an error if an ingredient is found in more than one category
            if sum([in_likes, in_neutrals, in_dislikes]) > 1:
                raise ValueError(f"Ingredient {ingredient} is found in multiple categories")
            
            # If the ingredient is available, update its vote count based on its category and weight
            if ingredient in supplier_availability and supplier_availability[ingredient] > 0:
                if in_likes:
                    votes[ingredient] += 5 * weights['likes']
                elif in_neutrals:
                    votes[ingredient] += 3 * weights['neutral']
                elif in_dislikes:
                    votes[ingredient] += 1 * weights['dislikes']  # Adjusted weight for dislikes
            else:
                # If the ingredient is not available, add it to the set of unavailable ingredients
                unavailable_ingredients.add(ingredient)
    
    # Return the votes and the set of unavailable ingredients
    return votes, unavailable_ingredients


def negotiate_ingredients_simple(preferences, ingredient_df, seed):
    # Define the ingredient groups as columns in the one-hot encoded DataFrame
    ingredient_groups = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E', 'Bread', 'Confectionary']
    
    # Initialize dictionaries for negotiated ingredients and unavailable ingredients
    negotiated_ingredients = {}
    weight_function = calculate_child_weight_simple

    # Collect votes and unavailable ingredients for all ingredients
    votes, unavailable_ingredients = collect_weighted_votes(preferences, ingredient_df, weight_function, seed)
    
    # Initialize category votes dictionary with ingredient groups
    category_votes = {group: {} for group in ingredient_groups}
    
    # Map ingredients to their groups for efficient lookup
    ingredient_to_groups = {ingredient: group for group in ingredient_groups for ingredient in ingredient_df[ingredient_df[group] == 1]['Category7']}
    
    # Collect votes for each ingredient group
    for ingredient, vote in votes.items():
        if ingredient not in unavailable_ingredients:
            group = ingredient_to_groups.get(ingredient)
            if group:
                category_votes[group][ingredient] = category_votes[group].get(ingredient, 0) + vote

    # Sort and store the ingredients for each category
    for group, votes_dict in category_votes.items():
        sorted_ingredients = sorted(votes_dict, key=votes_dict.get, reverse=True)
        negotiated_ingredients[group] = sorted_ingredients

    # Handle Misc category
    misc_df = ingredient_df[(ingredient_df[ingredient_groups].sum(axis=1) == 0)]
    misc_votes = {ingredient: votes[ingredient] for ingredient in misc_df['Category7'].tolist() if ingredient in votes and ingredient not in unavailable_ingredients}
    sorted_misc_ingredients = sorted(misc_votes, key=misc_votes.get, reverse=True)
    negotiated_ingredients['Misc'] = sorted_misc_ingredients

    return negotiated_ingredients, unavailable_ingredients



# def negotiate_ingredients_complex(preferences, rf_model, preprocessor, child_data, ingredients_data):
#     ingredient_groups = set(details['type'] for details in ingredients_data.values())
#     negotiated_ingredients = {}
#     unavailable_ingredients = {}

#     for ingredient_type in ingredient_types:
#         votes, unavailable = collect_weighted_votes(preferences, ingredient_type, rf_model, preprocessor, child_data, ingredients_data)
        
#         # Sort ingredients based on votes
#         sorted_ingredients = sorted(votes, key=votes.get, reverse=True)
#         top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]
        
#         # Check top 10% and ensure no child has all their dislikes in this section
#         for child, pref in preferences.items():
#             dislikes_in_top_10 = set(pref['dislikes']).intersection(top_10_percent)
#             if len(dislikes_in_top_10) == len(top_10_percent):
#                 # If all top 10% are disliked by one child, reassign scores to balance
#                 for dislike in dislikes_in_top_10:
#                     votes[dislike] -= 1  # Penalize disliked items slightly
#                     sorted_ingredients = sorted(votes, key=votes.get, reverse=True)
#                     top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]
        
#         negotiated_ingredients[ingredient_type] = sorted_ingredients
        
#         # Track unavailable ingredients
#         if unavailable:
#             unavailable_ingredients[ingredient_type] = unavailable
    
#     return negotiated_ingredients, unavailable_ingredients