
from models.preferences.data_utils import get_supplier_availability
from models.preferences.prediction import predict_preference

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

def collect_weighted_votes(preferences, ingredient_type, rf_model, preprocessor, child_data, ingredients_data, weight_function):
    supplier_availability = get_supplier_availability()
    
    available_ingredients = {ingredient: 0 for ingredient, details in ingredients_data.items() if details['type'] == ingredient_type and supplier_availability.get(ingredient, False)}
    unavailable_ingredients = [ingredient for ingredient, available in supplier_availability.items() if not available and ingredient in ingredients_data and ingredients_data[ingredient]['type'] == ingredient_type]
    
    for child, pref in preferences.items():
        
        # Prepare child features for prediction
        child_features = {
            "age": child_data[child]["age"],
            "gender": 1 if child_data[child]["gender"] == "M" else 0,
            "health_consideration": child_data[child]["health_consideration"],
            "favorite_cuisine": child_data[child]["favorite_cuisine"]
        }
        
        # Calculate weights based on preferences and stability
        weights = weight_function(child, pref)
        
        likes = set(pref["likes"])
        neutrals = set(pref["neutral"])
        dislikes = set(pref["dislikes"])
        
        # Update available ingredients' scores based on preferences and weights
        for ingredient in available_ingredients.keys():
            if ingredient in likes:
                available_ingredients[ingredient] += 5 * weights['likes']
            elif ingredient in neutrals:
                available_ingredients[ingredient] += 3 * weights['neutral']
            elif ingredient not in dislikes:
                # Predict preference for ingredients not explicitly liked or disliked
                ingredient_details = ingredients_data[ingredient]
                ingredient_features = {
                    "ingredient": ingredient,
                    "type": ingredient_details["type"],
                    "color": ingredient_details["color"],
                    "taste": ingredient_details["taste"]
                }
                predicted_preference = predict_preference(child_features, ingredient_features, rf_model, preprocessor)
                available_ingredients[ingredient] += predicted_preference * weights['dislikes']
    
    return available_ingredients, unavailable_ingredients


def negotiate_ingredients_simple(preferences, rf_model, preprocessor, child_data, ingredients_data):
    ingredient_types = set(details['type'] for details in ingredients_data.values())
    negotiated_ingredients = {}
    unavailable_ingredients = {}

    for ingredient_type in ingredient_types:
        votes, unavailable = collect_weighted_votes(preferences, ingredient_type, rf_model, preprocessor, child_data, ingredients_data)
        sorted_ingredients = sorted(votes, key=votes.get, reverse=True)
        negotiated_ingredients[ingredient_type] = sorted_ingredients
        if unavailable:
            unavailable_ingredients[ingredient_type] = unavailable
    
    return negotiated_ingredients, unavailable_ingredients

def negotiate_ingredients_complex(preferences, rf_model, preprocessor, child_data, ingredients_data):
    ingredient_types = set(details['type'] for details in ingredients_data.values())
    negotiated_ingredients = {}
    unavailable_ingredients = {}

    for ingredient_type in ingredient_types:
        votes, unavailable = collect_weighted_votes(preferences, ingredient_type, rf_model, preprocessor, child_data, ingredients_data)
        
        # Sort ingredients based on votes
        sorted_ingredients = sorted(votes, key=votes.get, reverse=True)
        top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]
        
        # Check top 10% and ensure no child has all their dislikes in this section
        for child, pref in preferences.items():
            dislikes_in_top_10 = set(pref['dislikes']).intersection(top_10_percent)
            if len(dislikes_in_top_10) == len(top_10_percent):
                # If all top 10% are disliked by one child, reassign scores to balance
                for dislike in dislikes_in_top_10:
                    votes[dislike] -= 1  # Penalize disliked items slightly
                    sorted_ingredients = sorted(votes, key=votes.get, reverse=True)
                    top_10_percent = sorted_ingredients[:max(1, len(sorted_ingredients) // 10)]
        
        negotiated_ingredients[ingredient_type] = sorted_ingredients
        
        # Track unavailable ingredients
        if unavailable:
            unavailable_ingredients[ingredient_type] = unavailable
    
    return negotiated_ingredients, unavailable_ingredients