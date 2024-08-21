import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random
from typing import Dict, Any, Tuple, List, Callable, Optional
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
import json
import os

def save_negotiated_and_unavailable_ingredients(args, file_path):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Prepare the data to be saved
    data_to_save = {
        'negotiated_ingredients': args.negotiated_ingredients,
        'unavailable_ingredients': list(args.unavailable_ingredients)
    }

    # Save data as JSON
    with open(file_path, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

def get_child_data():
    # Function to get feature data on children
    return {
    "child1": {"age": 10, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child2": {"age": 9, "gender": "F", "health_consideration": "health focused", "favorite_cuisine": "Italian", "feedback_chance": 0.5},
    "child3": {"age": 9, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "Italian", "feedback_chance": 1},
    "child4": {"age": 9, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child5": {"age": 11, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "Italian", "feedback_chance": 1},
    "child6": {"age": 11, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child7": {"age": 9, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "Italian", "feedback_chance": 1},
    "child8": {"age": 9, "gender": "F", "health_consideration": "health focused", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child9": {"age": 10, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child10": {"age": 11, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 0.5},
    "child11": {"age": 9, "gender": "F", "health_consideration": "moderate", "favorite_cuisine": "Italian", "feedback_chance": 0.5},
    "child12": {"age": 9, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 1},
    "child13": {"age": 9, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 1},
    "child14": {"age": 10, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "Seafood", "feedback_chance": 0},
    "child15": {"age": 11, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 0.5},
    "child16": {"age": 11, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 0},
    "child17": {"age": 9, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 1},
    "child18": {"age": 9, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 1},
    "child19": {"age": 9, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 0.5},
    "child20": {"age": 10, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "BBQ", "feedback_chance": 0},
    "child21": {"age": 10, "gender": "F", "health_consideration": "health focused", "favorite_cuisine": "BBQ", "feedback_chance": 0.5},
    "child22": {"age": 9, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "BBQ", "feedback_chance": 0.5},
    "child23": {"age": 9, "gender": "F", "health_consideration": "moderate", "favorite_cuisine": "BBQ", "feedback_chance": 0.5},
    "child24": {"age": 9, "gender": "M", "health_consideration": "indifferent", "favorite_cuisine": "BBQ", "feedback_chance": 0},
    "child25": {"age": 11, "gender": "F", "health_consideration": "health focused", "favorite_cuisine": "BBQ", "feedback_chance": 1},
    "child26": {"age": 11, "gender": "M", "health_consideration": "moderate", "favorite_cuisine": "BBQ", "feedback_chance": 0.5},
    "child27": {"age": 9, "gender": "F", "health_consideration": "moderate", "favorite_cuisine": "Seafood", "feedback_chance": 0.5},
    "child28": {"age": 9, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Seafood", "feedback_chance": 0.5},
    "child29": {"age": 9, "gender": "M", "health_consideration": "health focused", "favorite_cuisine": "Italian", "feedback_chance": 0},
    "child30": {"age": 10, "gender": "F", "health_consideration": "indifferent", "favorite_cuisine": "Italian", "feedback_chance": 1}
}



def get_modifiers(
    features: Dict[str, Any],
    ingredient_row: pd.Series,
    health_consideration_modifiers: Dict[str, Dict[str, float]],
    favorite_cuisine_modifiers: Dict[str, Dict[str, float]],
    taste_modifiers: Dict[str, float],
    colour_modifiers: Dict[str, float],
    gender_modifiers: Dict[str, float],
    age_modifiers: Dict[int, float],
    texture_modifiers: Dict[str, float],
    other_modifiers: Dict[str, Any],
    vegetable_groups: Dict[str, list],
    group_probabilities_modifiers: Dict[str, float]
) -> float:
    health_consideration = features["health_consideration"]
    age = features["age"]
    gender = features["gender"]
    favorite_cuisine = features["favorite_cuisine"]

    health_category = ingredient_row["Healthy"]
    ingredient_category1 = ingredient_row["Category1"]
    taste = ingredient_row["Taste"]
    colour = ingredient_row["Colour"]
    texture = ingredient_row["Texture"]
    ingredient = ingredient_row["Category7"]

    health_mod = health_consideration_modifiers[health_consideration][health_category]
    favorite_mod = favorite_cuisine_modifiers.get(favorite_cuisine, {}).get(ingredient_category1, 1)
    taste_mod = taste_modifiers.get(taste, taste_modifiers["Misc"])
    colour_mod = colour_modifiers[colour]
    texture_mod = texture_modifiers[texture]
    gender_mod = gender_modifiers[gender]
    age_mod = age_modifiers[age]

    group_name = next((group for group, ingredients in vegetable_groups.items() if ingredient in ingredients), None)
    group_mod = group_probabilities_modifiers.get(group_name, 1)

    fruit_mod = other_modifiers["fruit_factor"] if ingredient_category1 == "Fruits and fruit products" else 1
    vegetable_mod = other_modifiers["vegetables_factor"][gender] if ingredient_category1 == "Vegetables and vegetable products" else 1
    meat_mod = other_modifiers["meat_factor"][gender] if ingredient_category1 == "Meat and meat products" else 1
    random_mod = random.uniform(other_modifiers["random_factor"][0], other_modifiers["random_factor"][1])
    
    return (health_mod * favorite_mod * taste_mod * colour_mod * gender_mod * age_mod * 
            texture_mod * group_mod * fruit_mod * vegetable_mod * meat_mod)

def initialize_child_preference_data(child_data: Dict[str, Dict[str, Any]], ingredient_df: pd.DataFrame, split: float = 0.5, seed: int = None, plot_graphs: bool = False, child_key_plot: str = None) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, list]]]:
    random.seed(seed)
    children_data = {}
    all_scores = {}
    
    try:
        if child_key_plot is not None and child_key_plot not in child_data:
            raise ValueError(f"Child key '{child_key_plot}' not found in child_data.")
    except ValueError as e:
        print(e)
        child_key_plot = None  # Reset child_key_plot to None to handle the case gracefully

    # Factors affecting preferences with modifier values (increased impact)
    health_consideration_modifiers = {
        "health focused": {"healthy": 1.5, "average": 1, "unhealthy": 0.5},
        "moderate": {"healthy": 1.3, "average": 1, "unhealthy": 0.7},
        "indifferent": {"healthy": 0.7, "average": 1, "unhealthy": 1.3},
    }
    
    # Factors affecting like/neutral/dislike ration with health conciousness. The more health focused you are the less picky
    health_consideration_modifiers_ratio = {
        "health focused": {'likes': 0.7, 'neutral': 0.2, 'dislikes': 0.1},
        "moderate": {'likes': 0.5, 'neutral': 0.3, 'dislikes': 0.2},
        "indifferent": {'likes': 0.35, 'neutral': 0.25, 'dislikes': 0.4}
    }

    favorite_cuisine_modifiers = {
        "BBQ": {"Meat and meat products": 2},
        "Seafood": {"Fish seafood amphibians reptiles and invertebrates": 1.4},
        "Italian": {"Anchovies": 2, "Aubergines": 2, "Noodles": 2, "Pasta plain (not stuffed) uncooked": 2, "Pasta wholemeal": 2, "Tomatoes": 2},
    }

    taste_modifiers = {
        "Sweet": 1.3,
        "Salty": 1.3,
        "Sour": 0.7,
        "Earthy": 0.7,
        "Misc": 1,
    }

    colour_modifiers = {
        "Red": 1.3,
        "Green": 1.3,
        "Yellow": 1.2,
        "Orange": 1.2,
        "Pink": 1,
        "Purple": 1,
        "White": 0.8,
        "Brown": 0.8,
    }

    gender_modifiers = {
        "M": 0.5,
        "F": 1.5,
    }

    age_modifiers = {
        9: 0.5,
        10: 1,
        11: 1.5,
    }

    texture_modifiers = {
        "Crunchy": 0.7,
        "Soft": 1.3,
        "Soft/Crunchy": 0.2,
        "Firm": 1.3,
        "Leafy": 1,
        "Grainy": 1,
        "Liquid": 1,
        "Powdery": 1,
        "Creamy": 1,
        "Hard": 1,
    }

    other_modifiers = {
        "fruit_factor": 1.3,
        "vegetables_factor": {"M": 0.7, "F": 1.3},
        "meat_factor": {"M": 1.3, "F": 0.7},
        "random_factor": [0.8, 1.2],
    }

    vegetable_groups = {
        "Group A": ["Tomatoes", "Sweet corn", "Sweet potatoes", "Carrots"],
        "Group B": ["Onions", "Spring onions", "Pepper"],
        "Group C": ["Cauliflowers"],
        "Group D": ["Courgettes", "Spinaches", "Curly kales", "Peas"],
        "Group E": ["Beetroots", "Lettuces (generic)", "Broccoli"],
        "Group F": ["Aubergines", "Cucumber", "White cabbage", "Savoy cabbages", "Red cabbage", "Runner beans (with pods)"],
    }

    group_probabilities_modifiers = {
        "Group A": 1.4,
        "Group B": 1.3,
        "Group C": 0.7, 
        "Group D": 1, 
        "Group E": 0.9, 
        "Group F": 0.9 
    }

    # Step 1: Calculate scores for each child individually and accumulate them in all_scores
    all_labeled_scores = []  # Initialize list to store labeled scores
    for child_key, features in child_data.items():
        child_scores = []

        for _, row in ingredient_df.iterrows():
            score = get_modifiers(features, row, health_consideration_modifiers, favorite_cuisine_modifiers, taste_modifiers,
                                colour_modifiers, gender_modifiers, age_modifiers, texture_modifiers, other_modifiers,
                                vegetable_groups, group_probabilities_modifiers)
            child_scores.append((row["Category7"], score))  # Store score along with the ingredient

        # Sort scores for this child
        child_scores.sort(key=lambda x: x[1], reverse=True)

        # Determine the number of likes, neutral, and dislikes
        num_ingredients = len(child_scores)
        ratio = health_consideration_modifiers_ratio[child_data[child_key]['health_consideration']]
        
        num_likes = int(ratio['likes'] * num_ingredients)
        num_neutral = int(ratio['neutral'] * num_ingredients)
        num_dislikes = num_ingredients - num_likes - num_neutral

        # Label the scores and store them in all_labeled_scores
        for ingredient, score in child_scores[:num_likes]:
            all_labeled_scores.append((child_key, ingredient, score, 'likes'))
        for ingredient, score in child_scores[num_likes:num_likes + num_neutral]:
            all_labeled_scores.append((child_key, ingredient, score, 'neutral'))
        for ingredient, score in child_scores[num_likes + num_neutral:]:
            all_labeled_scores.append((child_key, ingredient, score, 'dislikes'))

    # Step 2: Generate known and unknown preferences
    all_data = {}
    for child_key in child_data.keys():
        known_preferences = {"likes": [], "neutral": [], "dislikes": []}
        unknown_preferences = {"likes": [], "neutral": [], "dislikes": []}

        for category in ["likes", "neutral", "dislikes"]:
            # Filter ingredients for the current child and category
            category_items = [ingredient for c, ingredient, score, label in all_labeled_scores if c == child_key and label == category]
            
            # Shuffle the list of ingredients randomly
            random.shuffle(category_items)
            
            # Determine the split index
            total_items = len(category_items)
            split_index = int(total_items * split)
            
            # Split into known and unknown preferences
            known_preferences[category] = category_items[:split_index]
            unknown_preferences[category] = category_items[split_index:]

        all_data[child_key] = {
            "known": known_preferences,
            "unknown": unknown_preferences
        }
    # Step 3: Optional plotting of histograms
    if plot_graphs:
        plot_histograms(all_labeled_scores, all_data, child_key=child_key_plot)

    return all_data

    
def plot_histograms(all_scores: List[Tuple[str, str, float, str]], all_preferences: Dict[str, Dict[str, Dict[str, list]]], child_key: str = None, fontsize: int = 15) -> None:
    # If no specific child_key is provided, plot for all children
    if child_key is None:
        # Plot for all children
        filtered_scores = [(ingredient, score, label) for c_key, ingredient, score, label in all_scores]
        title = "Score Distribution for All Children"
        
        # Select a random child for individual plotting
        random_child_key = random.choice(list(all_preferences.keys()))
        random_filtered_scores = [(ingredient, score, label) for c_key, ingredient, score, label in all_scores if c_key == random_child_key]
        random_title = f"Score Distribution for Child {random_child_key[5:7]}"
    else:
        # Plot for the specified child
        filtered_scores = [(ingredient, score, label) for c_key, ingredient, score, label in all_scores if c_key == child_key]
        title = f"Score Distribution for Child {child_key[5:7]}"
        random_filtered_scores = None
        random_title = None

    # Function to plot a histogram given filtered scores and a title
    def plot_histogram(ax, filtered_scores, title, fontsize):
        like_scores = [score for _, score, label in filtered_scores if label == 'likes']
        neutral_scores = [score for _, score, label in filtered_scores if label == 'neutral']
        dislike_scores = [score for _, score, label in filtered_scores if label == 'dislikes']

        # Sort for debugging purposes
        like_scores.sort(reverse=True)
        neutral_scores.sort(reverse=True)
        dislike_scores.sort(reverse=True)
    
        # Calculate total number of scores
        total_scores = len(like_scores) + len(neutral_scores) + len(dislike_scores)
    
        # Calculate percentages
        likes_percent = (len(like_scores) / total_scores) * 100 if total_scores > 0 else 0
        neutral_percent = (len(neutral_scores) / total_scores) * 100 if total_scores > 0 else 0
        dislikes_percent = (len(dislike_scores) / total_scores) * 100 if total_scores > 0 else 0

        # Create bins for the histograms based on all scores
        all_scores_combined = like_scores + neutral_scores + dislike_scores
        if not all_scores_combined:
            print("No scores to display.")
            return
    
        # Calculate bin edges using numpy
        bins = np.linspace(min(all_scores_combined), max(all_scores_combined), 21)  # 20 bins

        likes_label = f'Like ({likes_percent:.1f}\\%)'
        neutral_label = f'Neutral ({neutral_percent:.1f}\\%)'
        dislikes_label = f'Dislike ({dislikes_percent:.1f}\\%)'

        # Plot histograms in a stacked manner using the calculated bin edges
        ax.hist([like_scores, neutral_scores, dislike_scores], bins=bins, stacked=True, color=['green', 'blue', 'red'], 
                label=[likes_label, neutral_label, dislikes_label])

        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.legend(fontsize=fontsize-1, loc='best')


    # Create subplots
    if random_filtered_scores:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        plot_histogram(axs[0], filtered_scores, title, fontsize)
        plot_histogram(axs[1], random_filtered_scores, random_title, fontsize)
        # Set shared x-label and y-label
        fig.text(0.5, 0.04, 'Score', ha='center', fontsize=fontsize)
        fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=fontsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        plot_histogram(ax, filtered_scores, title, fontsize)
        ax.set_xlabel('Score', fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
    
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])  # Adjust layout to make space for shared labels
    plt.show()

def create_preference_score_function(negotiated_ingredients: set[str, Dict[str, float]], unavailable_ingredients: set[str]) -> Callable[[str], float]:
    """
    Create a preference score function based on negotiated ingredients, excluding unavailable ingredients.

    :param negotiated_ingredients: Dictionary of negotiated ingredients and their scores.
    :param unavailable_ingredients: List of unavailable ingredients.
    :return: Function that returns the score for a given ingredient.
    """
    ingredient_scores = {}
    if negotiated_ingredients:
        for group, ingredients in negotiated_ingredients.items():
            # Filter out unavailable ingredients
            available_ingredients = {k: v for k, v in ingredients.items() if k not in unavailable_ingredients}
            
            if available_ingredients:
                scores = np.array(list(available_ingredients.values())).reshape(-1, 1)
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(scores).flatten()
                for ingredient, norm_score in zip(available_ingredients.keys(), normalized_scores):
                    ingredient_scores[ingredient] = norm_score

    def score_function(ingredient: str) -> float:
        if ingredient not in ingredient_scores.keys() and ingredient not in unavailable_ingredients:
            raise ValueError(f"Ingredient '{ingredient}' not found in ingredient_scores ingredients.")
        else:
            return ingredient_scores[ingredient]

    return score_function

def plot_preference_and_sentiment_accuracies(prediction_accuracies, prediction_std_devs, sentiment_accuracies, iterations, save_path):
    """Function to plot the accuracies and standard deviations over iterations and save the plot to a specified file path."""
    plt.figure(figsize=(18, 6))

    # Unpack tuples into separate lists
    simple_accuracies, complex_accuracies = zip(*prediction_accuracies)
    simple_std_devs, complex_std_devs = zip(*prediction_std_devs)
    simple_sentiment_accuracies, complex_sentiment_accuracies = zip(*sentiment_accuracies)

    # Plot Prediction Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(range(1, iterations + 1), simple_accuracies, marker='o', linestyle='-', label='Simple Prediction Accuracy')
    plt.plot(range(1, iterations + 1), complex_accuracies, marker='x', linestyle='-', label='Complex Prediction Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy per Iteration')
    plt.legend()
    plt.grid(True)

    # Plot Prediction Standard Deviation
    plt.subplot(1, 3, 2)
    plt.plot(range(1, iterations + 1), simple_std_devs, marker='o', linestyle='-', color='green', label='Simple Prediction Std Dev')
    plt.plot(range(1, iterations + 1), complex_std_devs, marker='x', linestyle='-', color='blue', label='Complex Prediction Std Dev')
    plt.xlabel('Iteration')
    plt.ylabel('Standard Deviation')
    plt.title('Prediction Standard Deviation per Iteration')
    plt.legend()
    plt.grid(True)

    # Plot Sentiment Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(range(1, iterations + 2), simple_sentiment_accuracies, marker='o', linestyle='-', color='orange', label='Simple Sentiment Accuracy')
    plt.plot(range(1, iterations + 2), complex_sentiment_accuracies, marker='x', linestyle='-', color='purple', label='Complex Sentiment Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Sentiment Accuracy per Iteration')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_individual_child_known_percent(child_known_percents, save_path):
    """Function to plot the accuracies of each child over iterations and save the plot to a specified file path."""
    iterations = len(child_known_percents)

    plt.figure(figsize=(12, 8))

    # Prepare data for plotting
    children = list(child_known_percents[0][0].keys())
    known_per_child_simple = {child: [] for child in children}
    known_per_child_complex = {child: [] for child in children}

    for simple_data, complex_data in child_known_percents:
        for child in children:
            known_per_child_simple[child].append(simple_data[child])
            known_per_child_complex[child].append(complex_data[child])

    # Select 5 random children
    selected_children = random.sample(children, 5)
    
    # Plot percent for each child
    for child in selected_children:
        plt.plot(range(1, iterations + 1), known_per_child_simple[child], marker='o', linestyle='-', label=f'Simple - Child {child}')
        plt.plot(range(1, iterations + 1), known_per_child_complex[child], marker='x', linestyle='--', label=f'Complex - Child {child}')

    plt.xlabel('Iteration')
    plt.ylabel("Percent Known")
    plt.title('Child Percent Known per Iteration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    

def plot_utilities_from_json(file_path: str, save_path: str = None) -> None:
    with open(file_path, 'r') as file:
        data = json.load(file)

    if 'complex' in file_path:
        name_ext = 'complex'
    else:
        name_ext = 'simple'
    weeks = len(data)

    # Utility types and their sum keys
    utilities = ["true_raw_utility", "predicted_raw_utility", "true_utility", "predicted_utility"]
    sums = ["sum_true_raw_utility", "sum_predicted_raw_utility", "sum_true_utility", "sum_predicted_utility"]

    # Plot each type of utility for all weeks and days
    for utility in utilities:
        plt.figure(figsize=(12, 6))
        children_data = {}
        for week_data in data:
            week = week_data["week"]
            days = sorted(week_data[utility].keys(), key=int)[1:]  # Skip the first day
            for day in days:
                day_data = week_data[utility][day]
                for child, value in day_data.items():
                    if child not in children_data:
                        children_data[child] = []
                    children_data[child].append((f'Week {week}, Day {day}', value))

        # Select 5 random children
        selected_children = np.random.choice(list(children_data.keys()), size=5, replace=False)

        for child in selected_children:
            values = children_data[child]
            x, y = zip(*values)
            y_smoothed = gaussian_filter1d(y, sigma=2)  # Apply smoothing
            plt.plot(x, y_smoothed, label=f'Child {child}')
        
        plt.title(f'{utility.replace("_", " ").title()} per Child')
        plt.xlabel('Day and Week')
        plt.ylabel('Utility')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
        plt.xticks(rotation=45)
        plt.xticks(range(0, len(x), max(1, len(x)//10)), rotation=45)  # Reduce number of x-ticks
        save_or_show_plot(f'{utility}_per_child_{name_ext}.png', save_path)

    # Plot daily Gini coefficients for all weeks
    plt.figure(figsize=(12, 6))
    daily_gini_true = []
    daily_gini_pred = []
    days_labels = []
    for week_data in data:
        week = week_data["week"]
        days = sorted(week_data["daily_gini_coefficients"].keys(), key=int)[1:]  # Skip the first day
        for day in days:
            gini_data = week_data["daily_gini_coefficients"][day]
            daily_gini_true.append(gini_data["true_gini"])
            daily_gini_pred.append(gini_data["predicted_gini"])
            days_labels.append(f'Week {week}, Day {day}')
    daily_gini_true_smoothed = gaussian_filter1d(daily_gini_true, sigma=2)  # Apply smoothing
    daily_gini_pred_smoothed = gaussian_filter1d(daily_gini_pred, sigma=2)  # Apply smoothing
    plt.plot(days_labels, daily_gini_true_smoothed, label='True Gini')
    plt.plot(days_labels, daily_gini_pred_smoothed, label='Predicted Gini')
    plt.title('Daily Gini Coefficients')
    plt.xlabel('Day and Week')
    plt.ylabel('Coefficient')
    plt.legend(loc='lower center', ncol=2)
    plt.xticks(rotation=45)
    plt.xticks(range(0, len(days_labels), max(1, len(days_labels)//10)), rotation=45)  # Reduce number of x-ticks
    save_or_show_plot(f'daily_gini_coefficients{name_ext}.png', save_path)

    # Plot daily Gini coefficients for all weeks
    plt.figure(figsize=(12, 6))
    daily_gini_true = []
    daily_gini_pred = []
    days_labels = []
    for week_data in data:
        week = week_data["week"]
        for day, gini_data in week_data["daily_gini_coefficients"].items():
            daily_gini_true.append(gini_data["true_gini"])
            daily_gini_pred.append(gini_data["predicted_gini"])
            days_labels.append(f'Week {week}, Day {day}')
    daily_gini_true_smoothed = gaussian_filter1d(daily_gini_true, sigma=2)  # Apply smoothing
    daily_gini_pred_smoothed = gaussian_filter1d(daily_gini_pred, sigma=2)  # Apply smoothing
    plt.plot(days_labels, daily_gini_true_smoothed, label='True Gini')
    plt.plot(days_labels, daily_gini_pred_smoothed, label='Predicted Gini')
    plt.title('Daily Gini Coefficients')
    plt.xlabel('Day and Week')
    plt.ylabel('Coefficient')
    plt.legend(loc='lower center', ncol=2)
    plt.xticks(rotation=45)
    plt.xticks(range(0, len(days_labels), max(1, len(days_labels)//10)), rotation=45)  # Reduce number of x-ticks
    save_or_show_plot(f'daily_gini_coefficients_{name_ext}.png', save_path)

    # Plot cumulative Gini coefficients per week
    plt.figure(figsize=(12, 6))
    cumulative_gini_true = []
    cumulative_gini_pred = []
    week_labels = []
    for week_data in data:
        week = week_data["week"]
        cumulative_gini = week_data["cumulative_gini_coefficients"][-1]  # Get the last cumulative Gini for each week
        cumulative_gini_true.append(cumulative_gini["true_cumulative_gini"])
        cumulative_gini_pred.append(cumulative_gini["predicted_cumulative_gini"])
        week_labels.append(f'Week {week}')
    cumulative_gini_true_smoothed = gaussian_filter1d(cumulative_gini_true, sigma=2)  # Apply smoothing
    cumulative_gini_pred_smoothed = gaussian_filter1d(cumulative_gini_pred, sigma=2)  # Apply smoothing
    plt.plot(week_labels, cumulative_gini_true_smoothed, label='True Cumulative Gini')
    plt.plot(week_labels, cumulative_gini_pred_smoothed, label='Predicted Cumulative Gini')
    plt.title('Cumulative Gini Coefficients per Week')
    plt.xlabel('Week')
    plt.ylabel('Coefficient')
    plt.legend(loc='lower center', ncol=2)
    plt.xticks(rotation=45)
    plt.xticks(range(0, len(week_labels), max(1, len(week_labels)//10)), rotation=45)  # Reduce number of x-ticks
    save_or_show_plot(f'cumulative_gini_coefficients_{name_ext}.png', save_path)
    

def calculate_mape(true_values: Dict[str, float], predicted_values: Dict[str, float]) -> Optional[float]:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.
    Return None if division by zero would occur.
    """
    mape_sum = 0
    valid_count = 0
    
    for child in true_values:
        if true_values[child] != 0:
            mape_sum += abs(true_values[child] - predicted_values[child]) / true_values[child]
            valid_count += 1
        else:
            return None  # Return None if true value is zero
    
    if valid_count == 0:
        return None
    
    mape = 100 * mape_sum / valid_count
    return mape

def save_or_show_plot(filename: str, save_path: str = None) -> None:
    if save_path and filename:
        plt.savefig(f"{save_path}/{filename}")
    else:
        plt.show()
    plt.close()

def plot_utilities_and_mape(file_path: str, save_path: str = None) -> None:
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    if 'complex' in file_path:
        name_ext = 'complex'
    else:
        name_ext = 'simple'

    # Utility types and their sum keys
    sums = ["sum_true_utility", "sum_predicted_utility"]

    # Plot sums of utilities for all weeks
    for sum_utility in sums:
        plt.figure(figsize=(12, 6))
        all_days = []
        all_values = []
        true_utility_per_day = []
        predicted_utility_per_day = []
        for week_data in data:
            week = week_data["week"]
            days = sorted(week_data[sum_utility].keys(), key=int)
            values = [week_data[sum_utility][day] for day in days]
            true_utility = [week_data["true_utility"][day] for day in days]
            predicted_utility = [week_data["predicted_utility"][day] for day in days]
            all_days.extend([f'Week {week}, Day {day}' for day in days])
            all_values.extend(values)
            true_utility_per_day.extend(true_utility)
            predicted_utility_per_day.extend(predicted_utility)
        plt.plot(all_days[1:], all_values[1:])
        title = f'{sum_utility.replace("_", " ").title()} per Day'
        plt.title(title)
        plt.xlabel('Day and Week')
        plt.ylabel('Sum of Utilities')
        plt.xticks(range(0, len(all_days), max(1, len(all_days) // 10)), rotation=45)
        save_or_show_plot(f'{sum_utility}_per_day_{name_ext}.png', save_path)

    # Calculate and plot accuracy (MAPE) between true and predicted utility cumulative sums
    plot_mape(all_days, true_utility_per_day, predicted_utility_per_day, title, save_path, name_ext)

def plot_mape(days_labels: List[str], true_utility: List[Dict[str, float]], predicted_utility: List[Dict[str, float]], title: str, save_path: str = None, name_ext: str = 'NA') -> None:
    """
    Plot the Mean Absolute Percentage Error (MAPE) between true and predicted utility.
    """
    mape_values = []
    filtered_days_labels = []

    for day, (true_values, predicted_values) in enumerate(zip(true_utility, predicted_utility)):
        mape = calculate_mape(true_values, predicted_values)
        if mape is not None:
            mape_values.append(mape)
            filtered_days_labels.append(days_labels[day])
        else:
            mape_values.append(None)  # Appending None to maintain alignment with days_labels
    
    # Filter out None values from both lists together
    filtered_mape_values = [value for value in mape_values if value is not None]
    filtered_days_labels = [days_labels[i] for i in range(len(mape_values)) if mape_values[i] is not None]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_days_labels, filtered_mape_values, label='MAPE')
    plt.title(f'MAPE for {title}')
    plt.xlabel('Day and Week')
    plt.ylabel('MAPE (%)')
    plt.xticks(range(0, len(filtered_days_labels), max(1, len(filtered_days_labels) // 10)), rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    save_or_show_plot(f'MAPE_{title}_{name_ext}.png', save_path)


def print_preference_difference_and_accuracy(
    child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], 
    updated_preferences: Dict[str, Dict[str, List[str]]], 
    summary_only: bool = False
) -> Tuple[float, float, List[str], List[str]]:
    """
    Print the differences between actual and predicted preferences, calculate accuracy, and return lists of true and predicted labels.
    """
    total_actual = {'likes': 0, 'neutral': 0, 'dislikes': 0}
    total_correct = {'likes': 0, 'neutral': 0, 'dislikes': 0}

    def conditional_print(condition: bool, message: str) -> None:
        if condition:
            print(message)

    accuracies = []
    true_labels = []
    predicted_labels = []

    for child in child_preference_data:
        total_actual_child = {'likes': 0, 'neutral': 0, 'dislikes': 0}
        total_correct_child = {'likes': 0, 'neutral': 0, 'dislikes': 0}

        actual = {
            'likes': child_preference_data[child]['known']['likes'] + child_preference_data[child].get('unknown', {}).get('likes', []),
            'neutral': child_preference_data[child]['known']['neutral'] + child_preference_data[child].get('unknown', {}).get('neutral', []),
            'dislikes': child_preference_data[child]['known']['dislikes'] + child_preference_data[child].get('unknown', {}).get('dislikes', [])
        }

        predicted = {
            'likes': updated_preferences[child]['likes'],
            'neutral': updated_preferences[child]['neutral'],
            'dislikes': updated_preferences[child]['dislikes']
        }

        conditional_print(not summary_only, f"\nDifference for {child}:")

        for category in ['likes', 'neutral', 'dislikes']:
            for ingredient in actual[category]:
                total_actual[category] += 1
                total_actual_child[category] += 1

                # Accumulate true and predicted labels
                true_labels.append(category)
                if ingredient in predicted[category]:
                    predicted_labels.append(category)
                    total_correct[category] += 1
                    total_correct_child[category] += 1
                else:
                    if ingredient in predicted['neutral']:
                        predicted_labels.append('neutral')
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Neutral)")
                    elif ingredient in predicted['dislikes'] and category == 'likes':
                        predicted_labels.append('dislikes')
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Dislike)")
                    elif ingredient in predicted['likes'] and category == 'dislikes':
                        predicted_labels.append('likes')
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Like)")
                    else:
                        # If the ingredient doesn't match any predicted category, add a placeholder or skip
                        predicted_labels.append('unknown')

        child_total_actual = sum(total_actual_child.values())
        child_total_correct = sum(total_correct_child.values())
        if child_total_actual > 0:
            accuracy = child_total_correct / child_total_actual
            accuracies.append(accuracy)
            conditional_print(not summary_only, f"Accuracy for {child}: {accuracy:.2f}")
        else:
            accuracies.append(0)

    if len(accuracies) > 0:
        overall_accuracy = np.mean(accuracies)
        accuracy_std_dev = np.std(accuracies)
        conditional_print(True, f"\nOverall Accuracy of Preferences: {overall_accuracy:.6f}")
        conditional_print(True, f"Standard Deviation of Accuracies: {accuracy_std_dev:.6f}")
    else:
        overall_accuracy = 0
        accuracy_std_dev = 0
        conditional_print(True, "No data to calculate overall accuracy and standard deviation.")

    return overall_accuracy, accuracy_std_dev


def calculate_percent_of_known_ingredients_to_unknown(updated_true_preferences_with_feedback):
    
    percent_known = {}
    
    for child in updated_true_preferences_with_feedback:
        known = 0
        unknown = 0
        for category in ["likes", "neutral", "dislikes"]:
            known += len(updated_true_preferences_with_feedback[child]["known"][category])
            unknown += len(updated_true_preferences_with_feedback[child]["unknown"][category])

        total = known + unknown
        if total > 0:  # To avoid division by zero
            percent_known[child] = (known / total) * 100
        else:
            percent_known[child] = 0
    
    return percent_known
