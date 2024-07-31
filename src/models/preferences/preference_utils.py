import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random
from typing import Dict, Any, Tuple, List, Callable
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d

def get_child_data():
    # Function to get feature data on children
    return {
        "child1": {"age": 10, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child2": {"age": 9, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child3": {"age": 9, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child4": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child5": {"age": 11, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child6": {"age": 11, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child7": {"age": 9, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child8": {"age": 9, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child9": {"age": 10, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child10": {"age": 11, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child11": {"age": 9, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child12": {"age": 9, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child13": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child14": {"age": 10, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Seafood"},
        "child15": {"age": 11, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child16": {"age": 11, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child17": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child18": {"age": 9, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child19": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child20": {"age": 10, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "BBQ"},
        "child21": {"age": 10, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "BBQ"},
        "child22": {"age": 9, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "BBQ"},
        "child23": {"age": 9, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "BBQ"},
        "child24": {"age": 9, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "BBQ"},
        "child25": {"age": 11, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "BBQ"},
        "child26": {"age": 11, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "BBQ"},
        "child27": {"age": 9, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "Seafood"},
        "child28": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Seafood"},
        "child29": {"age": 9, "gender": "M", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child30": {"age": 10, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"}
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
            texture_mod * group_mod * fruit_mod * vegetable_mod * meat_mod * random_mod)

def initialize_child_preference_data(child_data: Dict[str, Dict[str, Any]], ingredient_df: pd.DataFrame, split: float = 0.8, seed: int = None, plot_graphs: bool = False) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, list]]]:
    random.seed(seed)
    children_data = {}
    all_scores = []
    all_preferences = {"likes": [], "neutral": [], "dislikes": []}

    # Factors affecting preferences with modifier values (increased impact)
    health_consideration_modifiers = {
        "very health conscious": {"healthy": 1.5, "average": 1, "unhealthy": 0.5},
        "moderately health conscious": {"healthy": 1.3, "average": 1, "unhealthy": 0.7},
        "don't care": {"healthy": 0.7, "average": 1, "unhealthy": 1.3},
    }

    favorite_cuisine_modifiers = {
        "BBQ": {"Meat and meat products": 1.4},
        "Seafood": {"Fish seafood amphibians reptiles and invertebrates": 1.4},
        "Italian": {"Anchovies": 1.4, "Aubergines": 1.4, "Noodles": 1.4, "Pasta plain (not stuffed) uncooked": 1.4, "Pasta wholemeal": 1.4, "Tomatoes": 1.4},
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
        "M": 0.7,
        "F": 1.3,
    }

    age_modifiers = {
        9: 0.7,
        10: 1,
        11: 1.3,
    }

    texture_modifiers = {
        "Crunchy": 0.7,
        "Soft": 1.3,
        "Soft/Crunchy": 0.6,
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
        "random_factor": [0.7, 1.3]
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

    for child_key, features in child_data.items():
        preferences = {"likes": [], "neutral": [], "dislikes": []}
        child_scores = []

        for _, row in ingredient_df.iterrows():
            score = get_modifiers(features, row, health_consideration_modifiers, favorite_cuisine_modifiers, taste_modifiers,
                                  colour_modifiers, gender_modifiers, age_modifiers, texture_modifiers, other_modifiers,
                                  vegetable_groups, group_probabilities_modifiers)
            child_scores.append((row["Category7"], score))

        child_scores.sort(key=lambda x: x[1], reverse=True)
        all_scores.extend(child_scores)

        num_ingredients = len(child_scores)
        num_likes = int(0.6 * num_ingredients)
        num_neutral = int(0.2 * num_ingredients)
        num_dislikes = num_ingredients - num_likes - num_neutral

        preferences["likes"] = [ingredient for ingredient, _ in child_scores[:num_likes]]
        preferences["neutral"] = [ingredient for ingredient, _ in child_scores[num_likes:num_likes + num_neutral]]
        preferences["dislikes"] = [ingredient for ingredient, _ in child_scores[num_likes + num_neutral:]]

        all_preferences["likes"].extend(preferences["likes"])
        all_preferences["neutral"].extend(preferences["neutral"])
        all_preferences["dislikes"].extend(preferences["dislikes"])

        children_data[child_key] = preferences

    all_data = {}

    for child_key, preferences in children_data.items():
        known_preferences = {"likes": [], "neutral": [], "dislikes": []}
        unknown_preferences = {"likes": [], "neutral": [], "dislikes": []}

        for category in ["likes", "neutral", "dislikes"]:
            total_items = len(preferences[category])
            split_index = int(total_items * split)
            known_preferences[category] = preferences[category][:split_index]
            unknown_preferences[category] = preferences[category][split_index:]

        all_data[child_key] = {
            "known": known_preferences,
            "unknown": unknown_preferences
        }

    if plot_graphs:
        plot_histograms(all_scores, all_preferences)

    return all_data

from sklearn.preprocessing import MinMaxScaler

def create_preference_score_function(negotiated: Dict[str, Dict[str, float]], unavailable: List[str]) -> Callable[[str], float]:
    """
    Create a preference score function based on negotiated ingredients, excluding unavailable ingredients.

    :param negotiated: Dictionary of negotiated ingredients and their scores.
    :param unavailable: List of unavailable ingredients.
    :return: Function that returns the score for a given ingredient.
    """
    ingredient_scores = {}
    if negotiated:
        for group, ingredients in negotiated.items():
            # Filter out unavailable ingredients
            available_ingredients = {k: v for k, v in ingredients.items() if k not in unavailable}
            
            if available_ingredients:
                scores = np.array(list(available_ingredients.values())).reshape(-1, 1)
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(scores).flatten()
                for ingredient, norm_score in zip(available_ingredients.keys(), normalized_scores):
                    ingredient_scores[ingredient] = norm_score

    def score_function(ingredient: str) -> float:
        if ingredient not in ingredient_scores.keys():
            raise ValueError(f"Ingredient '{ingredient}' not found in negotiated ingredients.")
        else:
            return ingredient_scores[ingredient]

    return score_function
    
def plot_histograms(scores: list, preferences: Dict[str, list]) -> None:
    total_ingredients = len(scores)
    likes_count = len(preferences["likes"])
    neutral_count = len(preferences["neutral"])
    dislikes_count = len(preferences["dislikes"])

    likes_percent = (likes_count / total_ingredients) * 100
    neutral_percent = (neutral_count / total_ingredients) * 100
    dislikes_percent = (dislikes_count / total_ingredients) * 100

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["likes"]], bins=20, color='green', alpha=0.7, label='Like')
    plt.title(f'Like Scores ({likes_percent:.2f}%)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["neutral"]], bins=20, color='blue', alpha=0.7, label='Neutral')
    plt.title(f'Neutral Scores ({neutral_percent:.2f}%)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["dislikes"]], bins=20, color='red', alpha=0.7, label='Dislike')
    plt.title(f'Dislike Scores ({dislikes_percent:.2f}%)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_preference_and_sentiment_accuracies(prediction_accuracies, prediction_std_devs, sentiment_accuracies, iterations, save_path):
    """Function to plot the accuracies and standard deviations over iterations and save the plot to a specified file path."""
    plt.figure(figsize=(18, 6))

    # Plot Prediction Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(range(1, iterations + 2), prediction_accuracies, marker='o', linestyle='-', label='Prediction Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy per Iteration')
    plt.legend()
    plt.grid(True)

    # Plot Prediction Standard Deviation
    plt.subplot(1, 3, 2)
    plt.plot(range(1, iterations + 2), prediction_std_devs, marker='o', linestyle='-', color='green', label='Prediction Std Dev')
    plt.xlabel('Iteration')
    plt.ylabel('Standard Deviation')
    plt.title('Prediction Standard Deviation per Iteration')
    plt.legend()
    plt.grid(True)

    # Plot Sentiment Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(range(1, iterations + 2), sentiment_accuracies, marker='o', linestyle='-', color='orange', label='Sentiment Accuracy')
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
    children = list(child_known_percents[0].keys())
    known_per_child = {child: [] for child in children}

    for iteration_data in child_known_percents:
        for child, percent in iteration_data.items():
            known_per_child[child].append(percent)

    # Select 5 random children
    selected_children = random.sample(children, 5)
    
    # Plot percent for each child
    for child in selected_children:
        percent = known_per_child[child]
        plt.plot(range(1, iterations + 1), percent, marker='o', label=f'Child {child}')

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
        save_or_show_plot(f'{utility}_per_child.png', save_path)

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
    save_or_show_plot('daily_gini_coefficients.png', save_path)

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
    save_or_show_plot('daily_gini_coefficients.png')

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
    save_or_show_plot('cumulative_gini_coefficients.png')
    

def calculate_mape(true_values: Dict[str, float], predicted_values: Dict[str, float]) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.
    """
    mape = 100 * sum(abs(true_values[child] - predicted_values[child]) / true_values[child] for child in true_values) / len(true_values)
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
        save_or_show_plot(f'{sum_utility}_per_day.png', save_path)

    # Calculate and plot accuracy (MAPE) between true and predicted utility cumulative sums
    plot_mape(all_days, true_utility_per_day, predicted_utility_per_day, title, save_path)

def plot_mape(days_labels: List[str], true_utility: List[Dict[str, float]], predicted_utility: List[Dict[str, float]], title: str, save_path: str = None) -> None:
    """
    Plot the Mean Absolute Percentage Error (MAPE) between true and predicted utility.
    """
    mape_values = []

    for true_values, predicted_values in zip(true_utility, predicted_utility):
        mape = calculate_mape(true_values, predicted_values)
        mape_values.append(mape)

    plt.figure(figsize=(12, 6))
    plt.plot(days_labels, mape_values, label='MAPE')
    plt.title(f'MAPE for {title}')
    plt.xlabel('Day and Week')
    plt.ylabel('MAPE (%)')
    plt.xticks(range(0, len(days_labels), max(1, len(days_labels) // 10)), rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    save_or_show_plot(f'MAPE_{title}.png', save_path)


def print_preference_difference_and_accuracy(child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], updated_preferences: Dict[str, Dict[str, List[str]]], summary_only: bool = False) -> None:
    """
    Print the differences between actual and predicted preferences, and calculate accuracy.
    """
    total_actual = {'likes': 0, 'neutral': 0, 'dislikes': 0}
    total_correct = {'likes': 0, 'neutral': 0, 'dislikes': 0}

    def conditional_print(condition: bool, message: str) -> None:
        if condition:
            print(message)

    accuracies = []

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
            conditional_print(not summary_only, f"Actual {category.capitalize()} but Predicted Differently:")
            for ingredient in actual[category]:
                total_actual[category] += 1
                total_actual_child[category] += 1
                if ingredient in predicted[category]:
                    total_correct[category] += 1
                    total_correct_child[category] += 1
                else:
                    if ingredient in predicted['neutral'] and category != 'neutral':
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Neutral)")
                    elif ingredient in predicted['dislikes'] and category == 'likes':
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Dislike)")
                    elif ingredient in predicted['likes'] and category == 'dislikes':
                        conditional_print(not summary_only, f"  {ingredient} (Actual: {category.capitalize()}, Predicted: Like)")

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
        return overall_accuracy, accuracy_std_dev
    else:
        conditional_print(True, "No data to calculate overall accuracy and standard deviation.")


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
