import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Tuple, List
import random
import numpy as np

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

def plot_histograms(scores: list, preferences: Dict[str, list]) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["likes"]], bins=20, color='green', alpha=0.7, label='Like')
    plt.title('Like Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["neutral"]], bins=20, color='blue', alpha=0.7, label='Neutral')
    plt.title('Neutral Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist([score for ingredient, score in scores if ingredient in preferences["dislikes"]], bins=20, color='red', alpha=0.7, label='Dislike')
    plt.title('Dislike Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to plot 2D MCA components
def plot_2d_mca(X, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
    plt.xlabel('MCA Component 1')
    plt.ylabel('MCA Component 2')
    plt.title('2D MCA')
    plt.legend(*scatter.legend_elements(), title="Preferences")
    plt.show()

# Function to plot 3D MCA components
def plot_3d_mca(X, y):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('MCA Component 1')
    ax.set_ylabel('MCA Component 2')
    ax.set_zlabel('MCA Component 3')
    ax.set_title('3D MCA')
    legend1 = ax.legend(*scatter.legend_elements(), title="Preferences")
    ax.add_artist(legend1)
    plt.show()

# Function to plot scree plot for MCA
def plot_scree_mca(preprocessor, n_components=10):
    # Extract the MCA step from the pipeline
    mca = preprocessor.named_steps['mca']
    
    # Plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), mca.explained_inertia_, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('MCA Component')
    plt.ylabel('Explained Inertia Ratio')
    plt.xticks(range(1, n_components + 1))
    plt.grid()
    plt.show()

# Function to plot 3D MCA components interactively
def plot_3d_mca_interactive(X, y):
    # Ensure we only take the first three components
    if X.shape[1] < 3:
        raise ValueError("Input data must have at least 3 principal components for a 3D plot.")
    
    # Create a DataFrame with the first three MCA components and the labels
    df = X.iloc[:, :3]
    df.columns = ['MC1', 'MC2', 'MC3']  # Rename columns for plotting
    df['Preference'] = y
    
    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(df, x='MC1', y='MC2', z='MC3', color='Preference', 
                        title='3D MCA Interactive Plot', labels={'Preference': 'Preference'})
    fig.show()



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
    
    