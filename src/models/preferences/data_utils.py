import random
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import prince
import plotly.express as px
import random
from utils.process_data import get_data
from imblearn.over_sampling import SMOTE
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np
import random
from typing import Dict

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

    return (health_mod * favorite_mod * taste_mod * colour_mod * gender_mod * age_mod * 
            texture_mod * group_mod * fruit_mod * vegetable_mod * meat_mod)


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

def initialize_children_data(child_data: Dict[str, Dict[str, Any]], ingredient_df: pd.DataFrame, split: float = 0.8, seed: int = None, plot_graphs: bool = False) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, list]]]:
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


def get_feedback(preferences: Dict[str, Dict[str, list]], menu_plan: list, seed=None):
    # Function to get feedback on meal plan which gives randomized comments on the ingredients for each child.
    # The function also sometimes doesn't provide feedback for some children. 
    comments = [
        ("Didn't like the {} and {} in the dish, but the {} was tasty.", ["dislikes", "dislikes", "likes"]),
        ("Did not enjoy the {} and {}.", ["dislikes", "dislikes"]),
        ("Enjoyed the {} and {}, but was okay with the {}.", ["likes", "likes", "neutral"]),
        ("Loved the {}, but didn't like the {} and {}.", ["likes", "dislikes", "dislikes"]),
        ("The {} was great, but the {} was just okay.", ["likes", "neutral"]),
        ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
        ("Loved the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
        ("Loved the {}, but the {} was not appealing.", ["likes", "dislikes"]),
        ("Enjoyed the {}, but the {} was not liked.", ["likes", "dislikes"]),
        ("Didn't like the {}, {} and {} together.", ["dislikes", "dislikes", "dislikes"]),
        ("Really liked the {} with {} and the {} was tasty.", ["likes", "likes", "likes"]),
        ("Didn't like the {} in the dish, but the {} was fine.", ["dislikes", "neutral"]),
        ("Enjoyed the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
        ("Didn't like the {} and {}.", ["dislikes", "dislikes"]),
        ("The {} and {} were amazing, but didn't enjoy the {} much.", ["likes", "likes", "dislikes"]),
        ("Loved the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
        ("Didn't enjoy the {} much, but the {} was okay.", ["dislikes", "neutral"]),
        ("The {} and {} dish was great.", ["likes", "likes"]),
        ("Didn't like the {}.", ["dislikes"]),
        ("Enjoyed the {} and {}.", ["likes", "likes"]),
        ("Loved the {} and {}.", ["likes", "likes"]),
        ("Didn't like the {} and the {}.", ["dislikes", "dislikes"]),
        ("Enjoyed the {} and {}, but the {} was okay.", ["likes", "likes", "neutral"]),
        ("Didn't like the {} and {} in the dish.", ["dislikes", "dislikes"]),
        ("Didn't like the {}, but the {} was okay.", ["dislikes", "neutral"]),
        ("Enjoyed the {} and {}, but didn't like the {}.", ["likes", "likes", "dislikes"]),
        ("Didn't like the {}.", ["dislikes"]),
        ("Loved the {} and {}, but the {} was not liked.", ["likes", "likes", "dislikes"]),
        ("Didn't like the {}, but the {} was okay.", ["dislikes", "neutral"]),
        ("Enjoyed the {} and {}.", ["likes", "likes"]),
        ("Liked the {} but not the {}.", ["likes", "dislikes"]),
        ("The {} was fine, but the {} wasn't good.", ["neutral", "dislikes"]),
        ("The {} and {} were great, but the {} was not.", ["likes", "likes", "dislikes"]),
        ("The {} was tasty, but the {} wasn't.", ["likes", "dislikes"]),
        ("The {} was okay, but the {} wasn't appealing.", ["neutral", "dislikes"]),
        ("Didn't like the {}, but the {} was good.", ["dislikes", "likes"]),
        ("The {} and {} were okay, but the {} wasn't.", ["neutral", "neutral", "dislikes"]),
        ("Really liked the {}, but the {} was too strong.", ["likes", "dislikes"]),
        ("Enjoyed the {}, but the {} was too bland.", ["likes", "dislikes"]),
        ("The {} was fine, but the {} needed more flavor.", ["neutral", "dislikes"]),
        ("Loved the {}, but the {} was not good.", ["likes", "dislikes"]),
        ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
        ("The {} was good, but the {} was not to my taste.", ["likes", "dislikes"]),
        ("Enjoyed the {}, but the {} was too overpowering.", ["likes", "dislikes"]),
        ("The {} was delicious, but the {} wasn't enjoyable.", ["likes", "dislikes"])
    ]
    

    random.seed(seed)
    feedback = {}

    for child, prefs in preferences.items():  # Iterate over each child and their preferences
        
        # Combine known and unknown preferences for likes, neutral, and dislikes
        available_ingredients = {
            "likes": prefs['known']['likes'] + prefs['unknown']['likes'],
            "neutral": prefs['known']['neutral'] + prefs['unknown']['neutral'],
            "dislikes": prefs['known']['dislikes'] + prefs['unknown']['dislikes'],
        }

        valid_comments = []  # Initialize the list of valid comments

        # Iterate over each comment template and its corresponding feedback types
        for comment_template, feedback_types in comments:
            matched_ingredients = []  # Initialize the list of matched ingredients
            used_ingredients = set()  # Initialize the set of used ingredients

            # Match ingredients according to feedback types
            for feedback_type in feedback_types:
                for category in available_ingredients:
                    if feedback_type in category:  # Check if the feedback type matches the category
                        # List possible ingredients not yet used
                        possible_ingredients = [ingredient for ingredient in menu_plan if ingredient in available_ingredients[category] and ingredient not in used_ingredients]
                        if possible_ingredients:  # If there are possible ingredients
                            chosen_ingredient = random.choice(possible_ingredients)  # Randomly select an ingredient
                            matched_ingredients.append(chosen_ingredient)  # Add the chosen ingredient to the matched list
                            used_ingredients.add(chosen_ingredient)  # Mark the ingredient as used
                            break  # Break after finding a valid ingredient

            # Check if we have matched the required number of ingredients
            if len(matched_ingredients) == len(feedback_types):
                valid_comments.append((comment_template, matched_ingredients, feedback_types))  # Add to valid comments if matches found

        # Select a random valid comment from the list of valid comments
        if valid_comments:
            comment_template, matched_ingredients, feedback_types = random.choice(valid_comments)  # Randomly select a valid comment
            comment = comment_template.format(*matched_ingredients)  # Format the comment with matched ingredients
            correct_action = {ingredient: feedback_types[idx] for idx, ingredient in enumerate(matched_ingredients)}
            feedback[child] = {"comment": comment, "correct_action": correct_action}  # Add the comment to the child's feedback

    return feedback  # Return the feedback dictionary


def get_supplier_availability(ingredient_df, mean_unavailable=5, std_dev_unavailable=2, seed = None):
    
    ingredients = ingredient_df['Category7'].tolist()
    
    # Function to randomly generate supplier availability for ingredients
    random.seed(seed)
    
    # Determine the number of unavailable ingredients
    num_unavailable = max(0, int(np.random.normal(mean_unavailable, std_dev_unavailable)))
    unavailable_ingredients = random.sample(ingredients, num_unavailable)
    
    # Generate supplier availability
    supplier_availability = {ingredient: ingredient not in unavailable_ingredients for ingredient in ingredients}
    
    return supplier_availability


def get_data_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("age", OneHotEncoder(), ["age"]),
            ("gender", OneHotEncoder(), ["gender"]),
            ("health_consideration", OneHotEncoder(), ["health_consideration"]),
            ("favorite_cuisine", OneHotEncoder(), ["favorite_cuisine"]),
            ("type", OneHotEncoder(), ["type"]),
            ("colour", OneHotEncoder(), ["colour"]),
            ("taste", OneHotEncoder(), ["taste"]),
            ("texture", OneHotEncoder(), ["texture"]),
            ("healthy", OneHotEncoder(), ["healthy"]),
        ],
    )
    return preprocessor
    
 # Function to prepare data for the machine learning model
def prepare_ml_data(preferences, ingredient_df, child_data, apply_SMOTE=False, seed=42):
    # Create a DataFrame from child_data
    child_df = pd.DataFrame.from_dict(child_data, orient='index').reset_index().rename(columns={'index': 'child'})
    
    # Create a DataFrame from preferences
    preferences_df = pd.DataFrame(preferences).T.reset_index().rename(columns={'index': 'child'})
    
    # Melt preferences DataFrame
    likes_df = preferences_df.explode('likes')[['child', 'likes']].rename(columns={'likes': 'ingredient'})
    likes_df['preference'] = 5
    
    neutral_df = preferences_df.explode('neutral')[['child', 'neutral']].rename(columns={'neutral': 'ingredient'})
    neutral_df['preference'] = 3
    
    dislikes_df = preferences_df.explode('dislikes')[['child', 'dislikes']].rename(columns={'dislikes': 'ingredient'})
    dislikes_df['preference'] = 1
    
    unknown_df = preferences_df.explode('unknown')[['child', 'unknown']].rename(columns={'unknown': 'ingredient'})
    unknown_df['preference'] = np.nan
    
    # Concatenate all preference DataFrames
    preferences_long_df = pd.concat([likes_df, neutral_df, dislikes_df, unknown_df])
    
    # Merge child data with preferences
    combined_df = pd.merge(child_df, preferences_long_df, on='child')
    
    # Merge the combined DataFrame with the ingredient DataFrame
    df = pd.merge(combined_df, ingredient_df[['Category7', 'Category1', 'Colour', 'Texture', 'Taste', 'Healthy']], left_on='ingredient', right_on='Category7')

    # Drop the redundant 'Category7' column after the merge
    df.drop(columns=['Category7'], inplace=True)

    # Rename columns to match the desired output
    df.rename(columns={
        'ingredient': 'ingredient',
        'Category1': 'type',
        'Colour': 'colour',
        'Texture': 'texture',
        'Taste': 'taste',
        'Healthy': 'healthy'
    }, inplace=True)
    
    # Select and reorder the final columns
    df = df[['age', 'gender', 'health_consideration', 'favorite_cuisine', 'ingredient', 'type', 'colour', 'texture', 'taste', 'healthy', 'preference']]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    df["preference"] = label_encoder.fit_transform(df["preference"].astype(str))

    # Define the preprocessor for numerical and categorical features
    preprocessor = get_data_preprocessor()

    # Drop the target column before fitting the preprocessor
    X = df.drop(columns=['preference'])
    y = df["preference"].values

    # Fit the preprocessor
    preprocessor.fit(X)

    # Apply the transformations and prepare the dataset
    X_transformed = preprocessor.transform(X)

    # Drop rows with NaN preferences in the original DataFrame
    known_df = df.dropna(subset=['preference'])

    # Extract the indices of known preferences
    known_indices = known_df.index

    # Filter X_transformed and y to only include known preferences
    X = X_transformed[known_indices]
    y = y[known_indices]
    
    if apply_SMOTE:
        # Convert sparse matrix to dense format
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        # Apply SMOTE to balance the classes
        smote = SMOTE(random_state=seed)
        X_res, y_res = smote.fit_resample(X_dense, y)
        
        # Convert back to DataFrame
        X_res_df = pd.DataFrame(X_res, columns=preprocessor.get_feature_names_out())
        y_res_df = pd.DataFrame(y_res, columns=['preference'])
        
        # Concatenate X and y
        df_res = pd.concat([X_res_df, y_res_df], axis=1)
        
        # Update df, X, and y
        df = df_res
        X = X_res
        y = y_res

    return X, y, df, label_encoder, preprocessor


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
