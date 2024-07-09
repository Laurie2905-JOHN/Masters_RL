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

def get_child_data():
    # Function to get feature data on children
    return {
        "child1": {"age": 10, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child2": {"age": 9, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child3": {"age": 9, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child4": {"age": 7, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child5": {"age": 11, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child6": {"age": 11, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child7": {"age": 9, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child8": {"age": 9, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child9": {"age": 10, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child10": {"age": 11, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child11": {"age": 7, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "Italian"},
        "child12": {"age": 9, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child13": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child14": {"age": 10, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "Chinese"},
        "child15": {"age": 11, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child16": {"age": 11, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child17": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"},
        "child18": {"age": 7, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child19": {"age": 9, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child20": {"age": 10, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "British"},
        "child21": {"age": 10, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "British"},
        "child22": {"age": 7, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "British"},
        "child23": {"age": 9, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "British"},
        "child24": {"age": 9, "gender": "M", "health_consideration": "don't care", "favorite_cuisine": "British"},
        "child25": {"age": 11, "gender": "F", "health_consideration": "very health conscious", "favorite_cuisine": "British"},
        "child26": {"age": 11, "gender": "M", "health_consideration": "moderately health conscious", "favorite_cuisine": "British"},
        "child27": {"age": 9, "gender": "F", "health_consideration": "moderately health conscious", "favorite_cuisine": "Chinese"},
        "child28": {"age": 7, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Chinese"},
        "child29": {"age": 9, "gender": "M", "health_consideration": "very health conscious", "favorite_cuisine": "Italian"},
        "child30": {"age": 10, "gender": "F", "health_consideration": "don't care", "favorite_cuisine": "Italian"}
    }

import random
from utils.process_data import get_data

def initialize_children_data(child_features, data_file="data.csv", seed=None):
    random.seed(seed)
    children_data = {}
    
    # Load the ingredient data
    ingredient_df = get_data(data_file)
    
    # Base probabilities for like, neutral, and dislike
    base_probabilities = {"like": 0.3, "neutral": 0.5, "dislike": 0.2}
    
    # Factors affecting preferences with modifier values
    health_consideration_modifiers = {
        "very health conscious": {"healthy": 1.2, "average": 1, "unhealthy": 0.8},
        "moderately health conscious": {"healthy": 1.1, "average": 1, "unhealthy": 0.9},
        "don't care": {"healthy": 0.8, "average": 1, "unhealthy": 1.2},
    }
    
    favorite_cuisine_modifiers = {
        "BBQ": {"Meat and meat products": 1.2},
        "Seafood": {"Fish seafood amphibians reptiles and invertebrates": 1.2},
        "Italian": {"Anchovies": 1.2, "Aubergines": 1.2, "Noodles": 1.2, "Pasta plain (not stuffed) uncooked": 1.2, "Pasta wholemeal": 1.2, "Tomatoes": 1.2},
    }
    
    taste_modifiers = {
        "Sweet": 1.1,
        "Salty": 1.1,
        "Sour": 0.8,
        "Earthy": 0.8,
        "Misc": 1,
    }
    
    colour_modifiers = {
        "Red": 1.1,
        "Green": 1.1,
        "Yellow": 1.05,
        "Orange": 1.05,
        "Pink": 1,
        "Purple": 1,
        "White": 0.95,
        "Brown": 0.95,
    }
    
    gender_modifiers = {
        "M": 0.9,
        "F": 1.1,
    }
    
    age_modifiers = {
        9: 0.9,
        10: 1,
        11: 1.1,
    }
    
    texture_modifiers = {
        "Crunchy": 0.9,
        "Soft": 1.1,
        "Soft/Crunchy": 0.8,
        "Firm": 1.1,
        "Leafy": 1,
        "Grainy": 1,
        "Liquid": 1,
        "Powdery": 1,
        "Creamy": 1,
        "Hard": 1,
    }
    
    other_modifiers = {
        "fruit_factor": 1.1,
        "vegetables_factor": {"M": 0.9, "F": 1},
        "meat_factor": {"M": 1, "F": 0.9},
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
        "Group A": {"like": 1.2, "neutral": 0.8, "dislike": 0.7},  # Most liked
        "Group B": {"like": 1.1, "neutral": 1, "dislike": 0.9},  # Camouflaged in recipes
        "Group C": {"like": 0.8, "neutral": 1, "dislike": 1.2},  # Strongly disliked
        "Group D": {"like": 1, "neutral": 1, "dislike": 1},      # Camouflaged but some forced
        "Group E": {"like": 0.9, "neutral": 1, "dislike": 1.1},  # Offered but often rejected
        "Group F": {"like": 0.9, "neutral": 1, "dislike": 1.1}   # Rarely offered, often rejected
    }
    

    # Initialize preferences for each child
    for child_key, features in child_features.items():
        # Extract child-specific features
        health_consideration = features["health_consideration"]
        age = features["age"]
        gender = features["gender"]
        favorite_cuisine = features["favorite_cuisine"]

        # Get the modifier values for the child's health consideration, age, and gender
        health_modifiers = health_consideration_modifiers[health_consideration]
        age_modifier = age_modifiers[age]
        gender_modifier = gender_modifiers[gender]

        preferences = {"likes": [], "neutral": [], "dislikes": []}

        for i, row in ingredient_df.iterrows():
            ingredient = row["Category7"]
            health_category = row["Healthy"]

            # Calculate modifiers based on various factors
            health_mod = health_modifiers[health_category]
            favorite_mod = favorite_cuisine_modifiers.get(favorite_cuisine, {}).get(row["Category1"], 1)
            taste_mod = taste_modifiers.get(row["Taste"], taste_modifiers["Misc"])
            colour_mod = colour_modifiers[row["Colour"]]
            texture_mod = texture_modifiers[row["Texture"]]

            # Determine the vegetable group and get the group probability modifiers
            group_name = next((group for group, ingredients in vegetable_groups.items() if ingredient in ingredients), None)
            group_modifiers = group_probabilities_modifiers[group_name] if group_name else {"like": 1, "neutral": 1, "dislike": 1}
            
            # Calculate other modifiers
            fruit_mod = other_modifiers["fruit_factor"] if row["Category1"] == "Fruits and fruit products" else 1
            vegetable_mod = other_modifiers["vegetables_factor"][gender] if row["Category1"] == "Vegetables and vegetable products" else 1
            meat_mod = other_modifiers["meat_factor"][gender] if row["Category1"] == "Meat and meat products" else 1

            # Calculate the overall probabilities by applying modifiers to the base probabilities
            like_probability = base_probabilities["like"] * health_mod * favorite_mod * taste_mod * colour_mod * gender_modifier * age_modifier * texture_mod * group_modifiers["like"] * fruit_mod * vegetable_mod * meat_mod
            neutral_probability = base_probabilities["neutral"] * health_mod * favorite_mod * taste_mod * colour_mod * gender_modifier * age_modifier * texture_mod * group_modifiers["neutral"] * fruit_mod * vegetable_mod * meat_mod
            dislike_probability = base_probabilities["dislike"] * health_mod * favorite_mod * taste_mod * colour_mod * gender_modifier * age_modifier * texture_mod * group_modifiers["dislike"] * fruit_mod * vegetable_mod * meat_mod
            
            # Normalize the probabilities
            total = like_probability + neutral_probability + dislike_probability
            like_probability /= total
            neutral_probability /= total
            dislike_probability /= total

            # Determine the preference based on the calculated probabilities
            rand_val = random.random()
            if rand_val < like_probability:
                preferences["likes"].append(ingredient)
            elif rand_val < like_probability + neutral_probability:
                preferences["neutral"].append(ingredient)
            else:
                preferences["dislikes"].append(ingredient)

        children_data[child_key] = preferences

    return children_data


def initialize_children_data(child_features, seed=None):
    # Function to get initial preference data on children, based on their health consideration and vegetable groups.
    # Following this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173934/ 
    # Define vegetable groups with descriptions
    vegetable_groups = {
        "Group A": ["Tomatoes", "Sweet corn", "Sweet potatoes", "Carrots"],
        "Group B": ["Onions", "Spring onions", "Pepper"],
        "Group C": ["Cauliflowers"],
        "Group D": ["Courgettes", "Spinaches", "Curly kales", "Peas"],
        "Group E": ["Beetroots", "Lettuces (generic)", "Broccoli"],
        "Group F": ["Aubergines", "Cucumber", "White cabbage", "Savoy cabbages", "Red cabbage", "Runner beans (with pods)"]
    }

    # Define probabilities based on group descriptions
    probabilities = {
        "Group A": {"like": 0.7, "neutral": 0.2, "dislike": 0.1},  # Most liked
        "Group B": {"like": 0.4, "neutral": 0.3, "dislike": 0.3},  # Camouflaged in recipes
        "Group C": {"like": 0.1, "neutral": 0.2, "dislike": 0.7},  # Strongly disliked
        "Group D": {"like": 0.3, "neutral": 0.3, "dislike": 0.4},  # Camouflaged but some forced
        "Group E": {"like": 0.2, "neutral": 0.3, "dislike": 0.5},  # Offered but often rejected
        "Group F": {"like": 0.2, "neutral": 0.3, "dislike": 0.5}   # Rarely offered, often rejected
    }
    
    random.seed(seed)

    children_data = {}

# Iterate over each child in the dataset
    for child_key, features in child_features.items():
        # Get the health consideration feature for the current child
        health_consideration = features["health_consideration"]
        
        # Calculate like probability modifier based on health consideration
        # More health conscious children have a higher modifier, making them less likely to dislike vegetables
        if health_consideration == "very health conscious":
            like_modifier = 0.2
        elif health_consideration == "moderately health conscious":
            like_modifier = 0.1
        else:
            like_modifier = 0

        # Initialize the preferences for the current child
        preferences = {"likes": [], "neutral": [], "dislikes": []}

        # Assign preferences based on group probabilities and health consideration
        for group, veggies in vegetable_groups.items():
            for veggie in veggies:
                # Generate a random value between 0 and 1
                rand_val = random.random()
                # Get the probability distribution for the current group
                prob = probabilities[group]
                
                # Determine preference based on probabilities and health consideration modifier
                if rand_val < prob["like"] - like_modifier:
                    preferences["likes"].append(veggie)
                elif rand_val < prob["like"] + prob["neutral"]:
                    preferences["neutral"].append(veggie)
                else:
                    preferences["dislikes"].append(veggie)
        # Store the preferences for the current child
        children_data[child_key] = preferences
    
    return children_data


def get_feedback(ingredient_list, mean_no_feedback=9, std_dev_no_feedback=4, seed=None):
    # Function to get feedback on meal plan which gives randomised comments on the ingredients for each child.
    # The function also sometimes doesn't provide feedback for some children. 
    comments = [
        "Didn't like the {} and {} in the dish, but the {} was tasty.",
        "Did not enjoy the {} and {}.",
        "Enjoyed the {} and {}, but was okay with the {}.",
        "Loved the {}, but didn't like the {} and {}.",
        "The {} was great, but the {} were just okay.",
        "Didn't enjoy the {}, but the {} and {} were okay.",
        "Loved the {} and {}, but not the {}.",
        "Loved the {} and {}, but the {} was not appealing.",
        "Enjoyed the {}, but the {} was not liked.",
        "Didn't like the {}, {} and {} together.",
        "Really liked the {} with {} and the {} was tasty.",
        "Didn't like the {} in the dish, but the {} was fine.",
        "Enjoyed the {} and {}, but not the {}.",
        "Didn't like the {} and {}.",
        "The {} and {} were amazing, but didn't enjoy the {} much.",
        "Loved the {} and {}, but not the {}.",
        "Didn't enjoy the {} much, but the {} were okay.",
        "The {} and {} dish was great.",
        "Didn't like the {}.",
        "Enjoyed the {} and {}.",
        "Loved the {} and {}.",
        "Didn't like the {} and the {}.",
        "Enjoyed the {} and {}, but the {} was okay.",
        "Didn't like the {} and {} in the dish.",
        "Didn't like the {} and {}, but the {} were okay.",
        "Enjoyed the {} and {}, but didn't like the {}.",
        "Didn't like the {}.",
        "Loved the {} and {}, but the {} was not liked.",
        "Didn't like the {} much and the {} were okay.",
        "Enjoyed the {} and {}.",
        "Liked the {} but not the {}.",
        "The {} was fine, but the {} and {} weren't good.",
        "The {} and {} were great, but the {} was not.",
        "The {} was tasty, but the {} and {} weren't.",
        "The {} was okay, but the {} and {} weren't appealing.",
        "Didn't like the {}, but the {} and {} were good.",
        "The {} and {} were okay, but the {} wasn't.",
        "Really liked the {}, but the {} was too strong.",
        "Enjoyed the {}, but the {} and {} were too bland.",
        "The {} was fine, but the {} and {} needed more flavor.",
        "Loved the {}, but the {} and {} were not good.",
        "Didn't enjoy the {}, but the {} was okay.",
        "The {} was good, but the {} and {} were not to my taste.",
        "Enjoyed the {}, but the {} was too overpowering.",
        "The {} was delicious, but the {} and {} weren't enjoyable."
    ]
    
    children = [f"child{i}" for i in range(1, 31)]
    random.seed(seed)
    random.shuffle(children)
    
    
    # Determine the number of children who will not give feedback
    num_children_no_feedback = max(0, int(np.random.normal(mean_no_feedback, std_dev_no_feedback)))
    children_no_feedback = random.sample(children, num_children_no_feedback)
    
    feedback = {}
    for child in children:
        if child in children_no_feedback:
            continue
        comment_template = random.choice(comments)
        ingredient_list = [random.choice(ingredient_list) for _ in range(comment_template.count("{}"))]
        comment = comment_template.format(*ingredient_list)
        feedback[child] = {"comments": comment}
    
    return feedback


def get_supplier_availability(ingredients, mean_unavailable=5, std_dev_unavailable=2, seed = None):
    
    # Function to randomly generate supplier availability for ingredients
    random.seed(seed)
    
    # Determine the number of unavailable ingredients
    num_unavailable = max(0, int(np.random.normal(mean_unavailable, std_dev_unavailable)))
    unavailable_ingredients = random.sample(ingredients, num_unavailable)
    
    # Generate supplier availability
    supplier_availability = {ingredient: ingredient not in unavailable_ingredients for ingredient in ingredients}
    
    return supplier_availability

# Function to convert to dense and DataFrame
def to_dense_dataframe(X):
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    return pd.DataFrame(X_dense)

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
        ]
    )
    return preprocessor
    
# Function to prepare data for the machine learning model
def prepare_ml_data(preferences, ingredient_df, child_data):
    
    # Create a DataFrame from child_data
    child_df = pd.DataFrame.from_dict(child_data, orient='index').reset_index().rename(columns={'index': 'child'})
    
    # Encode the gender column
    label_encoder_gender = LabelEncoder()
    child_df["gender"] = label_encoder_gender.fit_transform(child_df["gender"])
    
    # Create a DataFrame from preferences
    preferences_df = pd.DataFrame(preferences).T.reset_index().rename(columns={'index': 'child'})
    
    # Melt preferences DataFrame
    likes_df = preferences_df.explode('likes')[['child', 'likes']].rename(columns={'likes': 'ingredient'})
    likes_df['preference'] = 5
    
    neutral_df = preferences_df.explode('neutral')[['child', 'neutral']].rename(columns={'neutral': 'ingredient'})
    neutral_df['preference'] = 3
    
    dislikes_df = preferences_df.explode('dislikes')[['child', 'dislikes']].rename(columns={'dislikes': 'ingredient'})
    dislikes_df['preference'] = 1
    
    # Concatenate all preference DataFrames
    preferences_long_df = pd.concat([likes_df, neutral_df, dislikes_df])
    
    # Merge child data with preferences
    combined_df = pd.merge(child_df, preferences_long_df, on='child')
    
    # Merge the combined DataFrame with the ingredient DataFrame
    df = pd.merge(combined_df, ingredient_df[['Category7', 'Category1', 'Colour', 'Texture', 'Taste']], left_on='ingredient', right_on='Category7')
    
    # Rename columns to match the desired output
    df.rename(columns={
        'Category7': 'ingredient',
        'Category1': 'type',
        'Colour': 'colour',
        'Texture': 'texture',
        'Taste': 'taste'
    }, inplace=True)
    
    # Select and reorder the final columns
    df = df[['age', 'gender', 'health_consideration', 'favorite_cuisine', 'ingredient', 'type', 'colour', 'texture', 'taste', 'preference']]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    
    df["preference"] = label_encoder.fit_transform(df["preference"])

    # Define the preprocessor for numerical and categorical features
    preprocessor = get_data_preprocessor()

    # Fit the preprocessor
    preprocessor = preprocessor.fit(df)

    # Apply the transformations and prepare the dataset
    X = preprocessor.transform(df)
    y = df["preference"].values
    
    return X, y, label_encoder, preprocessor


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
