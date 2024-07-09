import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
from models.preferences.data_utils import prepare_ml_data

def fit_random_forest_classifier(preferences, ingredients_data, child_data, apply_SMOTE=True, seed=None):
    
    # Prepare the data
    X, y, _, _ = prepare_ml_data(preferences, ingredients_data, child_data)
    
    if apply_SMOTE:
        # Convert sparse matrix to dense format
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        # Apply SMOTE - expanding the data to balance the classes
        smote = SMOTE(random_state=seed)
        X, y = smote.fit_resample(X_dense, y)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    return rf_model

def predict_preference_using_model(child_features, ingredient_features, preferences, model, preprocessor):
    # Create a dictionary with combined features
    combined_features = {
        "age": [child_features["age"]],
        "gender": [1 if child_features["gender"] == "M" else 0],
        "health_consideration": [child_features["health_consideration"]],
        "favorite_cuisine": [child_features["favorite_cuisine"]],
        "ingredient": [ingredient_features["ingredient"]],
        "type": [ingredient_features["type"]],
        "color": [ingredient_features["color"]],
        "taste": [ingredient_features["taste"]],
        "preference": [5 if ingredient_features["ingredient"] in preferences[child]["likes"] else
                       3 if ingredient_features["ingredient"] in preferences[child]["neutral"] else 1]
    }
    
    # Create a DataFrame from the combined features dictionary
    df = pd.DataFrame(combined_features)
    
    X_preprocessed = preprocessor.transform(df)

    # Predict using the model
    y_pred = model.predict(X_preprocessed)
    
    return y_pred[0]