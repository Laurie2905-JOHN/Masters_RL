import logging
from models.preferences.data_utils import get_child_data, initialize_child_preference_data, prepare_ml_data, prepare_ml_preferences
from utils.process_data import get_data
from models.preferences.prediction import fit_random_forest_classifier, predict_preference_using_model, combine_features, get_true_preference_label, print_preference_difference_and_accuracy
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from models.preferences.voting import negotiate_ingredients_simple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(seed=4):
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    child_preference_data = initialize_child_preference_data(child_feature_data, ingredient_df, split=0.5, seed=seed, plot_graphs=False)
    ml_child_preferences = prepare_ml_preferences(child_preference_data, ingredient_df)
    
    X, y, known_df, label_encoder, preprocessor = prepare_ml_data(ml_child_preferences, child_feature_data, ingredient_df, apply_SMOTE=True, seed=seed)
    rf_model = fit_random_forest_classifier(X, y)
    
    total_predicted_preferences = []
    total_true_preferences = []

    # Data structure to hold individual child predictions and true labels
    child_predictions = {}
    child_true_labels = {}

    batch_data = []
    batch_labels = []
    child_indices = []

    for child, preferences in ml_child_preferences.items():
        child_predictions[child] = []
        child_true_labels[child] = []
        for ingredient in preferences['unknown']:
            child_and_ingredient_feature_df = combine_features(ingredient_df, ingredient, child_feature_data[child])
            batch_data.append(child_and_ingredient_feature_df)
            true_label = get_true_preference_label(ingredient, child_preference_data, child)
            batch_labels.append(true_label)
            child_indices.append(child)
    
    if batch_data:
        batch_df = pd.concat(batch_data, ignore_index=True)  # Ensure concatenation and reset index
        batch_predictions = predict_preference_using_model(batch_df, rf_model, preprocessor)
        
        # Ensure batch_predictions is iterable
        if not isinstance(batch_predictions, (list, pd.Series, np.ndarray)):
            batch_predictions = [batch_predictions]

        for i, predicted_preference in enumerate(batch_predictions):
            child = child_indices[i]
            child_predictions[child].append(predicted_preference)
            child_true_labels[child].append(batch_labels[i])
            total_predicted_preferences.append(predicted_preference)
            total_true_preferences.append(batch_labels[i])
    
    # Log individual classification reports
    for child in child_predictions:
        if child_predictions[child]:
            child_class_report = classification_report(child_true_labels[child], child_predictions[child])
            logging.info(f"{child} Classification Report:\n{child_class_report}")

    # Log total classification report
    total_class_report = classification_report(total_true_preferences, total_predicted_preferences)
    logging.info(f"Total Classification Report:\n{total_class_report}")

    # Update the preference list with the predictions
    updated_known_and_predicted_preferences = {}
    for child, preferences in ml_child_preferences.items():
        
        predicted_likes = {
            ingredient: 'like' if pred == 2 else ('neutral' if pred == 0 else 'dislike')
            for ingredient, pred in zip(preferences['unknown'], child_predictions[child])
        }

        # Initialize the child's dictionary
        if child not in updated_known_and_predicted_preferences:
            updated_known_and_predicted_preferences[child] = {
                'likes': preferences.get('likes', []),
                'neutral': preferences.get('neutral', []),
                'dislikes': preferences.get('dislikes', [])
            }
        
        # Update the preferences based on predictions
        for ingredient, pred in zip(preferences['unknown'], child_predictions[child]):
            if pred == 2:
                updated_known_and_predicted_preferences[child]['likes'].append(ingredient)
            elif pred == 0:
                updated_known_and_predicted_preferences[child]['neutral'].append(ingredient)
            else:
                updated_known_and_predicted_preferences[child]['dislikes'].append(ingredient)

    # See difference in actual and predicted preferences
    print_preference_difference_and_accuracy(child_preference_data, updated_known_and_predicted_preferences, summary_only=True)


    # Negotiate the ingredients
    # Simple
    negotiated_ingredient_order, unavailable_ingredients = negotiate_ingredients_simple(updated_known_and_predicted_preferences, ingredient_df, seed)
    sum1 = 0
    for key in negotiated_ingredient_order.values():
        print(key)
        sum1 += len(key)
   
            
    print(negotiated_ingredient_order)
    print(sum1)
    print(unavailable_ingredients)
if __name__ == "__main__":
    main()
