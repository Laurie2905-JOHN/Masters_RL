import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fit_random_forest_classifier(X, y):
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    
    return rf_model

def combine_features(ingredient_df, ingredient, child_feature_data):
    
    ingredient_details = ingredient_df[ingredient_df['Category7'] == ingredient].iloc[0]
    
    ingredient_features = {
    "ingredient": ingredient,
    "type": ingredient_details["Category1"],
    "texture": ingredient_details["Texture"],
    "colour": ingredient_details["Colour"],
    "taste": ingredient_details["Taste"],
    "healthy": ingredient_details["Healthy"]
    }

    child_features = {
        "age": child_feature_data["age"],
        "gender": child_feature_data["gender"],
        "health_consideration": child_feature_data["health_consideration"],
        "favorite_cuisine": child_feature_data["favorite_cuisine"]
    }
        
    # Combine the features from the child and ingredient
    combined_features =  child_features | ingredient_features 
    
    # Create a DataFrame from the combined features dictionary
    df = pd.DataFrame([combined_features])
    
    return df

def predict_preference_using_model(df, model, preprocessor):
    X_preprocessed = preprocessor.transform(df)
    
    # Predict using the model
    y_pred = model.predict(X_preprocessed)
    
    return y_pred  # Return all predictions


def print_preference_difference_and_accuracy(preferences, updated_preferences, summary_only=False):

    total_actual = {'likes': 0, 'neutral': 0, 'dislikes': 0}
    total_correct = {'likes': 0, 'neutral': 0, 'dislikes': 0}

    def conditional_print(condition, message):
        if condition:
            print(message)

    accuracies = []
    
    for child in preferences:
        
        total_actual_child = {'likes': 0, 'neutral': 0, 'dislikes': 0}
        total_correct_child = {'likes': 0, 'neutral': 0, 'dislikes': 0}
    
        actual = {
            'likes': preferences[child]['known']['likes'] + preferences[child].get('unknown', {}).get('likes', []),
            'neutral': preferences[child]['known']['neutral'] + preferences[child].get('unknown', {}).get('neutral', []),
            'dislikes': preferences[child]['known']['dislikes'] + preferences[child].get('unknown', {}).get('dislikes', [])
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
        conditional_print(True, f"\nOverall Accuracy of Preferences: {overall_accuracy:.2f}")
        conditional_print(True, f"Standard Deviation of Accuracies: {accuracy_std_dev:.2f}")
    else:
        conditional_print(True, "No data to calculate overall accuracy and standard deviation.")



def get_true_preference_label(ingredient, child_preference, child):
    """
    Get the true preference label for a given ingredient.

    Args:
        ingredient (str): The ingredient to check.
        child_preference (list): List of ingredients with unknown preference.
        child (str): The key to access the child's preferences in the preferences dictionary.

    Returns:
        int: The true preference label (2 for likes, 1 for neutral, 0 for dislikes).

    Raises:
        ValueError: If the ingredient is not found in the preferences.
    """

    if ingredient in child_preference[child]['unknown']["likes"] + child_preference[child]['known']["likes"]:
        true_label = 2
    elif ingredient in child_preference[child]['unknown']["neutral"] + child_preference[child]['known']["neutral"]:
        true_label = 1
    elif ingredient in child_preference[child]['unknown']["dislikes"] + child_preference[child]['known']["dislikes"]:
        true_label = 0
    else:
        raise ValueError(f"Ingredient {ingredient} not found in preferences")

    return true_label


def ml_model_test(model, X_test, y_test, label_encoder):
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print Metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Classification Report:\n", class_report)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()