import logging
import numpy as np

from models.preferences.data_utils import (
    get_child_data,
    initialize_child_preference_data,
    print_preference_difference_and_accuracy,
    calculate_percent_of_known_ingredients_to_unknown,
    plot_child_accuracies,
    plot_accuracies,
)

from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from models.preferences.menu_generator import RandomMenuGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
weight_function = 'simple'
iterations = 500
seed = None
model_name = 'perfect'
apply_SMOTE = True
initial_split = 0.5

def main():
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=initial_split, seed=seed, plot_graphs=False
    )

    # Lists to store accuracy values
    prediction_accuracies = []
    prediction_std_devs = []
    sentiment_accuracies = []
    percent_knowns = []
    previous_feedback = {}
    previous_fairness_index = {}
    previous_utility = {}

    # Initialize menu generator
    menu_generator = RandomMenuGenerator(seed=seed)

    # Initial prediction of preferences
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, apply_SMOTE=apply_SMOTE, seed=seed
    )
    updated_known_and_predicted_preferences = predictor.run_pipeline()

    accuracy, std_dev = print_preference_difference_and_accuracy(
        true_child_preference_data, updated_known_and_predicted_preferences, summary_only=True
    )

    prediction_accuracies.append(accuracy)
    prediction_std_devs.append(std_dev)
    
    # Log initial prediction results
    logging.info(f"Initial Prediction - Accuracy: {accuracy}, Std Dev: {std_dev}")
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_fairness_index, previous_utility
    )
    
    negotiated_ingredients, unavailable_ingredients = negotiator.negotiate_ingredients(weight_function)

    # Generate inital random menu based on negotiated list
    menu_plan = menu_generator.generate_random_menu(negotiated_ingredients, unavailable_ingredients)

    # Sentiment analysis initiation, initially with true preference data and will adapt it to updated preferences from feedback
    sentiment_analyzer = SentimentAnalyzer(
        true_child_preference_data, menu_plan, model_name=model_name, seed=seed
    )
    
    # Get updated preferences from feedback the sentiment accuracy and feedback given
    updated_known_unknown_preferences_with_feedback, sentiment_accuracy, feedback_given = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)

    # Calculate the percentage of known ingredients to unknown ingredients
    percent_of_known_preferences = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback)
    
    percent_knowns.append(percent_of_known_preferences)
    sentiment_accuracies.append(sentiment_accuracy)

    for i in range(iterations):
        
        logging.info(f"Iteration {i + 1}")
        
        # Prediction of preferences based on expected preferences from sentiment analysis
        predictor = PreferenceModel(
            ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback, apply_SMOTE=apply_SMOTE, seed=seed
        )
        
        updated_known_and_predicted_preferences = predictor.run_pipeline()

        # Compare accuracy with original true preferences
        accuracy, std_dev = print_preference_difference_and_accuracy(
            true_child_preference_data, updated_known_and_predicted_preferences, summary_only=True
        )
        
        # Log prediction results
        logging.info(f"Iteration {i + 1} - Prediction Accuracy: {accuracy}, Std Dev: {std_dev}")

        prediction_accuracies.append(accuracy)
        prediction_std_devs.append(std_dev)

        # Initial negotiation of ingredients
        negotiator = IngredientNegotiator(
            seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_fairness_index, previous_utility
        )

        # Negotiate ingredients
        negotiated_ingredients, unavailable_ingredients = negotiator.negotiate_ingredients(weight_function)

        # Generate random menu based on negotiated list
        menu_plan = menu_generator.generate_random_menu(negotiated_ingredients, unavailable_ingredients)
        
        # Sentiment analysis initiation
        sentiment_analyzer = SentimentAnalyzer(
            updated_known_unknown_preferences_with_feedback, menu_plan, model_name=model_name, seed=seed
        )
        
        # Get updated preferences from feedback the sentiment accuracy and feedback given
        updated_known_unknown_preferences_with_feedback, sentiment_accuracy, feedback_given = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)

        # Calculate the percentage of known ingredients to unknown ingredients
        percent_of_known_preferences = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback)
        
        percent_knowns.append(percent_of_known_preferences)
        sentiment_accuracies.append(sentiment_accuracy)
        
        # Log sentiment analysis results
        logging.info(f"Iteration {i + 1} - Sentiment Accuracy: {sentiment_accuracy}, Percent Known: {percent_of_known_preferences}")

        previous_feedback = feedback_given
        previous_fairness_index = {}
        previous_utility = {}

    # Plot and save the accuracies
    plot_accuracies(prediction_accuracies, prediction_std_devs, sentiment_accuracies, iterations, 'accuracy_plot.png')
    plot_child_accuracies(percent_knowns, 'child_accuracies_plot.png')


if __name__ == "__main__":
    main()
