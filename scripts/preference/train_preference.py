import logging
import numpy as np
import matplotlib.pyplot as plt
from models.preferences.data_utils import (
    get_child_data,
    initialize_child_preference_data,
    print_preference_difference_and_accuracy,
)
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from models.preferences.menu_generator import RandomMenuGenerator

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

weight_function='simple'
iterations=10 
seed=None

"""Main function to run the preference prediction and negotiation pipeline."""

ingredient_df = get_data("data.csv")
child_feature_data = get_child_data()
child_preference_data = initialize_child_preference_data(
    child_feature_data, ingredient_df, split=0.5, seed=seed, plot_graphs=False
)

# Lists to store accuracy values
prediction_accuracies = []
prediction_std = []
sentiment_accuracies = []

# Initialize menu generator
menu_generator = RandomMenuGenerator(seed=seed)

# Initial Prediction of Preferences
predictor = PreferenceModel(
    ingredient_df, child_feature_data, child_preference_data, apply_SMOTE=True, seed=seed
)
updated_known_and_predicted_preferences = predictor.run_pipeline()

accuracy, std_dev = print_preference_difference_and_accuracy(
    child_preference_data, updated_known_and_predicted_preferences, summary_only=True
)

prediction_accuracies.append(accuracy)
prediction_std.append(std_dev)

previous_feedback = {}
previous_fairness_index = {}
previous_utility = {}

for i in range(iterations):
    logging.info(f"Iteration {i + 1}")

    # Initial Negotiation of Ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_fairness_index, previous_utility
    )

    if weight_function == "simple":
        # Simple
        negotiated, unavailable = negotiator.negotiate_ingredients_simple()
    elif weight_function == "complex":
        # Complex
        negotiated, unavailable = negotiator.negotiate_ingredients_complex()
    else:
        raise ValueError("Invalid weight function")

    # # Close negotiator to get initial gini stats on voting
    # negotiator.close(log_file=f"Gini_iteration_{i + 1}.json")

    # Generate random menu
    menu_plan = menu_generator.generate_random_menu(negotiated, unavailable)

    # Sentiment analysis and get updated preference list
    sentiment_analyzer = SentimentAnalyzer(
        child_preference_data, menu_plan, seed=seed
    )
    (
        updated_true_preferences_with_feedback,
        sentiment_accuracy,
        feedback_given,
    ) = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)

    logging.info(f"Iteration {i + 1} - Sentiment Accuracy: {sentiment_accuracy}")
    sentiment_accuracies.append(sentiment_accuracy)

    # Update predictor with new preferences
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, updated_true_preferences_with_feedback, apply_SMOTE=True, seed=seed
    )
    updated_known_and_predicted_preferences = predictor.run_pipeline()

    accuracy, std_dev = print_preference_difference_and_accuracy(
        child_preference_data, updated_known_and_predicted_preferences, summary_only=True
    )
    logging.info(f"Iteration {i + 1} - Prediction Accuracy: {accuracy}")
    prediction_accuracies.append(accuracy)
    prediction_std.append(std_dev)

    previous_feedback = feedback_given
    previous_fairness_index = {}
    previous_utility = {}

