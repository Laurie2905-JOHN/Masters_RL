import logging
import os
import json
from models.preferences.preference_utils import (
    get_child_data,
    initialize_child_preference_data,
    print_preference_difference_and_accuracy,
    calculate_percent_of_known_ingredients_to_unknown,
    plot_individual_child_known_percent,
    plot_preference_and_sentiment_accuracies,
    plot_utilities_and_mape,
    plot_utilities_from_json,
)

from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from models.preferences.menu_generators import RandomMenuGenerator
from models.preferences.menu_generators import RLMenuGenerator
from models.preferences.utility_calculator import MenuUtilityCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
iterations = 185
seed = None
model_name = 'perfect'
apply_SMOTE = True
initial_split = 0.5
weight_function = 'simple'

# Set to zero for complete randomness
probability_best = 0.8
# Random is all equal and score is based on the score of the ingredient in terms of the negotiated list
weight_type = "random"
# weight_type = "score"
menu_plan_length = 10

# Directory paths
base_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'saved_data', 'data')
graphs_dir = os.path.join(base_dir, 'saved_data', 'graphs')

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# Determine the run number
existing_runs = [d for d in os.listdir(data_dir) if d.startswith('run_')]
run_number = len(existing_runs) + 1

# Create run directories
run_data_dir = os.path.join(data_dir, f'run_{run_number}')
run_graphs_dir = os.path.join(graphs_dir, f'run_{run_number}')
os.makedirs(run_data_dir, exist_ok=True)
os.makedirs(run_graphs_dir, exist_ok=True)

# Complex weight function arguments
complex_weight_func_args = {
    'use_normalize_total_voting_weight': False,
    'use_normalize_vote_categories': True,
    'use_compensatory': True,
    'use_feedback': True,
    'use_fairness': True,
    'target_gini': 0.15,
}

menu_generator_type = "RL"

def main():
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=initial_split, seed=seed, plot_graphs=False
    )

    with open(os.path.join(run_data_dir, 'true_child_preference_data.json'), 'w') as f:
        json.dump(true_child_preference_data, f)

    # Lists to store accuracy values
    prediction_accuracies = []
    prediction_std_devs = []
    sentiment_accuracies = []
    percent_knowns = []
    
    previous_feedback = {}
    previous_utility = {}
    
    # Initialize menu generator
    
    if menu_generator_type != "RL":
        menu_generator = RandomMenuGenerator(menu_plan_length = menu_plan_length, weight_type = weight_type, probability_best = probability_best, seed = seed)
    else:
        menu_generator = RLMenuGenerator(ingredient_df, menu_plan_length = menu_plan_length, weight_type = weight_type, seed = seed, model_save_path = 'rl_model')

    # Initialize utility calculator
    json_path = os.path.join(run_data_dir, "menu_utilities.json")
    utility_calculator = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path)
    
    # Initial prediction of preferences
    file_path = os.path.join(run_graphs_dir, "preferences_visualization.png")
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, visualize_data=True, apply_SMOTE=apply_SMOTE, file_path=file_path, seed=seed
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
        seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_utility, complex_weight_func_args
    )
    
    negotiated_ingredients, unavailable_ingredients = negotiator.negotiate_ingredients(weight_function=weight_function)
    
    # Calculate week and day
    week = 1
    day = 1
    
    # Save negotiation results
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)

    # Generate menu based on negotiated list
    menu_plan = menu_generator.generate_menu(negotiated_ingredients, unavailable_ingredients)
    
    # Calculate the predicted utility for all children for a given meal plan
    previous_utility = utility_calculator.calculate_day_menu_utility(updated_known_and_predicted_preferences, menu_plan)

    # Sentiment analysis initiation, initially with true preference data and will adapt it to updated preferences from feedback
    sentiment_analyzer = SentimentAnalyzer(
        true_child_preference_data, menu_plan, model_name=model_name, seed=seed
    )
    
    # Get updated preferences from feedback the sentiment accuracy and feedback given
    updated_known_unknown_preferences_with_feedback, sentiment_accuracy, feedback_given = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)
    
    # Assign the feedback given to the previous feedback for complex weight calculation
    previous_feedback = feedback_given

    # Calculate the percentage of known ingredients to unknown ingredients
    percent_of_known_preferences = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback)
    
    percent_knowns.append(percent_of_known_preferences)
    sentiment_accuracies.append(sentiment_accuracy)

    for i in range(iterations):
        week = (i // 5) + 1
        day = (i % 5) + 1

        logging.info(f"Week {week}, Day {day} - Iteration {i + 1}")
        
        # Prediction of preferences based on expected preferences from sentiment analysis
        predictor = PreferenceModel(
            ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback, visualize_data=False, apply_SMOTE=apply_SMOTE, seed=seed
        )
        
        updated_known_and_predicted_preferences = predictor.run_pipeline()

        # Compare accuracy with original true preferences
        accuracy, std_dev = print_preference_difference_and_accuracy(
            true_child_preference_data, updated_known_and_predicted_preferences, summary_only=True
        )
        
        # Log prediction results
        logging.info(f"Week {week}, Day {day} - Prediction Accuracy: {accuracy}, Std Dev: {std_dev}")

        prediction_accuracies.append(accuracy)
        prediction_std_devs.append(std_dev)

        # Initial negotiation of ingredients
        negotiator = IngredientNegotiator(
            seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_utility, complex_weight_func_args
        )

        # Negotiate ingredients
        negotiated_ingredients, unavailable_ingredients = negotiator.negotiate_ingredients(weight_function=weight_function)
        
        # Save negotiation results
        negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)
        
        # Generate menu based on negotiated list
        menu_plan = menu_generator.generate_menu(negotiated_ingredients, unavailable_ingredients)
        
        # Calculate the predicted utility for all children for a given meal plan
        previous_utility = utility_calculator.calculate_day_menu_utility(updated_known_and_predicted_preferences, menu_plan)
                        
        # Sentiment analysis initiation
        sentiment_analyzer = SentimentAnalyzer(
            updated_known_unknown_preferences_with_feedback, menu_plan, model_name=model_name, seed=seed
        )
        
        # Get updated preferences from feedback the sentiment accuracy and feedback given
        updated_known_unknown_preferences_with_feedback, sentiment_accuracy, feedback_given = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)

        # Assign the feedback given to the previous feedback for complex weight calculation
        previous_feedback = feedback_given
        
        # Calculate the percentage of known ingredients to unknown ingredients
        percent_of_known_preferences = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback)
        
        percent_knowns.append(percent_of_known_preferences)
        sentiment_accuracies.append(sentiment_accuracy)
        
        # Log sentiment analysis results
        logging.info(f"Week {week}, Day {day} - Sentiment Accuracy: {sentiment_accuracy}, Percent Known: {percent_of_known_preferences}")
    
    utility_calculator.close()
    
    # Plot and save the accuracies
    plot_preference_and_sentiment_accuracies(prediction_accuracies, prediction_std_devs, sentiment_accuracies, iterations, os.path.join(run_graphs_dir, 'accuracy_plot.png'))
    plot_individual_child_known_percent(percent_knowns, os.path.join(run_graphs_dir, 'child_known_percent_plot.png'))
    # Plot and save the utilities
    plot_utilities_and_mape(os.path.join(run_data_dir, "menu_utilities.json"), run_graphs_dir)
    plot_utilities_from_json(os.path.join(run_data_dir, "menu_utilities.json"), run_graphs_dir)
    
    # Plot the top 10 menus
    menu_generator.plot_top_menus(top_n=10, save_path=os.path.join(run_graphs_dir, "top_menus.png"))
    
if __name__ == "__main__":
    main()
