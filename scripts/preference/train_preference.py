import logging
import os
import json
import random
from sklearn.metrics import accuracy_score
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
from models.preferences.menu_generators import RandomMenuGenerator, RLMenuGenerator
from models.preferences.utility_calculator import MenuUtilityCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
iterations = 100
seed = None
sentiment_model_name = 'Vader'
initial_split = 0.05

# Set to zero for complete randomness
probability_best = 1
weight_type = "score"  # Can be "random" or "score"
menu_plan_length = 5

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

# Initialize lists to store accuracy values and standard deviations
prediction_accuracies_unknown = []
prediction_accuracies_total = []
prediction_std_total = []

# Complex weight function arguments
complex_weight_func_args = {
    'use_normalize_total_voting_weight': False,
    'use_normalize_vote_categories': True,
    'use_compensatory': True,
    'use_feedback': True,
    'use_fairness': True,
    'target_gini': 0.15,
}

menu_generator_type = "notRL"

def main():
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=initial_split, seed=seed, plot_graphs=False
    )

    with open(os.path.join(run_data_dir, 'true_child_preference_data.json'), 'w') as f:
        json.dump(true_child_preference_data, f)

    # Initialize menu generators
    seed_generator = random.randint(0, int(1e36))
    
    if menu_generator_type != "RL":
        menu_generator_simple = RandomMenuGenerator(menu_plan_length=menu_plan_length, weight_type=weight_type, probability_best=probability_best, seed=seed_generator)
        menu_generator_complex = RandomMenuGenerator(menu_plan_length=menu_plan_length, weight_type=weight_type, probability_best=probability_best, seed=seed_generator)
    else:
        menu_generator_simple = RLMenuGenerator(ingredient_df, menu_plan_length=menu_plan_length, weight_type=weight_type, seed=seed_generator, model_save_path='rl_model')
        menu_generator_complex = RLMenuGenerator(ingredient_df, menu_plan_length=menu_plan_length, weight_type=weight_type, seed=seed_generator, model_save_path='rl_model')
        
    # Initialize utility calculator
    json_path_simple = os.path.join(run_data_dir, "menu_utilities_simple.json")
    json_path_complex = os.path.join(run_data_dir, "menu_utilities_complex.json")
    
    utility_calculator_simple = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_simple)
    utility_calculator_complex = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_complex)
    
    # Initial prediction of preferences
    file_path = os.path.join(run_graphs_dir, "preferences_visualization.png")
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, visualize_data=True, file_path=file_path, seed=seed
    )
    
    updated_known_and_predicted_preferences, total_true_unknown_preferences, total_predicted_unknown_preferences, label_encoder = predictor.run_pipeline()
    
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    
    # Calculate unknown accuracy using sklearn's accuracy_score
    unknown_accuracy = accuracy_score(total_true_unknown_preferences, total_predicted_unknown_preferences)
    
    prediction_accuracies_unknown.append(unknown_accuracy)
    
    # Log initial prediction results
    logging.info(f"Initial Prediction - Global Accuracy: {unknown_accuracy}")
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
        
    negotiated_ingredients_simple, negotiated_ingredients_complex, unavailable_ingredients = negotiator.negotiate_ingredients()
    
    # Calculate week and day
    week = 1
    day = 1
    
    # Save negotiation results
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)

    # Generate menus based on negotiated lists
    menu_plan_simple = menu_generator_simple.generate_menu(negotiated_ingredients_simple, unavailable_ingredients, final_meal_plan_filename='final_meal_plan_simple', save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, week=week, day=day)
    menu_plan_complex = menu_generator_complex.generate_menu(negotiated_ingredients_complex, unavailable_ingredients, final_meal_plan_filename='final_meal_plan_complex', save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, week=week, day=day)
    
    # Calculate the predicted utility for all children for both menus
    previous_utility_simple = utility_calculator_simple.calculate_day_menu_utility(updated_known_and_predicted_preferences, menu_plan_simple)
    previous_utility_complex = utility_calculator_complex.calculate_day_menu_utility(updated_known_and_predicted_preferences, menu_plan_complex)

    # Sentiment analysis initiation for simple and complex menu plans
    sentiment_analyzer_simple = SentimentAnalyzer(
        true_child_preference_data, menu_plan_simple, child_data=child_feature_data, label_mapping = label_mapping, model_name=sentiment_model_name, seed=seed
    )
    sentiment_analyzer_complex = SentimentAnalyzer(
        true_child_preference_data, menu_plan_complex, child_data=child_feature_data, label_mapping = label_mapping,  model_name=sentiment_model_name, seed=seed
    )
    
    # Get updated preferences from feedback and sentiment analysis for both menu plans
    updated_known_unknown_preferences_with_feedback_simple, sentiment_accuracy_simple, feedback_given_simple, _, _ = sentiment_analyzer_simple.get_sentiment_and_update_data(plot_confusion_matrix=False)
    updated_known_unknown_preferences_with_feedback_complex, sentiment_accuracy_complex, feedback_given_complex, _, _ = sentiment_analyzer_complex.get_sentiment_and_update_data(plot_confusion_matrix=False)
    
    # Assign the feedback given to the previous feedback for complex weight calculation
    previous_feedback_simple = feedback_given_simple
    previous_feedback_complex = feedback_given_complex

    # Calculate the percentage of known ingredients to unknown ingredients for both
    percent_of_known_preferences_simple = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_simple)
    percent_of_known_preferences_complex = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_complex)
    
    percent_knowns = [(percent_of_known_preferences_simple, percent_of_known_preferences_complex)]
    sentiment_accuracies = [(sentiment_accuracy_simple, sentiment_accuracy_complex)]

    for i in range(iterations):
        week = (i // 5) + 1
        day = (i % 5) + 1

        logging.info(f"Week {week}, Day {day} - Iteration {i + 1}")
        
        # Prediction of preferences based on expected preferences from sentiment analysis
        predictor_simple = PreferenceModel(
            ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_simple, visualize_data=False, seed=seed
        )
        predictor_complex = PreferenceModel(
            ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_complex, visualize_data=False, seed=seed
        )
        
        updated_known_and_predicted_preferences_simple, total_true_unknown_preferences_simple, total_predicted_unknown_preferences_simple, label_encoder_simple = predictor_simple.run_pipeline()
        label_mapping_simple = {label: index for index, label in enumerate(label_encoder.classes_)}
        updated_known_and_predicted_preferences_complex, total_true_unknown_preferences_complex, total_predicted_unknown_preferences_complex, label_encoder_complex = predictor_complex.run_pipeline()
        label_mapping_complex = {label: index for index, label in enumerate(label_encoder.classes_)}
        
        # Calculate unknown accuracy using sklearn's accuracy_score for both simple and complex predictions
        unknown_accuracy_simple = accuracy_score(total_true_unknown_preferences_simple, total_predicted_unknown_preferences_simple)
        unknown_accuracy_complex = accuracy_score(total_true_unknown_preferences_complex, total_predicted_unknown_preferences_complex)
        
        unknown_accuracy_simple, unknown_std_simple = print_preference_difference_and_accuracy(
                true_child_preference_data, updated_known_and_predicted_preferences_simple, summary_only=True
            )
        
        unknown_accuracy_complex, unknown_std_complex = print_preference_difference_and_accuracy(
                true_child_preference_data, updated_known_and_predicted_preferences_complex, summary_only=True
            )
        # Log prediction results
        logging.info(f"Week {week}, Day {day} - Simple Global Accuracy: {unknown_accuracy_simple}")
        logging.info(f"Week {week}, Day {day} - Complex Global Accuracy: {unknown_accuracy_complex}")

        prediction_accuracies_unknown.append((unknown_accuracy_simple, unknown_accuracy_complex))
        prediction_accuracies_total.append((unknown_accuracy_simple, unknown_accuracy_complex))
        prediction_std_total.append((unknown_std_simple, unknown_std_complex))
        
        # Negotiate ingredients again
        negotiator = IngredientNegotiator(
            seed, ingredient_df, updated_known_and_predicted_preferences_simple, complex_weight_func_args, previous_feedback_complex, previous_utility_complex
        )

        # Negotiate ingredients
        negotiated_ingredients_simple, negotiated_ingredients_complex, unavailable_ingredients = negotiator.negotiate_ingredients()
        
        # Save negotiation results
        negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)
        
        # Generate menus based on negotiated lists
        menu_plan_simple = menu_generator_simple.generate_menu(negotiated_ingredients_simple, unavailable_ingredients, final_meal_plan_filename='final_meal_plan_simple', save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, week=week, day=day)
        menu_plan_complex = menu_generator_complex.generate_menu(negotiated_ingredients_complex, unavailable_ingredients, final_meal_plan_filename='final_meal_plan_complex', save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, week=week, day=day)
        
        # Calculate the predicted utility for all children for both menus
        previous_utility_simple = utility_calculator_simple.calculate_day_menu_utility(updated_known_and_predicted_preferences_simple, menu_plan_simple)
        previous_utility_complex = utility_calculator_complex.calculate_day_menu_utility(updated_known_and_predicted_preferences_complex, menu_plan_complex)
                        
        # Sentiment analysis initiation
        sentiment_analyzer_simple = SentimentAnalyzer(
            updated_known_unknown_preferences_with_feedback_simple, menu_plan_simple, child_data=child_feature_data, label_mapping = label_mapping_simple, model_name=sentiment_model_name, seed=seed
        )
        sentiment_analyzer_complex = SentimentAnalyzer(
            updated_known_unknown_preferences_with_feedback_complex, menu_plan_complex, child_data=child_feature_data, label_mapping = label_mapping_complex, model_name=sentiment_model_name, seed=seed
        )
        
        # Get updated preferences from feedback the sentiment accuracy and feedback given
        updated_known_unknown_preferences_with_feedback_simple, sentiment_accuracy_simple, feedback_given_simple, _, _ = sentiment_analyzer_simple.get_sentiment_and_update_data(plot_confusion_matrix=False)
        updated_known_unknown_preferences_with_feedback_complex, sentiment_accuracy_complex, feedback_given_complex, _, _ = sentiment_analyzer_complex.get_sentiment_and_update_data(plot_confusion_matrix=False)

        # Assign the feedback given to the previous feedback for complex weight calculation
        previous_feedback_simple = feedback_given_simple
        previous_feedback_complex = feedback_given_complex
        
        # Calculate the percentage of known ingredients to unknown ingredients
        percent_of_known_preferences_simple = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_simple)
        percent_of_known_preferences_complex = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_complex)
        
        percent_knowns.append((percent_of_known_preferences_simple, percent_of_known_preferences_complex))
        sentiment_accuracies.append((sentiment_accuracy_simple, sentiment_accuracy_complex))
        
        # Log sentiment analysis results
        logging.info(f"Week {week}, Day {day} - Simple Sentiment Accuracy: {sentiment_accuracy_simple}, Percent Known: {percent_of_known_preferences_simple}")
        logging.info(f"Week {week}, Day {day} - Complex Sentiment Accuracy: {sentiment_accuracy_complex}, Percent Known: {percent_of_known_preferences_complex}")
    
    utility_calculator_simple.close()
    utility_calculator_complex.close()
    
    # Plot and save the accuracies
    plot_preference_and_sentiment_accuracies(prediction_accuracies_total, prediction_std_total, sentiment_accuracies, iterations, os.path.join(run_graphs_dir, 'accuracy_plot.png'))
    plot_individual_child_known_percent(percent_knowns, os.path.join(run_graphs_dir, 'child_known_percent_plot.png'))
    # Plot and save the utilities
    plot_utilities_and_mape(os.path.join(run_data_dir, "menu_utilities_simple.json"), run_graphs_dir)
    plot_utilities_from_json(os.path.join(run_data_dir, "menu_utilities_simple.json"), run_graphs_dir)
    
    plot_utilities_and_mape(os.path.join(run_data_dir, "menu_utilities_complex.json"), run_graphs_dir)
    plot_utilities_from_json(os.path.join(run_data_dir, "menu_utilities_complex.json"), run_graphs_dir)
    
    # Plot the top 10 menus for both simple and complex methods
    menu_generator_simple.plot_top_menus(top_n=10, save_path=os.path.join(run_graphs_dir, "top_menus_simple.png"))
    menu_generator_complex.plot_top_menus(top_n=10, save_path=os.path.join(run_graphs_dir, "top_menus_complex.png"))
    
if __name__ == "__main__":
    main()
