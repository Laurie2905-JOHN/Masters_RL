import logging
import os
import json
import random
import time
from sklearn.metrics import accuracy_score
from models.preferences.preference_utils import (
    get_child_data,
    initialize_child_preference_data,
    calculate_percent_of_known_ingredients_to_unknown,
    print_preference_difference_and_accuracy
)
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.menu_generators import RandomMenuGenerator, RLMenuGenerator
from models.preferences.random_menu_eval import MenuEvaluator
import numpy as np
import sys
from models.preferences.utility_calculator import MenuUtilityCalculator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from tqdm import tqdm

# Configure logging to only capture errors and critical messages
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


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
json_path = os.path.join(run_data_dir, "menu_utilities_simple")

# Complex weight function arguments
complex_weight_func_args = {
    'use_compensatory': True,
    'use_feedback': True,
    'use_fairness': True,
    'target_gini': 0.15,
}

menu_plan_length = 5

# Global start time to monitor execution duration
global_start_time = time.time()

def save_intermediate_results(results, seed):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.int32):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Save results to a file
    with open(os.path.join(run_data_dir, f'intermediate_results_seed_{seed}.json'), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    
    logging.info(f"Intermediate results saved for seed {seed}.")

def run_menu_generation(seed, model_name="random"):
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=0.05, seed=seed, plot_graphs=False
    )

    with open(os.path.join(run_data_dir, 'true_child_preference_data.json'), 'w') as f:
        json.dump(true_child_preference_data, f)
    
    # Initial prediction of preferences
    file_path = os.path.join(run_graphs_dir, "preferences_visualization.png")
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, file_path=file_path, seed=seed
    )
    
    updated_known_and_predicted_preferences_start, total_true_unknown_preferences, total_predicted_unknown_preferences, label_encoder = predictor.run_pipeline()
    
    label_mapping_start = {label: index for index, label in enumerate(label_encoder.classes_)}
    
    # Calculate unknown accuracy using sklearn's accuracy_score  
    prediction_accuracies_unknown = accuracy_score(total_true_unknown_preferences, total_predicted_unknown_preferences)
    prediction_accuracies_std_total = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences_start)
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
    
    negotiated_ingredients_start, _, unavailable_ingredients_start = negotiator.negotiate_ingredients()
    evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients_start, unavailable_ingredients_start)
    
    # Save negotiation results
    week, day = 1, 1
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)

    menu_generators = {
        "genetic": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, include_preference=True, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
        "random": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        "best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "prob_best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed),
    }
    
    
    def run_mod(model_name):
    
        if "genetic" in model_name:
            # Run initial genetic optimization
            menu_plan, _ = menu_generators[model_name].run_genetic_menu_optimization_and_finalize(
                negotiated_ingredients_start, unavailable_ingredients_start, 
                final_meal_plan_filename=f'final_meal_plan_{model_name}',
                ngen=180, population_size=333, cxpb=0.742143, mutpb=0.3062819
            )
        elif 'RL' in model_name:
            menu_plan, _ = menu_generators[model_name].generate_menu(
                negotiated_ingredients_start, unavailable_ingredients_start, 
                final_meal_plan_filename=f'final_meal_plan_{model_name}', 
                save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
                week=week, day=day
            )
        else:
            menu_plan, _ = menu_generators[model_name].generate_menu(
                negotiated_ingredients_start, unavailable_ingredients_start, 
                quantity_type='random', 
                final_meal_plan_filename=f'final_meal_plan_{model_name}', 
                save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
                week=week, day=day
            )
        return menu_plan

    menu_plan = run_mod(model_name)
        
    # Feedback methods
    sentiment_analyzer_perfect = SentimentAnalyzer(
        true_child_preference_data, menu_plan, child_data=child_feature_data, label_mapping = label_mapping_start, model_name='perfect', seed=seed
    )

    sentiment_analyzer_sentiment = SentimentAnalyzer(
        true_child_preference_data, menu_plan, child_data=child_feature_data, label_mapping = label_mapping_start, model_name='Vader', seed=seed
    )
    
    # Get updated preferences from feedback and sentiment analysis for both menu plans
    updated_known_unknown_preferences_with_feedback_perfect, _, _, _, _ = sentiment_analyzer_perfect.get_sentiment_and_update_data(plot_confusion_matrix=False)
    updated_known_unknown_preferences_with_feedback_sentiment, sentiment_accuracy_sentiment, _, _, _ = sentiment_analyzer_sentiment.get_sentiment_and_update_data(plot_confusion_matrix=False)
    
    percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_perfect)
    percent_of_known_preferences_sentiment = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_sentiment)
    
    utility_calculator_perfect = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=f"{json_path}_generator_utility_calculator_perfect_split_1_seed_{str(seed)}.json")
    utility_calculator_sentiment = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=f"{json_path}_generator_utility_calculator_sentiment_split_1_seed_{str(seed)}.json")

        # Using start prediction preference before feedback
    _ = utility_calculator_perfect.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan.keys()))
    _ = utility_calculator_sentiment.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan.keys()))
    
    info = {}
    reward = {}
    
    # Evaluate the menu plan and calculate the reward
    reward1, info1 = evaluator.select_ingredients(menu_plan)
    
    info['sentiment'] = info1
    reward['sentiment'] = reward1
    info['perfect'] = info1
    reward['perfect'] = reward1
    
    # Generate and evaluate menus, store results
    results = {}
    results['0'] = {} 
    
    # Store the results including time taken and reward
    results['0'] = {
        'info': info,
        'reward': reward,
        'percent_of_known_preferences_perfect': percent_of_known_preferences_perfect,
        'percent_of_known_preferences_sentiment': percent_of_known_preferences_sentiment,
        'sentiment_accuracy_sentiment': sentiment_accuracy_sentiment,
        'prediction_accuracies_unknown_perfect': prediction_accuracies_unknown,
        'prediction_accuracies_unknown_sentiment': prediction_accuracies_unknown,
        'prediction_accuracies_std_total_perfect': prediction_accuracies_std_total,
        'prediction_accuracies_std_total_sentiment': prediction_accuracies_std_total
    }
    
    
    with logging_redirect_tqdm():
        
        for menu in tqdm(range(1, 100), desc=f"Processing Menus for {model_name}"):
            results[str(menu)] = {}
            # Prediction of preferences based on expected preferences from sentiment analysis
            predictor_sentiment = PreferenceModel(
                ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_sentiment, visualize_data=False, seed=seed
            )
            predictor_perfect = PreferenceModel(
                ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_perfect, visualize_data=False, seed=seed
            )
            
            # Prediction with no feedback can use starting preference prediction as same each time
            # predictor_no_feedback = PreferenceModel(
            #     ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, seed=seed
            # )

            updated_known_and_predicted_preferences_sentiment, total_true_unknown_preferences_sentiment, total_predicted_unknown_preferences_sentiment, label_encoder = predictor_sentiment.run_pipeline()
            
            label_mapping_sentiment = {label: index for index, label in enumerate(label_encoder.classes_)}
                        
            # Calculate unknown accuracy using sklearn's accuracy_score  
            prediction_accuracies_unknown_sentiment = accuracy_score(total_true_unknown_preferences_sentiment, total_predicted_unknown_preferences_sentiment)
            prediction_accuracies_std_total_sentiment = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences_sentiment)
            

            
            updated_known_and_predicted_preferences_perfect, total_true_unknown_preferences_perfect, total_predicted_unknown_preferences_perfect, label_encoder = predictor_perfect.run_pipeline()
            
            # Calculate unknown accuracy using sklearn's accuracy_score  
            prediction_accuracies_unknown_perfect = accuracy_score(total_true_unknown_preferences_perfect, total_predicted_unknown_preferences_perfect)
            prediction_accuracies_std_total_perfect = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences_perfect)
            
            label_mapping_perfect = {label: index for index, label in enumerate(label_encoder.classes_)}
            
            # Initial negotiation of ingredients
            negotiator_sentiment = IngredientNegotiator(
                seed, ingredient_df, updated_known_and_predicted_preferences_sentiment, complex_weight_func_args, previous_feedback={}, previous_utility={}
            )
            
            negotiator_perfect = IngredientNegotiator(
                seed, ingredient_df, updated_known_and_predicted_preferences_perfect, complex_weight_func_args, previous_feedback={}, previous_utility={}
            )
                    
            negotiated_ingredients_sentiment, _, unavailable_ingredients_sentiment = negotiator_sentiment.negotiate_ingredients()
            negotiated_ingredients_perfect, _, unavailable_ingredients_perfect = negotiator_perfect.negotiate_ingredients()
            
            negotiated_ingredients_dict = {
                'sentiment': negotiated_ingredients_sentiment,
                'perfect': negotiated_ingredients_perfect,  
            }
            
            unavailable_ingredients_dict = {
                'sentiment': unavailable_ingredients_sentiment,
                'perfect': unavailable_ingredients_perfect,   
            }
            
            evaluator_dict = {
                'sentiment': MenuEvaluator(ingredient_df, negotiated_ingredients_sentiment, unavailable_ingredients_sentiment),
                'perfect': MenuEvaluator(ingredient_df, negotiated_ingredients_perfect, unavailable_ingredients_perfect),
            }
        
            
            menu_plans = {}
            reward = {}
            info = {}
            
            for preferences, sentiment_name in [(updated_known_and_predicted_preferences_perfect, 'perfect'), 
                                                (updated_known_and_predicted_preferences_sentiment, 'sentiment')]:

                
                eval = evaluator_dict[sentiment_name]
                unavailable = unavailable_ingredients_dict[sentiment_name]
                negotiated = negotiated_ingredients_dict[sentiment_name]
                
                if "genetic" in model_name:
                    # Run initial genetic optimization
                    menu_plans[sentiment_name], _ = menu_generators[model_name].run_genetic_menu_optimization_and_finalize(
                        negotiated, unavailable, 
                        final_meal_plan_filename=f'final_meal_plan_{model_name}',
                        ngen=180, population_size=333, cxpb=0.742143, mutpb=0.3062819
                    )
                else:
                    menu_plans[sentiment_name], _ = menu_generators[model_name].generate_menu(
                        negotiated, unavailable, 
                        final_meal_plan_filename=f'final_meal_plan_{model_name}', 
                        save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
                        week=week, day=day
                    )
                    
                reward[sentiment_name], info[sentiment_name] = evaluator_dict[sentiment_name].select_ingredients(menu_plans[sentiment_name])
                
                # Predicted utility calculated before feedback incoperated
                if sentiment_name == 'perfect':
                    _ = utility_calculator_perfect.calculate_day_menu_utility(updated_known_and_predicted_preferences_perfect, list(menu_plans['perfect'].keys()))
                else:
                    _ = utility_calculator_sentiment.calculate_day_menu_utility(updated_known_and_predicted_preferences_sentiment, list(menu_plans['sentiment'].keys()))
                    
                if sentiment_name == 'perfect':
                    sentiment_analyzer_perfect = SentimentAnalyzer(
                        true_child_preference_data, menu_plans['perfect'], child_data=child_feature_data, label_mapping = label_mapping_perfect, model_name='perfect', seed=seed
                    )
                    updated_known_unknown_preferences_with_feedback_perfect, _, _, _, _ = sentiment_analyzer_perfect.get_sentiment_and_update_data(plot_confusion_matrix=False)
                    percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_perfect)
                elif sentiment_name == 'sentiment':
                    sentiment_analyzer_sentiment = SentimentAnalyzer(
                        true_child_preference_data, menu_plans['sentiment'], child_data=child_feature_data, label_mapping = label_mapping_sentiment, model_name='Vader', seed=seed
                    )
                    updated_known_unknown_preferences_with_feedback_sentiment, sentiment_accuracy_sentiment, _, _, _ = sentiment_analyzer_sentiment.get_sentiment_and_update_data(plot_confusion_matrix=False)
                    percent_of_known_preferences_sentiment = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_sentiment)
                

            # Store the results including time taken and reward
            results[str(menu)] = {
                'info': info,
                'reward': reward,
                'percent_of_known_preferences_perfect': percent_of_known_preferences_perfect,
                'percent_of_known_preferences_sentiment': percent_of_known_preferences_sentiment,
                'sentiment_accuracy_sentiment': sentiment_accuracy_sentiment
            }
            
            results[str(menu)]['prediction_accuracies_unknown_sentiment'] = prediction_accuracies_unknown_sentiment
            results[str(menu)]['prediction_accuracies_std_total_sentiment'] = prediction_accuracies_std_total_sentiment
            results[str(menu)]['prediction_accuracies_unknown_perfect'] = prediction_accuracies_unknown_perfect
            results[str(menu)]['prediction_accuracies_std_total_perfect'] = prediction_accuracies_std_total_perfect
            
        _ = utility_calculator_perfect.close()
        _ = utility_calculator_sentiment.close()
    
    return results

import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging to only capture errors and critical messages
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    seed = random.randint(0, int(1e6))
    
    menu_generators = [
        "random",
        "prob",
        "best",
        "prob_best"
        # "RL",
        "genetic",
    ]
    
    try:
        results = {}
        with logging_redirect_tqdm():
            for model_name in tqdm(menu_generators, desc="Testing Menu Generator Types"):
                results[model_name] = run_menu_generation(seed, model_name)

                # Check elapsed time
                elapsed_time = time.time() - global_start_time
                if elapsed_time > 29.5 * 3600:
                    logging.warning("Approaching time limit, saving intermediate results.")
                    save_intermediate_results(results, seed)
                    break

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        save_intermediate_results(results, seed)
        sys.exit(1)

    # Final save of all results
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.int32):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(os.path.join(run_data_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    
    logging.info("Final results saved.")

if __name__ == "__main__":
    main()

# Run 37 just RL hpc the others