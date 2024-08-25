import logging
import os
import json
import random
import time
import sys
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
from models.preferences.utility_calculator import MenuUtilityCalculator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
run_data_dir = os.path.join(data_dir, f'run_{run_number}_mech_test')
run_graphs_dir = os.path.join(graphs_dir, f'run_{run_number}_mech_test')
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
    
def run_mod(model_name, negotiated, unavailable, menu_generators, ingredient_df, method, week=1, day=1):
    
    menu_generator = menu_generators[method][model_name]
    
    if model_name != 'RL':
        menu_generator.update_evaluator(ingredient_df, negotiated, unavailable)

    if "genetic" in model_name:
        # Run initial genetic optimization
        menu_plan, _ = menu_generator.run_genetic_menu_optimization_and_finalize(
            negotiated, unavailable, 
            final_meal_plan_filename=f'final_meal_plan_{model_name}',
            ngen=180, population_size=333, cxpb=0.742143, mutpb=0.3062819
        )
    elif 'RL' in model_name:
        menu_plan, _ = menu_generator.generate_menu(
            negotiated, unavailable, 
            final_meal_plan_filename=f'final_meal_plan_{model_name}', 
            save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
            week=week, day=day
        )
    else:
        menu_plan, _ = menu_generator.generate_menu(
            negotiated, unavailable, 
            quantity_type='random', 
            final_meal_plan_filename=f'final_meal_plan_{model_name}', 
            save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
            week=week, day=day
        )
    return menu_plan

def run_menu_generation(seed, model_name, negotiated_ingredients_start, unavailable_ingredients_start, menu_generators, ingredient_df, true_child_preference_data, child_feature_data, label_mapping_start, prediction_accuracies_unknown_start, prediction_accuracies_std_total_start, updated_known_and_predicted_preferences_start):
    
    menu_plan_dict = {
        'sentiment': 0,
        'perfect': 0,
        'no_feedback': 0,
    }
    
    results = {}
    
    # Initialize utility calculators for each method
    utility_calculator_perfect = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=f"{json_path}_generator_utility_calculator_perfect_split_1_model_{model_name}_seed_{str(seed)}.json")
    utility_calculator_sentiment = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=f"{json_path}_generator_utility_calculator_sentiment_split_1_model_{model_name}_seed_{str(seed)}.json")
    utility_calculator_no_feedback = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=f"{json_path}_generator_utility_calculator_no_feedback_split_1_model_{model_name}_seed_{str(seed)}.json")
    
    for method in menu_generators.keys():
        # Initialize an empty dictionary for the current method
        results[method] = {}
        
        # Generate menu plan using the current method
        menu_plan_dict[method] = run_mod(model_name, negotiated_ingredients_start, unavailable_ingredients_start, menu_generators, ingredient_df, method, week=1, day=1)
        
        # Perform feedback analysis and calculate metrics
        if method == 'perfect':
            sentiment_analyzer = SentimentAnalyzer(
                true_child_preference_data, true_child_preference_data, menu_plan_dict[method], child_data=child_feature_data, label_mapping=label_mapping_start, model_name='perfect', seed=seed
            )
            updated_preferences_with_feedback_perfect, _, perfect_feedback, _, _ = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)
            percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_preferences_with_feedback_perfect)
            utility_perfect_true, utility_perfect_predicted = utility_calculator_perfect.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_dict[method].keys()))
            accuracy_unknown_perfect = prediction_accuracies_unknown_start
            accuracy_std_total_perfect = prediction_accuracies_std_total_start

        elif method == 'sentiment':
            sentiment_analyzer = SentimentAnalyzer(
                true_child_preference_data, true_child_preference_data, menu_plan_dict[method], child_data=child_feature_data, label_mapping=label_mapping_start, model_name='Vader', seed=seed
            )
            updated_preferences_with_feedback_sentiment, sentiment_accuracy, sentiment_feedback, _, _ = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)
            percent_of_known_preferences_sentiment = calculate_percent_of_known_ingredients_to_unknown(updated_preferences_with_feedback_sentiment)
            utility_sentiment_true, utility_sentiment_predicted = utility_calculator_sentiment.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_dict[method].keys()))
            accuracy_unknown_sentiment = sentiment_accuracy
            accuracy_std_total_sentiment = prediction_accuracies_std_total_start

        elif method == 'no_feedback':
            percent_of_known_preferences_no_feedback = calculate_percent_of_known_ingredients_to_unknown(true_child_preference_data)
            utility_no_feedback_true, utility_no_feedback_predicted  = utility_calculator_no_feedback.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_dict[method].keys()))
            accuracy_unknown_no_feedback = prediction_accuracies_unknown_start
            accuracy_std_total_no_feedback = prediction_accuracies_std_total_start
            no_feedback_feedback = {} 
            
        # Evaluate the menu plan and calculate the reward
        if model_name != 'RL':
            reward, info = menu_generators[method][model_name].evaluator.select_ingredients(menu_plan_dict[method])
        else:
            evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients = negotiated_ingredients_start, unavailable_ingredients = unavailable_ingredients_start)
            reward, info = evaluator.select_ingredients(menu_plan_dict[method])

        
        # Store the results including time taken and reward for the current method
        results[method].update({
            '0': {
                'info': info,
                'reward': reward,
                'percent_of_known_preferences': locals()[f'percent_of_known_preferences_{method}'],
                'true_utility': locals()[f'utility_{method}_true'],
                'predicted_utility': locals()[f'utility_{method}_predicted'],
                'accuracy_unknown': locals()[f'accuracy_unknown_{method}'],
                'accuracy_std_total': locals()[f'accuracy_std_total_{method}'],
                'feedback': locals()[f'{method}_feedback']
            }
        })
    
    # Loop for processing multiple menus
    for method in tqdm(menu_generators.keys(), desc="Testing Method Types"):
        for menu in tqdm(range(1, 100), desc=f"Processing Menus for method {method} and {model_name}"):
            
            week = (menu - 1) // 5 + 1  # Week number (1-based index)
            day = (menu - 1) % 5 + 1    # Day number within the week (1-based index)
            
            results[method][str(menu)] = {}
            
            if method == 'perfect':
                predictor = PreferenceModel(
                    ingredient_df, child_feature_data, updated_preferences_with_feedback_perfect, visualize_data=False, seed=seed
                )
                
                updated_known_and_predicted_preferences, total_true_unknown_preferences, total_predicted_unknown_preferences, label_encoder = predictor.run_pipeline()
            
                # Calculate unknown accuracy using sklearn's accuracy_score  
                accuracy_unknown = accuracy_score(total_true_unknown_preferences, total_predicted_unknown_preferences)
                accuracy_std_total = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences)
                
                label_mapping_perfect = {label: index for index, label in enumerate(label_encoder.classes_)}

                negotiator = IngredientNegotiator(
                    seed, ingredient_df, updated_known_and_predicted_preferences, complex_weight_func_args, previous_feedback={}, previous_utility={}
                )
                        
                negotiated_ingredients, _, unavailable_ingredients = negotiator.negotiate_ingredients()
                            
            elif method == 'sentiment':
                predictor = PreferenceModel(
                    ingredient_df, child_feature_data, updated_preferences_with_feedback_sentiment, visualize_data=False, seed=seed
                )

                updated_known_and_predicted_preferences, total_true_unknown_preferences, total_predicted_unknown_preferences, label_encoder = predictor.run_pipeline()
                
                label_mapping_sentiment = {label: index for index, label in enumerate(label_encoder.classes_)}
                        
                # Calculate unknown accuracy using sklearn's accuracy_score  
                accuracy_unknown = accuracy_score(total_true_unknown_preferences, total_predicted_unknown_preferences)
                accuracy_std_total = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences)
                
                # Initial negotiation of ingredients
                negotiator = IngredientNegotiator(
                    seed, ingredient_df, updated_known_and_predicted_preferences, complex_weight_func_args, previous_feedback={}, previous_utility={}
                )
                
                negotiated_ingredients, _, unavailable_ingredients = negotiator.negotiate_ingredients()

            elif method == 'no_feedback':
                pass
            
            else:
                raise ValueError("Invalid method")
            
            negotiated_ingredients_dict = {
                'sentiment': negotiated_ingredients if method == 'sentiment' else None,
                'perfect': negotiated_ingredients if method == 'perfect' else None,
                'no_feedback': negotiated_ingredients_start  
            }
            
            unavailable_ingredients_dict = {
                'sentiment': unavailable_ingredients if method == 'sentiment' else None,
                'perfect': unavailable_ingredients if method == 'perfect' else None,
                'no_feedback': unavailable_ingredients_start  
            }

            # Generate menu plan with the updated ingredients
            menu_plan_dict[method] = run_mod(model_name, negotiated_ingredients_dict[method], unavailable_ingredients_dict[method], menu_generators, ingredient_df, method, week=week, day=day)
            
            
            # Perform feedback analysis and calculate metrics
            if method == 'perfect':
                sentiment_analyzer = SentimentAnalyzer(
                    updated_preferences_with_feedback_perfect, true_child_preference_data, menu_plan_dict[method], child_data=child_feature_data, label_mapping=label_mapping_perfect, model_name='perfect', seed=seed
                )
                updated_preferences_with_feedback_perfect, _, perfect_feedback, _, _ = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)
                percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_preferences_with_feedback_perfect)
                utility_perfect_true, utility_perfect_predicted = utility_calculator_perfect.calculate_day_menu_utility(updated_known_and_predicted_preferences, list(menu_plan_dict[method].keys()))
                accuracy_unknown_perfect = accuracy_unknown
                accuracy_std_total_perfect = accuracy_std_total

            elif method == 'sentiment':
                sentiment_analyzer = SentimentAnalyzer(
                    updated_preferences_with_feedback_sentiment, true_child_preference_data, menu_plan_dict[method], child_data=child_feature_data, label_mapping=label_mapping_sentiment, model_name='Vader', seed=seed
                )
                updated_preferences_with_feedback_sentiment, sentiment_accuracy, sentiment_feedback, _, _ = sentiment_analyzer.get_sentiment_and_update_data(plot_confusion_matrix=False)
                percent_of_known_preferences_sentiment = calculate_percent_of_known_ingredients_to_unknown(updated_preferences_with_feedback_sentiment)
                utility_sentiment_true, utility_sentiment_predicted = utility_calculator_sentiment.calculate_day_menu_utility(updated_known_and_predicted_preferences, list(menu_plan_dict[method].keys()))
                accuracy_unknown_sentiment = sentiment_accuracy
                accuracy_std_total_sentiment = accuracy_std_total

            elif method == 'no_feedback':
                no_feedback_feedback = {} 
                percent_of_known_preferences_no_feedback = calculate_percent_of_known_ingredients_to_unknown(true_child_preference_data)
                utility_no_feedback_true, utility_no_feedback_predicted = utility_calculator_no_feedback.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_dict[method].keys()))
                accuracy_unknown_no_feedback = accuracy_unknown_no_feedback
                accuracy_std_total_no_feedback = accuracy_std_total_no_feedback
                
            # Evaluate the menu plan and calculate the reward
            if model_name != 'RL':
                reward, info = menu_generators[method][model_name].evaluator.select_ingredients(menu_plan_dict[method])
            else:
                evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients = negotiated_ingredients_dict[method], unavailable_ingredients = unavailable_ingredients_dict[method])
                reward, info = evaluator.select_ingredients(menu_plan_dict[method])
            
            # Store the results including time taken and reward
            results[method][str(menu)].update({
                'info': info,
                'reward': reward,
                'percent_of_known_preferences': locals()[f'percent_of_known_preferences_{method}'],
                'true_utility': locals()[f'utility_{method}_true'],
                'predicted_utility': locals()[f'utility_{method}_predicted'],
                'accuracy_unknown': locals()[f'accuracy_unknown_{method}'],
                'accuracy_std_total': locals()[f'accuracy_std_total_{method}'],
                'feedback': locals()[f'{method}_feedback']
            })
                
    # Close utility calculators
    _ = utility_calculator_perfect.close()
    _ = utility_calculator_sentiment.close()
    _ = utility_calculator_no_feedback.close()
        
    return results



def main():
    seed = random.randint(0, int(1e6))
    seed = 222 
    
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
    prediction_accuracies_unknown_start = accuracy_score(total_true_unknown_preferences, total_predicted_unknown_preferences)
    prediction_accuracies_std_total_start = print_preference_difference_and_accuracy(true_child_preference_data, updated_known_and_predicted_preferences_start)
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
    
    negotiated_ingredients_start, _, unavailable_ingredients_start = negotiator.negotiate_ingredients()
    evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients_start, unavailable_ingredients_start)
    
    week, day = 1, 0
    
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)

    # Creating instances once with the initial evaluators and updating them with the new negotiated ingredients and unavailable ingredients inside run_mod func
    menu_generators = {
        'perfect':
        {
        "genetic": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, include_preference=True, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
        "random": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        "best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "prob_best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed)
        },
        'sentiment':
        {
        "genetic": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, include_preference=True, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
        "random": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        "best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "prob_best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed)
        },
        'no_feedback':
        {
        "genetic": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, include_preference=True, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
        "random": RandomMenuGenerator(evaluator=evaluator, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        "best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "prob_best": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed)
        }
    }
    
    menu_gen_names = [
        # "random",
        # "prob",
        # "best",
        # "prob_best",
        "RL"
        # "genetic"
    ]
    
    try:
        results = {}
        with logging_redirect_tqdm():
            for model_name in tqdm(menu_gen_names, desc="Testing Menu Generator Types"):
                results[model_name] = run_menu_generation(seed, model_name, negotiated_ingredients_start, unavailable_ingredients_start, menu_generators, ingredient_df, true_child_preference_data, child_feature_data, label_mapping_start, prediction_accuracies_unknown_start, prediction_accuracies_std_total_start, updated_known_and_predicted_preferences_start)

                # Check elapsed time
                elapsed_time = time.time() - global_start_time
                if (elapsed_time > 35 * 3600):
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
