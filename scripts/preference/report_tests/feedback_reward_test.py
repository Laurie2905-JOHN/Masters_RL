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
    'use_compensatory': False,
    'use_feedback': True,
    'use_fairness': False,
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
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
    
    negotiated_ingredients_start, _, unavailable_ingredients_start = negotiator.negotiate_ingredients(simple_only=False)
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
        
    results = {}

    menu_plan = run_mod(model_name)
        
    # Feedback methods
    sentiment_analyzer = SentimentAnalyzer(
        true_child_preference_data, menu_plan, child_data=child_feature_data, label_mapping = label_mapping_start, model_name='perfect', seed=seed
    )
    
    # Get updated preferences from feedback and sentiment analysis for both menu plans
    updated_known_unknown_preferences_with_feedback, _, _, _, _ = sentiment_analyzer_perfect.get_sentiment_and_update_data(plot_confusion_matrix=False)
    
    percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_perfect)
    
    # Initialize utility calculator
    json_path_simple = os.path.join(run_data_dir, "menu_utilities_simple.json")
    json_path_complex = os.path.join(run_data_dir, "menu_utilities_complex.json")
    
    utility_calculator_simple = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_simple)
    utility_calculator_complex = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_complex)
    
    info = {}
    reward = {}
    
    # Evaluate the menu plan and calculate the reward
    reward1, info1 = evaluator.select_ingredients(menu_plan)
    
    info['simple'] = info1
    reward['simple'] = reward1
    info['complex'] = info1
    reward['complex'] = reward1
    # Generate and evaluate menus, store results
    
    # Store the results including time taken and reward
    results['0'] = {
        'info': info,
        'reward': reward,
        'percent_of_known_preferences_perfect': percent_of_known_preferences_perfect,
    }
    
    # Using start prediction preference before feedback
    _ = utility_calculator_simple.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan.keys()))
    _ = utility_calculator_complex.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan.keys())) 
    
    with logging_redirect_tqdm():
        
        for menu in tqdm(range(1, 100), desc=f"Processing Menus for {model_name}"):
            results[str(menu)] = {}
            # Prediction of preferences based on expected preferences from sentiment analysis
            predictor_simple = PreferenceModel(
                ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_sentiment, visualize_data=False, seed=seed
            )
            predictor_complex = PreferenceModel(
                ingredient_df, child_feature_data, updated_known_unknown_preferences_with_feedback_perfect, visualize_data=False, seed=seed
            )
            
            # Prediction with no feedback can use starting preference prediction as same each time
            # predictor_no_feedback = PreferenceModel(
            #     ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, seed=seed
            # )
            
            updated_known_and_predicted_preferences_perfect, _, _, label_encoder = predictor_perfect.run_pipeline()
            
            label_mapping_perfect = {label: index for index, label in enumerate(label_encoder.classes_)}
            

            
            negotiator_perfect = IngredientNegotiator(
                seed, ingredient_df, updated_known_and_predicted_preferences_perfect, complex_weight_func_args, previous_feedback={}, previous_utility={}
            )
                    

            negotiated_ingredients_perfect, _, unavailable_ingredients_perfect = negotiator_perfect.negotiate_ingredients(simple_only=False)
            
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

            sentiment_name = 'perfect'
            
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
                
            sentiment_analyzer_perfect = SentimentAnalyzer(
                true_child_preference_data, menu_plans['perfect'], child_data=child_feature_data, label_mapping = label_mapping_perfect, model_name='perfect', seed=seed
            )
            updated_known_unknown_preferences_with_feedback_perfect, _, _, _, _ = sentiment_analyzer_perfect.get_sentiment_and_update_data(plot_confusion_matrix=False)
            percent_of_known_preferences_perfect = calculate_percent_of_known_ingredients_to_unknown(updated_known_unknown_preferences_with_feedback_perfect)

            # Store the results including time taken and reward
            results[str(menu)] = {
                'info': info,
                'reward': reward,
                'percent_of_known_preferences_perfect': percent_of_known_preferences_perfect,
            }
                
            utility_results = {}
            utility_results['perfect'] = utility_calculator_perfect.close()
                
            results["utility_results"] = utility_results
    
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
        # "prob",
        # "best",
        # "prob_best"
        # "RL",
        # "genetic",
    ]
    
    try:
        results = {}
        with logging_redirect_tqdm():
            for model_name in tqdm(menu_generators, desc="Testing Menu Generator Types"):
                results[model_name] = run_menu_generation(seed, model_name)

                # Check elapsed time
                elapsed_time = time.time() - global_start_time
                if elapsed_time > 24 * 3600:
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