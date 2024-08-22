import logging
import os
import json
import random
import time
from sklearn.metrics import accuracy_score
from models.preferences.preference_utils import (
    get_child_data,
    initialize_child_preference_data,)
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
    'use_feedback': False,
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
    
def run_mod(menu_generators, model_name, negotiated_ingredients, unavailable_ingredients, evaluator, ingredient_df, week, day, seed):
    if "genetic" in model_name:
        # Run initial genetic optimization
        menu_plan, _ = menu_generators[model_name].run_genetic_menu_optimization_and_finalize(
            negotiated_ingredients, unavailable_ingredients, 
            final_meal_plan_filename=f'final_meal_plan_{model_name}',
            ngen=180, population_size=333, cxpb=0.742143, mutpb=0.3062819
        )
    elif 'RL' in model_name:
        menu_plan, _ = menu_generators[model_name].generate_menu(
            negotiated_ingredients, unavailable_ingredients, 
            final_meal_plan_filename=f'final_meal_plan_{model_name}', 
            save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
            week=week, day=day
        )
    else:
        menu_plan, _ = menu_generators[model_name].generate_menu(
            negotiated_ingredients, unavailable_ingredients, 
            quantity_type='random', 
            final_meal_plan_filename=f'final_meal_plan_{model_name}', 
            save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
            week=week, day=day
        )
    return menu_plan

def run_menu_generation(seed, model_name="random"):
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=1, seed=seed, plot_graphs=False
    )

    with open(os.path.join(run_data_dir, 'true_child_preference_data.json'), 'w') as f:
        json.dump(true_child_preference_data, f)
    
    # Initial prediction of preferences
    file_path = os.path.join(run_graphs_dir, "preferences_visualization.png")
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, file_path=file_path, seed=seed
    )
    
    updated_known_and_predicted_preferences_start, _, _, label_encoder = predictor.run_pipeline()
    
    label_mapping_start = {label: index for index, label in enumerate(label_encoder.classes_)}
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
    
    # Negotiation will be the same at the start as feedback has not been provided
    negotiated_ingredients_simple_start, _, unavailable_ingredients_start = negotiator.negotiate_ingredients(simple_only=True)
    
    evaluator_start = MenuEvaluator(ingredient_df, negotiated_ingredients_simple_start, unavailable_ingredients_start)
        
    # Save negotiation results
    week, day = 1, 1
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)
    
    # Initialize menu generators once
    menu_generators = {
        "genetic": RandomMenuGenerator(evaluator=evaluator_start, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, include_preference=True, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
        "random": RandomMenuGenerator(evaluator=evaluator_start, include_preference=True, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob": RandomMenuGenerator(evaluator=evaluator_start, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        "best": RandomMenuGenerator(evaluator=evaluator_start, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "prob_best": RandomMenuGenerator(evaluator=evaluator_start, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed),
    }

    menu_plan_start = run_mod(menu_generators, model_name, negotiated_ingredients_simple_start, unavailable_ingredients_start, evaluator_start, ingredient_df, week, day, seed)   

    results = {}
    
    # Initialize utility calculator
    json_path_simple = os.path.join(run_data_dir, "menu_utilities_simple.json")
    json_path_complex = os.path.join(run_data_dir, "menu_utilities_complex.json")
    
    utility_calculator_simple = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_simple)
    utility_calculator_complex = MenuUtilityCalculator(true_child_preference_data, child_feature_data, menu_plan_length=menu_plan_length, save_to_json=json_path_complex)
    
    info = {}
    reward = {}
    
    # Evaluate the menu plan and calculate the reward
    reward1, info1 = evaluator_start.select_ingredients(menu_plan_start)
    
    info['simple'] = info1
    reward['simple'] = reward1
    info['complex'] = info1
    reward['complex'] = reward1

    # Store the results including time taken and reward
    results['0'] = {
        'info': info,
        'reward': reward,
    }
    
    # Using start prediction preference before feedback
    _ = utility_calculator_simple.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_start.keys()))
    predicted_utility_complex = utility_calculator_complex.calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plan_start.keys())) 
    
    with logging_redirect_tqdm():
        for menu in tqdm(range(1, 10), desc=f"Processing Menus for {model_name}"):
            results[str(menu)] = {}

            negotiator_simple = IngredientNegotiator(
                seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback={}, previous_utility={}
            )
                    
            negotiated_ingredients_simple, _, unavailable_ingredients_simple = negotiator_simple.negotiate_ingredients(simple_only=True)
            
            negotiator_complex = IngredientNegotiator(
                seed, ingredient_df, updated_known_and_predicted_preferences_start, complex_weight_func_args, previous_feedback=predicted_utility_complex, previous_utility=predicted_utility_complex
            )
            
            _, negotiated_ingredients_complex, unavailable_ingredients_complex = negotiator_complex.negotiate_ingredients(simple_only=False)

            negotiated_ingredients_dict = {
                'simple': negotiated_ingredients_simple,
                'complex': negotiated_ingredients_complex,  
            }
            
            unavailable_ingredients_dict = {
                'simple': unavailable_ingredients_simple,
                'complex': unavailable_ingredients_complex,   
            }
            
            evaluator_dict = {
                'simple': evaluator_start,
                'complex': evaluator_start,
            }
            
            utility_dict = {
                'simple': utility_calculator_simple,
                'complex': utility_calculator_complex,
            }
       
            menu_plans = {}
            reward = {}
            info = {}
            
            for method in ['simple', 'complex']:
                menu_plans[method] = run_mod(menu_generators, model_name, negotiated_ingredients_dict[method], unavailable_ingredients_dict[method], evaluator_dict[method], ingredient_df, week, day, seed)
                
                reward1, info1 = evaluator_dict[method].select_ingredients(menu_plans[method])
                
                info[method] = info1
                reward[method] = reward1
                
                # Store the results including time taken and reward
                results[str(menu)].update({
                    'info': info,
                    'reward': reward,
                })
                
                if method == 'complex':
                    predicted_utility_complex = utility_dict[method].calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plans[method].keys()))
                else:
                    _ = utility_dict[method].calculate_day_menu_utility(updated_known_and_predicted_preferences_start, list(menu_plans[method].keys()))

                utility_results = {}
                utility_results[method] = utility_dict[method].close()
                
                results[str(menu)]["utility_results"] = utility_results

    return results


import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging to only capture errors and critical messages
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    seed = 300
    
    menu_generators = [
        "random",
        # "prob",
        "best",
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
                if elapsed_time > 35 * 3600:
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

# Run 67 just RL hpc the others