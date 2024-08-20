import logging
import os
import json
import random
import time
from sklearn.metrics import accuracy_score
from models.preferences.preference_utils import (
    get_child_data,
    initialize_child_preference_data,
)
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.menu_generators import RandomMenuGenerator, RLMenuGenerator
from models.preferences.random_menu_eval import MenuEvaluator
import numpy as np
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

initial_split = 0.95
menu_plan_length = 10

def run_menu_generation(seed):
    # Load data
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    true_child_preference_data = initialize_child_preference_data(
        child_feature_data, ingredient_df, split=initial_split, seed=seed, plot_graphs=False
    )

    with open(os.path.join(run_data_dir, 'true_child_preference_data.json'), 'w') as f:
        json.dump(true_child_preference_data, f)
        
    # Initial prediction of preferences
    file_path = os.path.join(run_graphs_dir, "preferences_visualization.png")
    predictor = PreferenceModel(
        ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, file_path=file_path, seed=seed
    )
    
    updated_known_and_predicted_preferences, _, _, _ = predictor.run_pipeline()
    
    # Initial negotiation of ingredients
    negotiator = IngredientNegotiator(
        seed, ingredient_df, updated_known_and_predicted_preferences, complex_weight_func_args, previous_feedback={}, previous_utility={}
    )
    
    negotiated_ingredients_simple, _, unavailable_ingredients = negotiator.negotiate_ingredients()
    evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients_simple, unavailable_ingredients)
    
    # Save negotiation results
    week, day = 1, 1
    negotiator.close(os.path.join(run_data_dir, "log_file.json"), week=week, day=day)

    menu_generators = {
        "genetic": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "random_random": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "random_refined": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='random', probability_best=0, seed=seed),
        "prob_refined": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0, seed=seed),
        # "prob_best_refined": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=0.5, seed=seed),
        "best_refined": RandomMenuGenerator(evaluator=evaluator, menu_plan_length=menu_plan_length, weight_type='score', probability_best=1, seed=seed),
        "RL": RLMenuGenerator(ingredient_df, menu_plan_length=menu_plan_length, seed=seed, model_save_path='rl_model'),
    }

    # Generate and evaluate menus, store results
    results = {}
    for menu_plan_num in range(menu_plan_length):
        results[menu_plan_num] = {}
        for name, generator in menu_generators.items():
            
            print(f"Running {name} menu generator for menu plan {menu_plan_num}")
            
            # Start timing
            start_time = time.time()

            if name == "genetic":
                menu_plan, _ = generator.run_genetic_menu_optimization_and_finalize(
                    negotiated_ingredients_simple, unavailable_ingredients, 
                    final_meal_plan_filename=f'final_meal_plan_{name}',
                    ngen=180, population_size=333, cxpb=0.742143, mutpb=0.3062819
                )
            elif name == "RL":
                menu_plan, _ = generator.generate_menu(
                    negotiated_ingredients_simple, unavailable_ingredients, 
                    final_meal_plan_filename=f'final_meal_plan_{name}', 
                    save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
                    week=week, day=day
                )
            else:
                menu_plan, _ = generator.generate_menu(
                    negotiated_ingredients_simple, unavailable_ingredients, 
                    quantity_type='genetic_algorithm' if 'refined' in name else 'random', 
                    final_meal_plan_filename=f'final_meal_plan_{name}', 
                    save_paths={'data': run_data_dir, 'graphs': run_graphs_dir}, 
                    week=week, day=day
                )

            # End timing
            end_time = time.time()

            # Calculate the time taken
            time_taken = end_time - start_time

            # Evaluate the menu plan and calculate the reward
            reward, info = evaluator.select_ingredients(menu_plan)
            
            # Store the results including time taken and reward
            results[menu_plan_num][name] = {
                'info': info,
                'reward': reward,
                'time_taken': time_taken
            }

            print(f"{name} menu plan generated in {time_taken:.2f} seconds with a reward of {reward}")

    return results

def main():
    all_results = []
    seed = random.randint(0, int(1e6))
    for i in range(1):
        iteration_results = run_menu_generation(seed)
        all_results.append(iteration_results)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.int32):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Save all results using the custom encoder
    with open(os.path.join(run_data_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder)
    
if __name__ == "__main__":
    main()
