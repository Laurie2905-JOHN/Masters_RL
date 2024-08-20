import optuna
import random
import numpy as np
import os
from models.preferences.random_menu_eval import MenuEvaluator
from sklearn.metrics import accuracy_score
from models.preferences.preference_utils import (
    get_child_data,
    initialize_child_preference_data,
)
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.menu_generators import RandomMenuGenerator
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances
)

# Set up paths and constants
seed = None
menu_plan_length = 5
weight_type = 'random'
probability_best = 1
save_dir = os.path.join(os.getcwd(), 'saved_models')

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Complex weight function arguments
complex_weight_func_args = {
    'use_normalize_total_voting_weight': False,
    'use_normalize_vote_categories': True,
    'use_compensatory': True,
    'use_feedback': True,
    'use_fairness': True,
    'target_gini': 0.15,
}

# Load data
ingredient_df = get_data("data.csv")
child_feature_data = get_child_data()
true_child_preference_data = initialize_child_preference_data(
    child_feature_data, ingredient_df, split=0.35, seed=seed, plot_graphs=False
)

# Define the objective function for Optuna
def objective(trial):
    # Suggest values for the parameters to optimize
    ngen = trial.suggest_int('ngen', 10, 50)  # Number of generations
    population_size = trial.suggest_int('population_size', 100, 1000)  # Population size
    cxpb = trial.suggest_float('cxpb', 0.5, 0.9)  # Crossover probability
    mutpb = trial.suggest_float('mutpb', 0.1, 0.5)  # Mutation probability
    
    negotiation_scores = []
    
    for negotiation_run in range(1):  # Perform 5 different negotiations
        # Generate a new random seed for each negotiation
        seed = random.randint(0, int(1e6))

        true_child_preference_data = initialize_child_preference_data(
            child_feature_data, ingredient_df, split=0.35, seed=seed, plot_graphs=False
        )

        # Run the pipeline with the updated seed
        predictor = PreferenceModel(
            ingredient_df, child_feature_data, true_child_preference_data, visualize_data=False, file_path='', seed=seed
        )
        updated_known_and_predicted_preferences, total_true_unknown_preferences, total_predicted_unknown_preferences, label_encoder = predictor.run_pipeline()

        # Initial negotiation of ingredients
        negotiator = IngredientNegotiator(
            seed, ingredient_df, updated_known_and_predicted_preferences, complex_weight_func_args, previous_feedback={}, previous_utility={}
        )
        negotiated_ingredients_simple, negotiated_ingredients_complex, unavailable_ingredients = negotiator.negotiate_ingredients()

        # Instantiate the MenuEvaluator
        evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients_simple, unavailable_ingredients)
        
        # Instantiate the RandomMenuGenerator with the updated seed
        menu_generator_simple = RandomMenuGenerator(
            evaluator,
            menu_plan_length=10,
            weight_type=weight_type,
            probability_best=probability_best,
            seed=seed
        )

        # Perform 5 runs for each negotiation
        scores = []
        for run in range(3):

            # Run the genetic algorithm
            optimized_menu, score = menu_generator_simple.genetic_algorithm_select_and_optimize(
                negotiated_ingredients_simple,
                unavailable_ingredients,
                ngen=ngen,
                population_size=population_size,
                cxpb=cxpb,
                mutpb=mutpb
            )
            
            scores.append(score)
        
        # Calculate the median score for this negotiation
        median_score = np.median(scores)
        negotiation_scores.append(median_score)
    
    # Return the median score across all negotiations
    return np.median(negotiation_scores)

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)  # Adjust the number of trials as needed

# Print the best parameters and the best value (score)
print(f"Best Parameters: {study.best_params}")
print(f"Best Score: {study.best_value}")

# Save the plots to the specified directory
plot_optimization_history(study).write_image(os.path.join(save_dir, 'optimization_history.png'))
plot_parallel_coordinate(study).write_image(os.path.join(save_dir, 'parallel_coordinate.png'))
plot_param_importances(study).write_image(os.path.join(save_dir, 'param_importances.png'))

# Optional: Show the plots (useful for quick checks during development)
# plot_optimization_history(study).show()
# plot_parallel_coordinate(study).show()
# plot_param_importances(study).show()
