import random
from typing import Dict, List, Set, Optional, Callable
import matplotlib.pyplot as plt
from collections import Counter
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure, HumanOutputFormat
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from models.callbacks.callback import *
import argparse
import os
import json
from scipy.optimize import dual_annealing
from deap import base, creator, tools, algorithms

from utils.train_utils import (
    setup_environment, 
    get_unique_directory, 
    load_yaml, 
    set_default_prefixes, 
    get_activation_fn, 
    select_device, 
    set_seed, 
    ensure_dir_exists, 
    load_model, 
    create_model,
    save_hyperparams,
)
from sb3_contrib.common.maskable.utils import get_action_masks


class BaseMenuGenerator(ABC):
    def __init__(self, menu_plan_length: int = 10, weight_type: str = None, probability_best: float = 0.5, seed: Optional[int] = None):
        """
        Initializes the BaseMenuGenerator with an optional seed for randomization.
        
        :param menu_plan_length: Length of the menu plan.
        :param weight_type: Type of weighting to use.
        :param probability_best: Probability of selecting the best item.
        :param seed: An optional seed for random number generation.
        """
        self.seed = seed
        self.random = random.Random(seed)
        self.generated_count = 1
        self.menu_plan_length = menu_plan_length
        self.menu_counter = Counter()
        self.selected_ingredients = set()
        self.weight_type = weight_type
        self.probability_best = probability_best
        
        # Only remove from these groups as other groups do not have the count number to remove from
        self.groups_to_remove_from = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E']

    def plot_top_menus(self, top_n: int = 10, save_path: str = 'top_menus.png') -> None:
        """
        Plots the top N generated menus and their frequencies as a percentage.
        
        :param top_n: Number of top menus to plot.
        :param save_path: Path to save the bar chart.
        """
        # Get the most common menus
        top_menus = self.menu_counter.most_common(top_n)
        
        # Calculate total count of all generated menus
        total_menus = sum(self.menu_counter.values())
        
        # Prepare data for plotting
        menus, counts = zip(*top_menus)
        percentages = [count / total_menus * 100 for count in counts]
        
        # Create labels for the menus
        menu_labels = ['\n'.join([f"{group}: {item}" for group, item in menu]) for menu in menus]
        
        # Plotting
        plt.figure(figsize=(14, 10))
        plt.barh(menu_labels, percentages, color='skyblue')
        plt.xlabel('Percentage of Total Menus Generated')
        plt.ylabel('Menu')
        plt.title(f'Top {top_n} Generated Menus')
        plt.gca().invert_yaxis()  # Highest percentage at the top
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @abstractmethod
    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, final_meal_plan_filename: str = 'NA') -> Dict[str, str]:
        pass

from models.preferences.random_menu_eval import MenuEvaluator


class RandomMenuGenerator(BaseMenuGenerator):
    def __init__(self, evaluator: MenuEvaluator, include_preference: bool = True, menu_plan_length: int = 10, weight_type: str = None, probability_best: float = 0.5, plot_menu_flag: bool = False, seed: Optional[int] = None):
        super().__init__(menu_plan_length, weight_type, probability_best, seed)
        self.evaluator = evaluator
        
        self.ingredient_group_portion_targets = {
            'Group A fruit': (40, 130),
            'Group A veg': (40, 80),
            'Group BC': (40, 90),
            'Group D': (40, 150),
            'Group E': (50, 150),
            'Bread': (50, 70),
        }
        
        self.ingredient_groups_to_remove = ['Misc', 'Confectionary']
        
        self.plot_menu_flag = plot_menu_flag

        self.include_preference = include_preference
        
    def update_evaluator(self, ingredient_df, negotiated: Dict = {}, unavailable: set = {}) -> None:
        """
        Updates the evaluator used for selecting ingredients and generating menus.
        
        :param evaluator: The new MenuEvaluator instance to use.
        """
        
        self.evaluator = MenuEvaluator(ingredient_df, negotiated_ingredients = negotiated, unavailable_ingredients = unavailable)  
        
    def get_evaluator(self) -> MenuEvaluator:
        """
        Returns the evaluator used for selecting ingredients and generating menus.
        
        :return: The MenuEvaluator instance.
        """
        return self.evaluator
                
    def normalize_scores(self) -> None:
        """
        Normalizes the scores of the ingredients within each group to ensure they sum up to 1.
        """
        self.normalized_scores = {group: [] for group in self.ingredient_groups}
        for group in self.ingredient_groups:
            total_score = self.total_scores[group]
            if total_score > 0:
                for score in self.ingredients_scores[group]:
                    self.normalized_scores[group].append(score / total_score)
            else:
                if len(self.ingredients_scores[group]) == 0:
                    self.normalized_scores[group] = [0]
                else:
                    self.normalized_scores[group] = [1 / len(self.ingredients_scores[group])] * len(self.ingredients_scores[group])

    def initialize_ingredient_in_groups(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]]) -> None:
        """
        Initializes the ingredients and their scores for each group, excluding unavailable ingredients,
        and normalizes the scores. Also makes a copy of the original ingredients and scores to reset after 10 generations.
        """
        self.negotiated = negotiated
        self.unavailable = unavailable or set()
        
        self.ingredient_groups = list(negotiated.keys())
        
        # Remove specific values from the list
        for group in self.ingredient_groups_to_remove:
            if group in self.ingredient_groups:
                self.ingredient_groups.remove(group)
            if group in self.negotiated:
                del self.negotiated[group]
                
        # Initialize dictionaries
        self.ingredients_in_groups = {group: [] for group in self.ingredient_groups}
        self.ingredients_scores = {group: [] for group in self.ingredient_groups}
        self.total_scores = {group: 0 for group in self.ingredient_groups}

        # Populate dictionaries and normalize scores
        for group, items in self.negotiated.items():
            for ingredient, score in items.items():
                if ingredient not in self.unavailable and ingredient not in self.selected_ingredients:
                    self.ingredients_in_groups[group].append(ingredient)
                    self.ingredients_scores[group].append(score)
                    self.total_scores[group] += score

        # Normalize scores initially
        self.normalize_scores()

        # Make a copy of the original ingredients list and scores to reset after 10 generations
        self.original_ingredients_in_groups = {group: items.copy() for group, items in self.ingredients_in_groups.items()}
        self.original_ingredients_scores = {group: scores.copy() for group, scores in self.ingredients_scores.items()}
        self.original_total_scores = self.total_scores.copy()

    def generate_best_item(self, group: str) -> str:
        """
        Generates the best item from a specified group based on the highest score.
        """
        if len(self.ingredients_in_groups[group]) == 0:
            raise ValueError(f"No ingredients available in group {group}")
        max_score_index = self.ingredients_scores[group].index(max(self.ingredients_scores[group]))
        return self.ingredients_in_groups[group][max_score_index]

    def generate_random_item(self, group: str, num_items: int = 1) -> List[str]:
        """
        Generates a list of random items from a specified group based on the normalized scores.
        """
        if len(self.ingredients_in_groups[group]) < num_items:
            raise ValueError(f"Not enough ingredients in group {group} to sample {num_items} items")
        items = self.ingredients_in_groups[group]
        if self.weight_type == "score":
            weights = self.normalized_scores[group]
        elif self.weight_type == "random":
            weights = None
            
        return self.random.choices(items, weights=weights, k=num_items)

    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, quantity_type: str = 'random', final_meal_plan_filename: str = 'NA', save_paths=None, week=0, day=0) -> Dict[str, float]:
        """
        Generates a menu by selecting an item from each ingredient group with a certain probability of choosing
        the best item and a complementary probability of choosing a random item. If no ingredients are available,
        it resets the ingredients.
        """
        self.initialize_ingredient_in_groups(negotiated, unavailable)

        if self.generated_count > self.menu_plan_length:
            print(f"\nGenerated {self.menu_plan_length} meal plans, resetting ingredients.")
            self.reset_ingredients()

        menu = {}
        for group in self.ingredient_groups:
            try:
                if self.random.random() < self.probability_best:
                    item = self.generate_best_item(group)
                else:
                    item = self.generate_random_item(group, 1)[0]
                menu[group] = item
                if group in self.groups_to_remove_from:
                    self.selected_ingredients.add(item)
            except Exception as e:
                raise ValueError(f"Error generating item for group {group}: {e}")

        # Generate quantities using the specified method and map them to ingredients
        quantities, score = self.generate_quantities(menu, method=quantity_type)

        ingredient_quantities = {ingredient: quantities[ingredient] for group, ingredient in menu.items()}
        print(f"Generated ingredient quantities ({quantity_type}): {ingredient_quantities} with score {score}")

        # Convert the menu dictionary to a tuple to track frequency
        menu_tuple = tuple(sorted(menu.items()))
        self.menu_counter[menu_tuple] += 1

        self.generated_count += 1

        if "complex" in final_meal_plan_filename:
            type = "complex"
        else:
            type = "simple"

        # Save data to JSON
        data_to_save = {
            'week': week,
            'day': day,
            'type': type,
            'meal_plan': list(menu.values()),
            'quantities': ingredient_quantities  # Save the generated ingredient quantities
        }

        if save_paths and 'data' in save_paths:
            json_filename = os.path.join(save_paths['data'], 'Random_menu_results.json')

            if os.path.exists(json_filename):
                with open(json_filename, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = []

                # Append new data
                existing_data.append(data_to_save)

                # Save updated data
                with open(json_filename, 'w') as json_file:
                    json.dump(existing_data, json_file, indent=4)
        
        _, info = self.evaluator.select_ingredients(ingredient_quantities, self.include_preference)
        
        if self.plot_menu_flag:
            self.evaluator.plot_menu(info)

        return ingredient_quantities, score

    def generate_quantities(self, menu: Dict[str, str], method: str = "random") -> Dict[str, float]:
        """
        Generates quantities for the selected menu items using a specified method.
        """
        if method == "random":
            return self.random_quantity_within_bounds(menu=menu)
        elif method == "simulated_annealing":
            return self.simulated_annealing_quantities(menu=menu)
        elif method == "genetic_algorithm":
            return self.genetic_algorithm_quantities(menu=menu)
        elif method == "particle_swarm":
            return self.particle_swarm_quantities(menu=menu)
        elif method == "bayesian":
            return self.bayesian_quantities(menu=menu)
        elif method == "hill_climbing":
            return self.hill_climbing_quantities(menu=menu)
        else:
            raise ValueError(f"Unknown method: {method}")


    def get_group_from_ingredient(self, ingredient: str, menu: Dict[str, str]) -> str:
        """
        Get the group name for a given ingredient.
        
        :param ingredient: The ingredient name.
        :param menu: The menu dictionary with groups and ingredients.
        :return: The group name corresponding to the ingredient.
        """
        for group, ing in menu.items():
            if ing == ingredient:
                return group
        raise ValueError(f"Ingredient '{ingredient}' not found in the menu.")

    def random_quantity_within_bounds(self, menu: Dict[str, str]) -> Dict[str, float]:
        """
        Generate random quantities within the defined bounds for each ingredient in the menu.
        
        :param menu: The menu dictionary with groups and ingredients.
        :return: A dictionary with ingredients as keys and random quantities as values.
        """
        quantities = {}
        for group, ingredient in menu.items():
            min_portion, max_portion = self.ingredient_group_portion_targets[group]
            quantities[ingredient] = random.randint(min_portion, max_portion)
        best_quantities = quantities.copy()
        score = self.evaluator.objective_function(best_quantities, self.include_preference)
        return quantities, score

    def simulated_annealing_quantities(self, menu) -> Dict[str, float]:
        """
        Optimizes the quantities using scipy's dual_annealing.
        """
        bounds = [(self.ingredient_group_portion_targets[self.get_group_from_ingredient(ingredient, menu)][0], 
                self.ingredient_group_portion_targets[self.get_group_from_ingredient(ingredient, menu)][1]) 
                for ingredient in menu.values()]

        def objective_function_wrapper(quantities):
            quantities_dict = dict(zip(menu.values(), quantities))
            return -self.evaluator.objective_function(quantities_dict, self.include_preference)  # Minimization in dual_annealing

        result = dual_annealing(objective_function_wrapper, bounds)
        
        best_quantities = dict(zip(menu.values(), result.x))
        best_score = -result.fun  # Converting back to maximization
        
        return best_quantities, best_score

    def genetic_algorithm_quantities(self, menu) -> Dict[str, float]:
        """
        Optimizes the quantities using a Genetic Algorithm.
        """
        
        # Define the individual and fitness classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Flatten the menu structure for easier processing with DEAP
        ingredients = [(group, ingredient) for group, ingredient in menu.items()]
        
        # Define individual creation function
        def create_individual():
            quantities, _ = self.random_quantity_within_bounds(menu=menu)
            return creator.Individual([quantities[ingredient] for group, ingredient in menu.items()])

        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=300)
        toolbox.register("evaluate", lambda ind: (
            self.evaluator.objective_function({ingredient: qty for (group, ingredient), qty in zip(menu.items(), ind)}, self.include_preference),
        ))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize population
        population = toolbox.population()

        # Evolve the population
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)

        # Get the best individual
        best_individual = tools.selBest(population, k=1)[0]
        best_quantities = {ingredient: qty for (group, ingredient), qty in zip(menu.items(), best_individual)}
        best_score = self.evaluator.objective_function(best_quantities, self.include_preference)

        return best_quantities, best_score

    def particle_swarm_quantities(self, menu) -> Dict[str, float]:
        """
        Optimizes the quantities using Particle Swarm Optimization.
        """
        from pyswarm import pso
        
        # Flatten the menu structure for easier processing
        ingredients = [ingredient for group, ingredient in menu.items()]
        
        # Objective function for PSO
        def objective_function(quantities):
            quantities_dict = dict(zip(ingredients, quantities))
            return -self.evaluator.objective_function(quantities_dict, self.include_preference)
        
        # Define bounds for each ingredient quantity
        lower_bounds = [self.ingredient_group_portion_targets[group][0] for group in menu.keys()]
        upper_bounds = [self.ingredient_group_portion_targets[group][1] for group in menu.keys()]
        
        # Perform PSO
        best_quantities_list, best_score = pso(objective_function, lower_bounds, upper_bounds)
        
        # Reconstruct the best quantities dictionary
        best_quantities = dict(zip(ingredients, best_quantities_list))
        
        return best_quantities, -best_score

    
    def bayesian_quantities(self, menu) -> Dict[str, float]:
        """
        Optimizes the quantities using Bayesian Optimization.
        """
        from skopt import gp_minimize
        
        # Flatten the menu structure for easier processing
        ingredients = [ingredient for group, ingredient in menu.items()]
        
        # Objective function for Bayesian Optimization
        def objective_function(quantities):
            quantities_dict = dict(zip(ingredients, quantities))
            return -self.evaluator.objective_function(quantities_dict, self.include_preference)
        
        # Define bounds for each ingredient quantity
        bounds = [(self.ingredient_group_portion_targets[group][0], 
                self.ingredient_group_portion_targets[group][1]) for group in menu.keys()]
        
        # Perform Bayesian Optimization
        result = gp_minimize(objective_function, bounds, n_calls=50)
        
        # Reconstruct the best quantities dictionary
        best_quantities = dict(zip(ingredients, result.x))
        
        return best_quantities, -result.fun

    
    def hill_climbing_quantities(self, menu) -> Dict[str, float]:
        """
        Optimizes the quantities using Hill Climbing.
        """
        # Flatten the menu structure for easier processing
        ingredients = [ingredient for group, ingredient in menu.items()]
        
        # Initial random quantities within bounds
        quantities, _ = self.random_quantity_within_bounds(menu=menu)
        best_quantities = {ingredient: quantities[ingredient] for ingredient in ingredients}
        best_score = self.evaluator.objective_function(best_quantities, self.include_preference)
        
        improved = True
        while improved:
            improved = False
            for ingredient in best_quantities:
                group = self.get_group_from_ingredient(ingredient, menu)
                for delta in [-1, 1]:
                    new_quantities = best_quantities.copy()
                    new_quantities[ingredient] = max(
                        min(best_quantities[ingredient] + delta, self.ingredient_group_portion_targets[group][1]), 
                        self.ingredient_group_portion_targets[group][0]
                    )
                    new_score = self.evaluator.objective_function(new_quantities, self.include_preference)
                    
                    if new_score > best_score:
                        best_quantities, best_score = new_quantities, new_score
                        improved = True
        
        return best_quantities, best_score

    def run_genetic_menu_optimization_and_finalize(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, ngen: int = 100, population_size: int = 500, cxpb: float = 0.7, mutpb: float = 0.3, final_meal_plan_filename: str = 'NA', save_paths=None, week=0, day=0) -> Dict[str, float]:
        """
        Runs the full optimization process and finalizes the selected menu in a single function.

        :param negotiated: A dictionary of ingredient groups with corresponding ingredients and their scores.
        :param unavailable: A set of ingredients that are unavailable and should not be selected.
        :param ngen: Number of generations for the genetic algorithm.
        :param population_size: Size of the population in each generation.
        :param cxpb: Probability of crossover.
        :param mutpb: Probability of mutation.
        :param final_meal_plan_filename: The filename to save the final meal plan.
        :param save_paths: Paths to save the generated data and graphs.
        :param week: The current week number.
        :param day: The current day number.
        :return: A dictionary with selected ingredients and their optimized quantities.
        """
        
        self.initialize_ingredient_in_groups(negotiated, unavailable)
        
        # Step 1: Generate the optimized menu
        best_individual = self.generate_optimized_genetic_menu(negotiated, unavailable, ngen, population_size, cxpb, mutpb)

        # Step 2: Finalize the menu and return the results
        return self.finalize_genetic_menu(best_individual, negotiated, save_paths, week, day, final_meal_plan_filename)

    def finalize_genetic_menu(self, best_individual, negotiated: Dict[str, Dict[str, float]], save_paths=None, week=0, day=0, final_meal_plan_filename: str = 'NA'):
        """
        Finalizes the selected menu by adding ingredients to the unavailable list and saving the results.

        :param best_individual: The best individual (optimized menu) from the genetic algorithm.
        :param negotiated: A dictionary of ingredient groups with corresponding ingredients and their scores.
        :param save_paths: Paths to save the generated data and graphs.
        :param week: The current week number.
        :param day: The current day number.
        :param final_meal_plan_filename: The filename to save the final meal plan.
        :return: A dictionary with selected ingredients and their optimized quantities.
        """

        # Flatten the menu structure for easier processing with DEAP
        ingredients_per_group = [(group, items) for group, items in self.ingredients_in_groups.items()]
        
        # Create the final menu with selected ingredients and optimized quantities
        final_menu = {group: ingredient for (group, _), (ingredient, _) in zip(ingredients_per_group, best_individual)}
        final_quantities = {ingredient: quantity for (ingredient, quantity) in best_individual}

        # Add the selected ingredients to the unavailable list
        for group, ingredient in final_menu.items():
            if group in self.groups_to_remove_from:
                self.selected_ingredients.add(ingredient)

        # Save data to JSON
        data_to_save = {
            'week': week,
            'day': day,
            'type': "complex" if "complex" in final_meal_plan_filename else "simple",
            'meal_plan': list(final_menu.values()),
            'quantities': final_quantities  # Save the generated ingredient quantities
        }

        if save_paths and 'data' in save_paths:
            json_filename = os.path.join(save_paths['data'], 'Random_menu_results.json')

            if os.path.exists(json_filename):
                with open(json_filename, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = []

            # Append new data
            existing_data.append(data_to_save)

            # Save updated data
            with open(json_filename, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
        
        # Optional: Handle plotting or other finalization tasks
        if self.generated_count >= self.menu_plan_length:
            print(f"\nGenerated {self.menu_plan_length} meal plans, resetting ingredients.")
            self.reset_ingredients()

        # Increment the count of generated menus
        self.generated_count += 1

        # Plotting the menu
        _, info = self.evaluator.select_ingredients(final_quantities, self.include_preference)
        if self.plot_menu_flag:
            self.evaluator.plot_menu(info)

        return final_quantities, self.evaluator.objective_function(final_quantities, self.include_preference)

    def generate_optimized_genetic_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, ngen: int = 100, population_size: int = 500, cxpb: float = 0.7, mutpb: float = 0.3) -> Dict[str, float]:
        """
        Generates and optimizes a menu using a Genetic Algorithm.

        :param negotiated: A dictionary of ingredient groups with corresponding ingredients and their scores.
        :param unavailable: A set of ingredients that are unavailable and should not be selected.
        :param ngen: Number of generations for the genetic algorithm.
        :param population_size: Size of the population in each generation.
        :param cxpb: Probability of crossover.
        :param mutpb: Probability of mutation.
        :return: The best individual (optimized menu) from the genetic algorithm.
        """

        # Ensure `unavailable` is initialized properly
        if unavailable is None:
            unavailable = set()

        # Initialize the ingredients in groups and normalize scores
        self.initialize_ingredient_in_groups(negotiated, unavailable)
        
        # Flatten the menu structure for easier processing with DEAP
        ingredients_per_group = [(group, items) for group, items in self.ingredients_in_groups.items()]

        # Define the individual and fitness classes for DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Define a function to create an individual by selecting an ingredient and generating a quantity for each group
        def create_individual():
            individual = []
            for group, ingredients in ingredients_per_group:
                if not ingredients:
                    raise ValueError(f"No available ingredients in group {group} after filtering unavailable ingredients.")
                selected_ingredient = self.random.choice(ingredients)
                min_portion, max_portion = self.ingredient_group_portion_targets[group]
                selected_quantity = self.random.randint(min_portion, max_portion)
                individual.append((selected_ingredient, selected_quantity))
            return creator.Individual(individual)

        # Create a toolbox for the Genetic Algorithm
        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        # Evaluation function: calculate the score based on the selected ingredients and generated quantities
        def evaluate(individual):
            # Convert the individual into a menu dictionary with quantities
            menu = {group: ingredient for (group, _), (ingredient, _) in zip(ingredients_per_group, individual)}
            quantities = {ingredient: quantity for (ingredient, quantity) in individual}

            # Evaluate the menu and quantities using the provided evaluator
            score = self.evaluator.objective_function(quantities, self.include_preference)
            return score,

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_individual, indpb=mutpb)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Initialize population
        population = toolbox.population()

        # Evolve the population for a longer period
        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

        # Get the best individual (ingredient selection and quantity)
        best_individual = tools.selBest(population, k=1)[0]

        return best_individual

    
    def mutate_individual(self, individual, indpb):
        """
        Mutates an individual by either changing the selected ingredient or adjusting the quantity.

        :param individual: The individual to mutate.
        :param indpb: Independent probability for each attribute to be mutated.
        :return: A tuple containing the mutated individual and a boolean indicating whether any mutation occurred.
        """
        for i, (ingredient, quantity) in enumerate(individual):
            group = None
            for g, ingredients in self.ingredients_in_groups.items():
                if ingredient in ingredients:
                    group = g
                    break

            if group is None:
                raise ValueError(f"Ingredient '{ingredient}' not found in any group.")

            if self.random.random() < indpb:
                new_ingredient = self.random.choice(self.ingredients_in_groups[group])
                individual[i] = (new_ingredient, quantity)

            if self.random.random() < indpb:
                min_portion, max_portion = self.ingredient_group_portion_targets[group]
                new_quantity = self.random.randint(min_portion, max_portion)
                individual[i] = (ingredient, new_quantity)

        return individual,

    def reset_ingredients(self) -> None:
        """
        Resets the ingredients and their scores to the original state and normalizes the scores.
        """
        self.selected_ingredients.clear()
        self.ingredients_in_groups = {group: items.copy() for group, items in self.original_ingredients_in_groups.items()}
        self.ingredients_scores = {group: scores.copy() for group, scores in self.original_ingredients_scores.items()}
        self.total_scores = self.original_total_scores.copy()
        self.normalize_scores()
        self.generated_count = 1



class RLMenuGenerator(BaseMenuGenerator):
    def __init__(self, ingredient_df: Dict[str, Dict[str, float]], include_preference: bool = True, menu_plan_length: int = 10, weight_type: str = None, seed: Optional[int] = None, model_save_path: str = 'rl_model', load_model_from_file = False):
        super().__init__(menu_plan_length, weight_type, seed)
        self.model_save_path = model_save_path
        self.ingredient_df = ingredient_df
        self.load_model_from_file = load_model_from_file
        self.include_preference = include_preference
        
    def train_rl_model(self, negotiated_ingredients, unavailable_ingredients: Optional[Set[str]] = None) -> MaskablePPO:
        setup_params = load_yaml("scripts/hyperparams/setup_preference.yaml")
        algo_hyperparams_dir = "scripts/hyperparams/masked_ppo.yaml"
        vec_normalize_yaml = "scripts/hyperparams/vec_normalize.yaml"
        algo_hyperparams = load_yaml(algo_hyperparams_dir)
        vec_normalize_params = load_yaml(os.path.join(vec_normalize_yaml))
        hyperparams = {**algo_hyperparams, **vec_normalize_params, **setup_params}
        args = argparse.Namespace(**hyperparams)
        args.negotiated_ingredients = negotiated_ingredients
        args.unavailable_ingredients = unavailable_ingredients
        args.include_preference = self.include_preference
        
        args = set_default_prefixes(args)
        args.policy_kwargs = dict(
            net_arch=dict(
                pi=[args.net_arch_width] * args.net_arch_depth,
                vf=[args.net_arch_width] * args.net_arch_depth
            ),
            activation_fn=get_activation_fn(args.activation_fn),
            ortho_init=args.ortho_init
        )
        
        args.seed = self.seed
        if not isinstance(args.seed, (int, type(None))):
            raise ValueError("Seed must be an integer or None")

        
        args.device = select_device(args)
        set_seed(args.seed, args.device)
        print(f"Using device: {args.device}")
        
        args.ingredient_df = self.ingredient_df
        for directory in [args.reward_dir, args.log_dir, args.save_dir, args.best_dir, args.hyperparams_dir]:
            ensure_dir_exists(directory, args.verbose)
        reward_save_path = None
        if args.plot_reward_history:
            reward_dir, reward_prefix = get_unique_directory(args.reward_dir, f"{args.reward_prefix}_seed_{args.seed}_env", '.json')
            reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
        env = setup_environment(args, reward_save_path=reward_save_path, eval=False)
        tensorboard_log_dir = os.path.join(args.log_dir, f"{args.log_prefix}_seed_{args.seed}")
        if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.lower() != 'none':
            model = load_model(args, env, tensorboard_log_dir, args.seed)
            reset_num_timesteps = False
        else:
            model = create_model(args, env, tensorboard_log_dir, args.seed)
            reset_num_timesteps = True
        best_dir, best_prefix = get_unique_directory(args.best_dir, f"{args.best_prefix}_seed_{args.seed}", "")
        best_model_path = os.path.join(best_dir, best_prefix)
        new_logger = configure(tensorboard_log_dir, format_strings=["stdout", "tensorboard"])
        for handler in new_logger.output_formats:
            if isinstance(handler, HumanOutputFormat):
                handler.max_length = 100
        hyperparams = {
            'algo': args.algo,
            'common_hyperparams': {
                'verbose': args.verbose,
                'tensorboard_log': tensorboard_log_dir,
                'device': args.device,
                'seed': args.seed,
                'gamma': args.gamma,
                'initialization_strategy': args.initialization_strategy,
            },
            'vecnormalize_params': {
                'vecnorm_norm_obs': args.vecnorm_norm_obs,
                'vecnorm_norm_reward': args.vecnorm_norm_reward,
                'vecnorm_clip_obs': args.vecnorm_clip_obs,
                'vecnorm_clip_reward': args.vecnorm_clip_reward,
                'vecnorm_epsilon': args.vecnorm_epsilon,
                'vecnorm_norm_obs_keys': args.vecnorm_norm_obs_keys,
            }
        }
        save_hyperparams(hyperparams, args, args.seed)
        save_dir, save_prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{args.seed}", "")
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // args.num_envs, 1),
            save_path=save_dir, name_prefix=save_prefix,
            save_vecnormalize=True,
            save_replay_buffer=True,
            verbose=args.verbose
        )
        save_vecnormalize_best_callback = SaveVecNormalizeBestCallback(
            save_path=best_model_path
        )
        
        args.num_envs = 1
        eval_env = setup_environment(args, reward_save_path=reward_save_path)
        
        eval_callback = MaskableEvalCallback(
            eval_env=eval_env,
            best_model_save_path=best_model_path,
            callback_on_new_best=save_vecnormalize_best_callback,
            n_eval_episodes=5,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            deterministic=True,
            log_path=tensorboard_log_dir,
            render=False,
            verbose=args.verbose
        )
        
        
        info_logger_callback = InfoLoggerCallback()
        callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])
        
        if self.load_model_from_file:
            model = load_model(args, env, tensorboard_log_dir, args.seed)
        else:
            model.set_logger(new_logger)
            model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
            env.close()
            
        return model, eval_env

    def evaluate_model(self, model, env, num_episodes, deterministic):
        predictions = []
        for episode in range(num_episodes):
            obs = env.reset()
            done, state = False, None
            episode_predictions = []
            counter = 0
            
            while True:
                counter += 1
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, state=state, deterministic=deterministic, action_masks=action_masks)

                obs, _, done, info = env.step(action)
                
                info = info[0]
                
                if done:
                    episode_predictions.append((
                        info['nutrient_averages'],
                        info['ingredient_group_count'],
                        info['ingredient_environment_count'],
                        info['cost'],
                        info['co2e_g'],
                        info['group_portions'],
                        info['preference_score'],
                        info['current_meal_plan'],
                    ))
                    break
            predictions.append(episode_predictions)
        return predictions

    def average_dicts(self, dicts):
        keys = dicts[0].keys()
        return {key: np.mean([d[key] for d in dicts]) for key in keys}

    def plot_results(self, predictions, num_episodes, save_paths, week, day):
        flattened_predictions = [pred for episode in predictions for pred in episode]
        nutrient_averages, ingredient_group_counts, ingredient_environment_counts, costs, co2_g, group_portions, preference_score, current_meal_plan = zip(*flattened_predictions)
        avg_nutrient_averages = self.average_dicts(nutrient_averages)
        avg_ingredient_group_counts = self.average_dicts(ingredient_group_counts)
        avg_ingredient_environment_counts = self.average_dicts(ingredient_environment_counts)
        avg_preference_score = np.mean(preference_score)
        avg_cost = self.average_dicts(costs)
        avg_co2_g = self.average_dicts(co2_g)
        avg_group_portions = self.average_dicts(group_portions)
        avg_nutrient_averages['co2_g'] = avg_co2_g['co2_g']
        avg_misc_averages = {}
        avg_misc_averages['cost'] = avg_cost['cost']
        avg_misc_averages['preference_score'] = avg_preference_score
        targets = {
            'nutrients': {
                'calories': (530, 'range'), 'fat': (20.6, 'max'), 'saturates': (6.5, 'max'),
                'carbs': (70.6, 'min'), 'sugar': (15.5, 'max'), 'fibre': (4.2, 'min'),
                'protein': (7.5, 'min'), 'salt': (1, 'max'), 'co2_g': (800, 'max')
            },
            'ingredient_groups': {
                'fruit': (1, 'min'), 'veg': (1, 'min'), 'protein': (1, 'min'),
                'carbs': (1, 'min'), 'dairy': (1, 'min'),
                'bread': (1, 'min'), 'confectionary': (0, 'max')
            },
            'ingredient_environment': {
                'animal_welfare': (0.5, 'min'), 'rainforest': (0.5, 'min'), 'water': (0.5, 'min'),
                'co2_rating': (0.5, 'min')
            },
            'preference_and_cost': {
                'cost': (2, 'max'), 'preference_score': (0.5, 'min')
            },
            'group_portions': {
                'fruit': ((40, 130), 'range1'), 'veg': ((40, 80), 'range1'), 'protein': ((40, 90), 'range1'),
                'carbs': ((40, 150), 'range1'), 'dairy': ((50, 150), 'range1'),
                'bread': ((50, 70), 'range1'), 'confectionary': ((0, 0), 'range1')
            },
        }
        # Create a figure with a 3x2 grid of subplots
        fig, axs = plt.subplots(3, 2, figsize=(16, 8))
        axs = axs.flatten()  # Flatten the 2D array of axes to a 1D array for easy iteration
        font_size = 8
        
        def plot_bars(ax, data, title, targets=None, rotation=0):
            labels = list(data.keys())
            values = list(data.values())
            colors = [
                'red' if (
                    targets and (
                        (targets[label][1] == 'max' and value > targets[label][0]) or 
                        (targets[label][1] == 'min' and value < targets[label][0]) or 
                        (targets[label][1] == 'range' and (value <= targets[label][0] * 0.9 or value >= targets[label][0] * 1.1)) or
                        (targets[label][1] == 'range1' and (value < targets[label][0][0] or value > targets[label][0][1]))
                    )
                ) else 'green'
                for label, value in zip(labels, values)
            ]
            bars = ax.bar(labels, values, color=colors, width=0.5)
            ax.set_ylabel('Value')
            ax.set_title(f"{title}{num_episodes} Episodes")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([label.replace('_', ' ').capitalize() for label in labels], rotation=rotation, ha='center', fontsize=font_size)
            ax.set_ylim(0, max(values) * 1.3)
            
            for i, (bar, value, label) in enumerate(zip(bars, values, labels)):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                            xytext=(0, 3), textcoords="offset points", ha='center', 
                            va='bottom', color=colors[i], clip_on=True)

        plot_bars(axs[0], avg_nutrient_averages, 'Nutrient Average Over ', targets['nutrients'])
        plot_bars(axs[1], avg_ingredient_group_counts, 'Ingredient Group Count Average Over ', targets['ingredient_groups'], rotation=25)
        plot_bars(axs[2], avg_ingredient_environment_counts, 'Ingredient Environment Count Average Over ', targets['ingredient_environment'])
        plot_bars(axs[3], avg_misc_averages, 'Preference Score and Cost Averages Over ', targets['preference_and_cost'])
        plot_bars(axs[4], avg_group_portions, 'Group Portions Averages Over ', targets['group_portions'], rotation=25)
        
        num_plots = 1
        
        selected_ingredient = np.array(list(current_meal_plan[0].keys()))
        current_selection = np.array(list(current_meal_plan[0].values()))
        
        bars = axs[5].bar(selected_ingredient, current_selection, color='blue', width=0.5)
        axs[5].set_ylabel('Grams of Ingredient')
        axs[5].set_title(f'Selected Ingredients: Episode {0+1}')
        axs[5].set_xticks(np.arange(len(selected_ingredient)))
        axs[5].set_xticklabels(selected_ingredient, rotation=15, ha='center', fontsize=font_size)
        axs[5].set_ylim(0, max(current_selection) * 1.3)
        
        for bar, actual in zip(bars, current_selection):
            height = bar.get_height()
            axs[5].annotate(f'{actual:.2f} g', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', clip_on=True)
    
            
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=1.1, wspace=0.1)
        
        # Save the figure
        try:
            if save_paths['graphs']:
                    graph_path = os.path.join(save_paths['graphs'], 'menu_graphs')
                    os.makedirs(graph_path, exist_ok=True)
                    plt.savefig(os.path.join(graph_path, f'RL_results_week_{week}_day_{day}.png'))
        except:
            raise ValueError("Error saving the graph")
            
        # Save data to JSON
        data_to_save = {
            'meal_plan': current_meal_plan,
            'nutrient_averages': avg_nutrient_averages,
            'ingredient_group_counts': avg_ingredient_group_counts,
            'ingredient_environment_counts': avg_ingredient_environment_counts,
            'consumption_averages': avg_misc_averages,
            'cost': avg_cost,
            'co2_g': avg_co2_g,
            'group_portions': avg_group_portions
        }
    
        if save_paths and 'data' in save_paths:
            json_filename = os.path.join(save_paths['data'], 'RL_menu_results.json')

            # Load existing data if the file exists
            if os.path.exists(json_filename):
                with open(json_filename, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = []

            # Append new data
            existing_data.append(data_to_save)

            # Save updated data
            with open(json_filename, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, plot_flag: bool = False, final_meal_plan_filename: str = 'NA', save_paths: Dict = None, week=0, day=0) -> Dict[str, str]:
        if self.generated_count > self.menu_plan_length:
            print(f"\nGenerated {self.menu_plan_length} meal plans, resetting ingredients.")
            self.reset_ingredients()
        
        unavailable_ingredients = unavailable.union(self.selected_ingredients)
        # Train the RL modeln
        model, eval_env = self.train_rl_model(negotiated, unavailable_ingredients)
        
        eval_env.training = False
        eval_env.norm_reward = False

        # Evaluate the model
        num_episodes = 1
        predictions = self.evaluate_model(model, eval_env, num_episodes, deterministic=True)

        # Plot and save the results
        if plot_flag:
            self.plot_results(predictions, num_episodes, save_paths, week, day)

        # Generate the final meal plan
        final_meal_plan = self.get_final_meal_plan(predictions)
        
        for ingredient in final_meal_plan.keys():
            for group, ingredients in negotiated.items():
                if group in self.groups_to_remove_from:
                    if ingredient in ingredients:
                        self.selected_ingredients.add(ingredient)
        
        self.generated_count += 1
        
        # Empyty score for now
        score = {}
        
        return final_meal_plan, score
    
        
        
    def get_final_meal_plan(self, predictions):
        # Extract the final meal plan from predictions
        flattened_predictions = [pred for episode in predictions for pred in episode]
        _, _, _, _, _, _, _, current_meal_plan = zip(*flattened_predictions)
        
        # Get the last meal plan as the final meal plan
        final_meal_plan = current_meal_plan[-1]
        
        return final_meal_plan

    def reset_ingredients(self) -> None:
        self.selected_ingredients.clear()
        self.generated_count = 1

