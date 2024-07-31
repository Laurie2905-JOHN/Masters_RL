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
from utils.train_utils import *
from utils.process_data import get_data

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
    def normalize_scores(self) -> None:
        pass
    
    @abstractmethod
    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None) -> Dict[str, str]:
        pass


class RandomMenuGenerator(BaseMenuGenerator):
    """
    Inherits from BaseMenuGenerator and uses random selection logic.
    """
    def __init__(self, menu_plan_length: int = 10, weight_type: str = None, probability_best: float = 0.5, seed: Optional[int] = None):
        super().__init__(menu_plan_length, weight_type, probability_best, seed)

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
        
        :param negotiated: A dictionary where keys are ingredient groups and values are dictionaries of ingredients with their scores.
        :param unavailable: A set of unavailable ingredients.
        """
        self.negotiated = negotiated
        self.unavailable = unavailable or set()
        
        self.ingredient_groups = list(negotiated.keys())

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
        
        :param group: The ingredient group to sample from.
        :return: The best ingredient.
        """
        if len(self.ingredients_in_groups[group]) == 0:
            raise ValueError(f"No ingredients available in group {group}")
        max_score_index = self.ingredients_scores[group].index(max(self.ingredients_scores[group]))
        return self.ingredients_in_groups[group][max_score_index]

    def generate_random_item(self, group: str, num_items: int = 1) -> List[str]:
        """
        Generates a list of random items from a specified group based on the normalized scores.
        
        :param group: The ingredient group to sample from.
        :param num_items: The number of items to sample.
        :return: A list of randomly selected ingredients.
        :raises ValueError: If there are not enough ingredients in the group to sample the requested number of items.
        """
        
        if len(self.ingredients_in_groups[group]) < num_items:
            raise ValueError(f"Not enough ingredients in group {group} to sample {num_items} items")
        items = self.ingredients_in_groups[group]
        if self.weight_type == "score":
            weights = self.normalized_scores[group]
        elif self.weight_type == "random":
            weights = None
            
        return self.random.choices(items, weights=weights, k=num_items)
    
    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None) -> Dict[str, str]:
        """
        Generates a menu by selecting an item from each ingredient group with a certain probability of choosing
        the best item and a complementary probability of choosing a random item. If no ingredients are available,
        it resets the ingredients.
        
        :param negotiated: A dictionary where keys are ingredient groups and values are dictionaries of ingredients with their scores.
        :param unavailable: A set of unavailable ingredients.
        :return: A dictionary representing the generated menu.
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
            except ValueError(f"Error generating item for group {group}"):
                raise 
        
        # Convert the menu dictionary to a tuple to track frequency
        menu_tuple = tuple(sorted(menu.items()))
        self.menu_counter[menu_tuple] += 1
        
        print("\nGenerated meal plan number", self.generated_count)
        print(list(menu.values()))
        self.generated_count += 1
        return menu

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
    def __init__(self, ingredient_df: Dict[str, Dict[str, float]], menu_plan_length: int = 10, weight_type: str = None, seed: Optional[int] = None, model_save_path: str = 'rl_model'):
        super().__init__(menu_plan_length, weight_type, seed)
        self.model_save_path = model_save_path
        self.env = None
        self.model = None
        self.ingredient_df = ingredient_df

    def normalize_scores(self) -> None:
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

    @staticmethod
    def create_preference_score_function(negotiated: Dict[str, Dict[str, float]], unavailable: List[str]) -> Callable[[str], float]:
        """
        Create a preference score function based on negotiated ingredients, excluding unavailable ingredients.

        :param negotiated: Dictionary of negotiated ingredients and their scores.
        :param unavailable: List of unavailable ingredients.
        :return: Function that returns the score for a given ingredient.
        """
        ingredient_scores = {}
        if negotiated:
            for group, ingredients in negotiated.items():
                # Filter out unavailable ingredients
                available_ingredients = {k: v for k, v in ingredients.items() if k not in unavailable}
                
                if available_ingredients:
                    scores = np.array(list(available_ingredients.values())).reshape(-1, 1)
                    scaler = MinMaxScaler()
                    normalized_scores = scaler.fit_transform(scores).flatten()
                    for ingredient, norm_score in zip(available_ingredients.keys(), normalized_scores):
                        ingredient_scores[ingredient] = norm_score

        def score_function(ingredient: str) -> float:
            return ingredient_scores.get(ingredient, 0)

        return score_function
    
    def train_rl_model(self, preference_score_function: Callable[[str], float], unavailable_ingredients: Optional[Set[str]] = None) -> MaskablePPO:
        """
        Train a reinforcement learning model using the given preference score function and unavailable ingredients.

        :param preference_score_function: Function to calculate preference scores for ingredients.
        :param unavailable_ingredients: Set of unavailable ingredients.
        :return: Trained RL model.
        """
        # Load setup file
        setup_params = load_yaml("scripts/hyperparams/setup_preference.yaml")
        
        # Select the MASKED PPO yaml
        algo_hyperparams_dir = "scripts/hyperparams/masked_ppo.yaml"
        vec_normalize_yaml = "scripts/hyperparams/vec_normalize.yaml"
        
        # Load hyperparameters from the selected YAML file
        algo_hyperparams = load_yaml(algo_hyperparams_dir)

        # Load additional parameters
        vec_normalize_params = load_yaml(os.path.join(vec_normalize_yaml))
        
        # Combine all parameters
        hyperparams = {**algo_hyperparams, **vec_normalize_params, **setup_params}

        # Convert dictionary to an argparse.Namespace object
        args = argparse.Namespace(**hyperparams)
                
        # Set default prefixes if not provided
        args = set_default_prefixes(args)

        # Define policy kwargs
        args.policy_kwargs = dict(
            net_arch=dict(
                pi=[args.net_arch_width] * args.net_arch_depth,
                vf=[args.net_arch_width] * args.net_arch_depth
            ),
            activation_fn=get_activation_fn(args.activation_fn),
            ortho_init=args.ortho_init
        )
        
        if args.seed is None:
            args.seed = generate_random_seeds(1)
            
        # Generate random seeds if not provided
        if type(args.seed) is not int:
            raise ValueError("Seed must be an integer")

        args.device = select_device(args)
        set_seed(args.seed, args.device)

        print(f"Using device: {args.device}")

        try:
            args.ingredient_df = self.ingredient_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return

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

        eval_callback = MaskableEvalCallback(
            eval_env=setup_environment(args, reward_save_path=reward_save_path),
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

        model.set_logger(new_logger)
        
        model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
        
        env.close()

        if args.plot_reward_history:
            reward_prefix = reward_prefix.split(".")[0]
            dir, pref = get_unique_directory(args.reward_dir, f"{reward_prefix}_plot", '.png')
            plot_path = os.path.abspath(os.path.join(dir, pref))
            plot_reward_distribution(reward_save_path, plot_path)

        return model

    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None) -> Dict[str, str]:
        
        if self.generated_count > self.menu_plan_length:
            print(f"\nGenerated {self.menu_plan_length} meal plans, resetting ingredients.")
            self.reset_ingredients()

        menu = {}
        
        preference_score_function = self.create_preference_score_function(negotiated, unavailable)
        
        menu_generator = self.train_rl_model(preference_score_function, unavailable)
        
        menu_generator.predict()
        
        menu_tuple = tuple(sorted(menu.items()))
        self.menu_counter[menu_tuple] += 1
        
        for item in menu.keys():
            if item in self.groups_to_remove_from:
                self.selected_ingredients.add(item)
                        
        print("\nGenerated meal plan number", self.generated_count)
        print(list(menu.values()))
        self.generated_count += 1
        return menu

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

# Example usage
# rm = RandomMenuGenerator(seed=42)
# rl = RLMenuGenerator(seed=42)
