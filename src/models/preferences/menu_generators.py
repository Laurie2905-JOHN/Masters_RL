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
    create_model
)
from utils.process_data import get_data
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
        self.ingredient_df = ingredient_df

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
        args = set_default_prefixes(args)
        args.policy_kwargs = dict(
            net_arch=dict(
                pi=[args.net_arch_width] * args.net_arch_depth,
                vf=[args.net_arch_width] * args.net_arch_depth
            ),
            activation_fn=get_activation_fn(args.activation_fn),
            ortho_init=args.ortho_init
        )
        if args.seed is None:
            args.seed = generate_random_seeds(1)[0]
        if type(args.seed) is not int:
            raise ValueError("Seed must be an integer")
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
        
        eval_env = setup_environment(args, reward_save_path=reward_save_path),
        
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
        
        if load_model:
            model = load_model(args, env, tensorboard_log_dir, args.seed)
        else:
            model.set_logger(new_logger)
            model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
            env.close()
            
        return model, eval_env
    
    def evaluate_model(self, model, env, num_episodes=10, deterministic=False):
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
                        info['consumption_average'],
                        info['cost'],
                        info['co2_g'],
                        info['reward'],
                        info['group_portions'],
                        info['targets_not_met_count'],
                        info['current_meal_plan'],
                    ))
                    break
            predictions.append(episode_predictions)
        return predictions

    def average_dicts(self, dicts):
        keys = dicts[0].keys()
        return {key: np.mean([d[key] for d in dicts]) for key in keys}

    def plot_results(self, predictions, num_episodes, save_path):
        flattened_predictions = [pred for episode in predictions for pred in episode]
        nutrient_averages, ingredient_group_counts, ingredient_environment_counts, consumption_averages, costs, co2_g, _, group_portions, _, current_meal_plan = zip(*flattened_predictions)
        avg_nutrient_averages = self.average_dicts(nutrient_averages)
        avg_ingredient_group_counts = self.average_dicts(ingredient_group_counts)
        avg_ingredient_environment_counts = self.average_dicts(ingredient_environment_counts)
        avg_consumption_averages = self.average_dicts(consumption_averages)
        avg_cost = self.average_dicts(costs)
        avg_co2_g = self.average_dicts(co2_g)
        avg_group_portions = self.average_dicts(group_portions)
        avg_consumption_averages['cost'] = avg_cost['cost']
        avg_consumption_averages['co2_g'] = avg_co2_g['co2_g']
        targets = {
            'nutrients': {
                'calories': (530, 'range'), 'fat': (20.6, 'max'), 'saturates': (6.5, 'max'),
                'carbs': (70.6, 'min'), 'sugar': (15.5, 'max'), 'fibre': (4.2, 'min'),
                'protein': (7.5, 'min'), 'salt': (1, 'max')
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
            'consumption_cost_co2_g': {
                'average_mean_consumption': (5.8, 'min'), 'average_cv_ingredients': (8.5, 'min'),
                'cost': (2, 'max'), 'co2_g': (800, 'max'),
            },
            'group_portions': {
                'fruit': ((40, 130), 'range1'), 'veg': ((40, 80), 'range1'), 'protein': ((40, 90), 'range1'),
                'carbs': ((40, 150), 'range1'), 'dairy': ((50, 150), 'range1'),
                'bread': ((50, 70), 'range1'), 'confectionary': ((0, 0), 'range1')
            },
        }
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(4, 2)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
               fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
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
        plot_bars(axs[3], avg_consumption_averages, 'Consumption and Cost Averages Over ', targets['consumption_cost_co2_g'])
        plot_bars(axs[4], avg_group_portions, 'Group Portions Averages Over ', targets['group_portions'], rotation=25)
        
        num_plots = min(len(current_meal_plan), 3)
        
        for i in range(num_plots):
            selected_ingredient = np.array(list(current_meal_plan[i].keys()))
            current_selection = np.array(list(current_meal_plan[i].values()))
            
            bars = axs[5 + i].bar(selected_ingredient, current_selection, color='blue', width=0.5)
            axs[5 + i].set_ylabel('Grams of Ingredient')
            axs[5 + i].set_title(f'Selected Ingredients: Episode {i+1}')
            axs[5 + i].set_xticks(np.arange(len(selected_ingredient)))
            axs[5 + i].set_xticklabels(selected_ingredient, rotation=15, ha='center', fontsize=font_size)
            axs[5 + i].set_ylim(0, max(current_selection) * 1.3)
            
            for bar, actual in zip(bars, current_selection):
                height = bar.get_height()
                axs[5 + i].annotate(f'{actual:.2f} g', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', clip_on=True)
        
        for j in range(num_plots + 5, len(axs)):
            fig.delaxes(axs[j])
            
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=1.1, wspace=0.1)
        
        # Save the figure
        if save_path:
            plt.savefig(os.path.join(save_path, 'results.png'))
        plt.show()

    def generate_menu(self, negotiated: Dict[str, Dict[str, float]], unavailable: Optional[Set[str]] = None, save_path: str = None) -> Dict[str, str]:
        if self.generated_count > self.menu_plan_length:
            print(f"\nGenerated {self.menu_plan_length} meal plans, resetting ingredients.")
            self.reset_ingredients()

        unavailable_ingredients = unavailable.union(self.selected_ingredients)
        # Train the RL model
        model, eval_env = self.train_rl_model(negotiated, unavailable_ingredients)
        
        eval_env.training = False
        eval_env.norm_reward = False

        # Evaluate the model
        num_episodes = 1
        predictions = self.evaluate_model(model, eval_env, "MASKED_PPO", num_episodes, deterministic=True)

        # Plot and save the results
        self.plot_results(predictions, num_episodes, save_path)

        # Generate the final meal plan
        final_meal_plan = self.get_final_meal_plan(predictions)
        
        # Save the meal plan
        if save_path:
            with open(os.path.join(save_path, 'final_meal_plan.txt'), 'w') as f:
                for item, amount in final_meal_plan.items():
                    f.write(f"{item}: {amount}\n")
        
        self.selected_ingredients.update(final_meal_plan.keys())
            
        return final_meal_plan

    def get_final_meal_plan(self, predictions):
        # Extract the final meal plan from predictions
        flattened_predictions = [pred for episode in predictions for pred in episode]
        _, _, _, _, _, _, _, _, _, current_meal_plan = zip(*flattened_predictions)
        
        # Get the last meal plan as the final meal plan
        final_meal_plan = current_meal_plan[-1]
        
        return final_meal_plan

    def reset_ingredients(self) -> None:
        self.selected_ingredients.clear()
        self.ingredients_in_groups = {group: items.copy() for group, items in self.original_ingredients_in_groups.items()}
        self.ingredients_scores = {group: scores.copy() for group, scores in self.original_ingredients_scores.items()}
        self.total_scores = self.original_total_scores.copy()
        self.generated_count = 1

