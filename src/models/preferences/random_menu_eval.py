import numpy as np
from collections import Counter, deque
from models.envs.env import BaseEnvironment
from models.envs.score_calculator import ScoreCalculatorShaped
from typing import Dict
from models.preferences.preference_utils import create_preference_score_function
import matplotlib.pyplot as plt

class MenuEvaluator(BaseEnvironment):
    def __init__(self, ingredient_df, negotiated_ingredients, unavailable_ingredients):
        super().__init__(ingredient_df, negotiated_ingredients = negotiated_ingredients, unavailable_ingredients = unavailable_ingredients)  # Initialize BaseEnvironment attributes and methods
        self.ingredient_df = ingredient_df
        self.current_selection = None
        self.reward_dict = self.initialize_rewards()
        self.scores = None
        self.total_quantity_ratio = None
        self.preference_score_function = create_preference_score_function(negotiated_ingredients, unavailable_ingredients)
        self.preference_target = 0.7
        self.score_weights = [2, 1, 1, 1, 1]
        
    def select_ingredients(self, selected_ingredients: Dict[str, float]) -> None:
        """
        Update current selection based on the provided ingredients and their quantities.
        
        :param selected_ingredients: A dictionary where keys are ingredient names and values are quantities.
        """
        self.current_selection = np.zeros(len(self.ingredient_df), dtype=np.float32)  # Reset selection
        
        for ingredient, quantity in selected_ingredients.items():
            try:
                index = self.ingredient_df[self.ingredient_df['Category7'] == ingredient].index[0]
                self.current_selection[index] = quantity
            except IndexError:
                raise ValueError(f"Ingredient '{ingredient}' not found in ingredient_df.")

        self.total_quantity_ratio = self.current_selection / (sum(self.current_selection) + 1e-9)
        
        self.get_metrics()  # Update metrics based on the new selection
        
        self.initialize_rewards()  # Reset reward dictionary
        
        reward = self.calculate_reward()
        
        info = self._get_info()
        
        # Create the info dictionary with various components
        info = {
            'nutrient_averages': info['nutrient_averages'],
            'ingredient_group_count': info['ingredient_group_count'],
            'ingredient_environment_count': info['ingredient_environment_count'],
            'cost': info['cost'],
            'co2e_g': info['co2e_g'],
            'reward': self.reward_dict,
            'group_portions':   info['group_portions'],
            'current_meal_plan': info['current_meal_plan']
        }
        
        if 'preference_score' in self.reward_dict.keys():
            info['preference_score'] = self.reward_dict['preference_score']
    
        return reward, info

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the current ingredient selection.
        
        :return: The calculated reward.
        """
        # Instantiate ScoreCalculator with the main class instance
        score_calculator = ScoreCalculatorShaped(self)
        
        # Calculate the scores and targets not met
        self.scores, targets_not_met = score_calculator.calculate_scores()
        
        # Store the individual scores in the reward dictionary
        self.reward_dict['nutrient_score'] = self.scores[0]
        self.reward_dict['cost_score'] = self.scores[1]
        self.reward_dict['co2_score'] = self.scores[2]
        self.reward_dict['environment_score'] = self.scores[3]
        self.reward_dict['preference_score'] = self.scores[4]
        self.reward_dict['targets_not_met'] = targets_not_met
        
        # Calculate the immediate reward based on the current scores
        current_reward = np.dot(self.scores, self.score_weights)
        
        return current_reward / sum(self.score_weights)  # Normalize by the sum of weights

    def objective_function(self, menu: Dict[str, float]) -> float:
        """
        Objective function that returns the calculated reward, used for optimization.
        
        :param menu: A dictionary of ingredients and their respective quantities.
        :return: The reward value calculated from the current selection of ingredients.
        """
        reward, _ = self.select_ingredients(menu)
        return reward
    
    def plot_menu(self, info: dict):
        
        # Configure Matplotlib to use LaTeX for rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",  # Use serif font in conjunction with LaTeX
            "text.latex.preamble": r"\usepackage{times}",
        })

        """Plot the menu and related metrics based on the provided info dictionary."""

        # Unpack the info dictionary
        nutrient_averages = info['nutrient_averages']
        ingredient_group_counts = info['ingredient_group_count']
        cost = info['cost']
        ingredient_environment_counts = info['ingredient_environment_count'] | cost
        co2_g = info['co2e_g']
        reward_dict = info['reward']
        group_portions = info['group_portions']
        current_meal_plan = info['current_meal_plan']

        # Aggregate data if necessary
        avg_nutrient_averages = nutrient_averages  # Assuming it's already averaged
        avg_ingredient_group_counts = ingredient_group_counts  # Assuming it's already averaged
        avg_ingredient_environment_counts = ingredient_environment_counts  # Assuming it's already averaged
        avg_group_portions = group_portions  # Assuming it's already averaged
        avg_co2g_grams_per_meal = co2_g | {'grams_per_meal': sum(list(avg_group_portions.values()))} # Assuming it's already averaged

        reward_dict_scores = reward_dict  # Assuming it's already aggregated
        reward_dict_scores['targets_not_met'] = len(reward_dict_scores['targets_not_met'])
        # Create a grid layout with subplots
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(4, 2)  # 4 rows, 2 columns layout
        # First, create the plot that spans the entire top row
        axs = [fig.add_subplot(gs[0, :])]  # Top row spans both columns

        # Then, create the other subplots in the remaining space
        axs.extend([fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), 
                    fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                    fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])])

        font_size = 11

        targets = {
            'nutrients': {
                'calories': (530, 'range'), 'fat': (20.6, 'max'), 'saturates': (6.5, 'max'),
                'carbs': (70.6, 'min'), 'sugar': (15.5, 'max'), 'fibre': (4.2, 'min'),
                'protein': (7.5, 'min'), 'salt': (1.5, 'max')
            },
            'ingredient_groups': {
                'fruit': (1, 'min'), 'veg': (1, 'min'), 'protein': (1, 'min'),
                'carbs': (1, 'min'), 'dairy': (1, 'min'),
                'bread': (1, 'min'), 'confectionary': (0, 'max')
            },
            'ingredient_environment': {
                'animal_welfare': (0.5, 'min'), 'rainforest': (0.5, 'min'), 'water': (0.5, 'min'),
                'CO2_rating': (0.5, 'min'), 'cost': (1.5, 'max'),
            },
            'avg_co2g_grams_per_meal': { 
                'co2e_g': (500, 'max'), 'grams_per_meal': (800, 'max'),
            },
            'group_portions': {
                'fruit': ((40, 130), 'range1'), 'veg': ((40, 80), 'range1'), 'protein': ((40, 90), 'range1'),
                'carbs': ((40, 150), 'range1'), 'dairy': ((50, 150), 'range1'),
                'bread': ((50, 70), 'range1'), 'confectionary': ((0, 0), 'range1')
            },
            'scores': {
                'nutrient_score': ((0, 1), 'range1'), 'cost_score': ((0, 1), 'range1'), 'co2_score': ((0, 1), 'range1'),
                'environment_score': (0.7, 'min'), 'preference_score': (0.7, 'min'), 'targets_not_met': (0, 'max')
            },
        }

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
            ax.set_title(f"{title}", fontsize = 14)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([label.replace('_', ' ').title() for label in labels], rotation=rotation, ha='center', fontsize=12)
            ax.set_ylim(0, max(values) * 1.3)
            for i, (bar, value, label) in enumerate(zip(bars, values, labels)):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), 
                            xytext=(0, 3), textcoords="offset points", ha='center', 
                            va='bottom', color=colors[i], clip_on=True, fontsize= 12)
                
        # Create a list of keys to delete
        keys_to_delete = [key for key in reward_dict_scores.keys() if key not in targets['scores'].keys()]

        # Now, delete those keys outside the iteration loop
        for key in keys_to_delete:
            del reward_dict_scores[key]

        # Plot each of the metrics
        plot_bars(axs[1], avg_nutrient_averages, 'Nutrients', targets['nutrients'], rotation=25)
        plot_bars(axs[2], avg_ingredient_group_counts, 'Ingredient Group Count', targets['ingredient_groups'], rotation=25)
        plot_bars(axs[3], avg_group_portions, 'Group Portions', targets['group_portions'], rotation=25)
        plot_bars(axs[4], avg_ingredient_environment_counts, 'Ingredient Environment Count', targets['ingredient_environment'], rotation=25)
        plot_bars(axs[5], avg_co2g_grams_per_meal, 'CO2e_kg and Total Grams per Meal', targets['avg_co2g_grams_per_meal'], rotation=25)
        plot_bars(axs[6], reward_dict_scores, 'Reward Scores', targets['scores'], rotation=20)

        # Single plot for selected ingredients spanning the entire width
        selected_ingredient = np.array(list(current_meal_plan.keys()))
        current_selection = np.array(list(current_meal_plan.values()))
        
        bars = axs[0].bar(selected_ingredient, current_selection, color='blue', width=0.5)
        axs[0].set_ylabel('Quantity (g)')
        axs[0].set_title(f'Meal Plan', fontsize = 14)
        axs[0].set_xticks(np.arange(len(selected_ingredient)))
        axs[0].set_xticklabels([label.replace('_', ' ').title() for label in selected_ingredient], rotation=15, ha='center', fontsize=font_size)
        
        axs[0].set_ylim(0, max(current_selection) * 1.3)
        
        for bar, actual in zip(bars, current_selection):
            height = bar.get_height()
            axs[0].annotate(f'{actual:.2f} g', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', clip_on=True)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=1.1, wspace=0.1)
        plt.savefig('menu_evaluation_report.png')
        plt.show()
