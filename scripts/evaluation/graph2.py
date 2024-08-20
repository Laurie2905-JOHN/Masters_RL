import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from utils.process_data import get_data
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.evaluation import evaluate_policy
import json
from collections import defaultdict

# Configure Matplotlib to use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use serif font in conjunction with LaTeX
    "text.latex.preamble": r"\usepackage{times}",
})

def evaluate_model(model, env, deterministic=False):
    """Evaluate the trained model and collect predictions."""

    obs = env.reset()
    done, state = False, None
    episode_predictions = []
    
    counter = 0
    
    while not done:  # Continue until the episode is done
        counter += 1
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, state=state, deterministic=deterministic, action_masks=action_masks)
        obs, reward, done, info = env.step(action)
        
        info = info[0]

        if done:
            print(f"Episode ended at step {counter} with done={done}.")
            episode_predictions.append((
                info['nutrient_averages'],
                info['ingredient_group_count'],
                info['ingredient_environment_count'],
                info['cost'],
                info['co2e_g'],
                info['reward'],
                info['group_portions'],
                info['targets_not_met_count'],
                info['current_meal_plan'],
            ))
    
    return episode_predictions, list(info['current_meal_plan'].keys())

def average_dicts(dict_list):
    """Averages the values in a list of dictionaries."""
    averaged_dict = defaultdict(float)

    for d in dict_list:
        for key, value in d.items():
            averaged_dict[key] += value

    for key in averaged_dict:
        averaged_dict[key] /= len(dict_list)

    return dict(averaged_dict)

def plot_results(predictions, num_episodes):
    """Plot the results from the accumulated predictions."""
    # Flatten the predictions from all episodes
    flattened_predictions = [pred for episode in predictions for pred in episode]

    # Extract and average the various metrics
    nutrient_averages, ingredient_group_counts, ingredient_environment_counts, costs, co2_g, reward_dicts, group_portions, _, current_meal_plan = zip(*flattened_predictions)

    avg_nutrient_averages = average_dicts(nutrient_averages)
    avg_ingredient_group_counts = average_dicts(ingredient_group_counts)
    avg_ingredient_environment_counts = average_dicts(ingredient_environment_counts) | average_dicts(costs)
    avg_group_portions = average_dicts(group_portions)
    avg_grams_per_meal = {'grams_per_meal': sum(avg_group_portions.values())}

    for reward_dict in reward_dicts:
        reward_dict['targets_not_met'] = len(reward_dict['targets_not_met'])

    avg_co2g_grams_per_meal = average_dicts(co2_g)
    avg_co2g_grams_per_meal.update(avg_grams_per_meal)

    reward_dict_scores = average_dicts(reward_dicts)
    del reward_dict_scores['bonus_score']

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

    # Plot each of the metrics
    plot_bars(axs[1], avg_nutrient_averages, 'Nutrients', targets['nutrients'], rotation=25)
    plot_bars(axs[2], avg_ingredient_group_counts, 'Ingredient Group Count', targets['ingredient_groups'], rotation=25)
    plot_bars(axs[3], avg_group_portions, 'Group Portions', targets['group_portions'], rotation=25)
    plot_bars(axs[4], avg_ingredient_environment_counts, 'Ingredient Environment Count', targets['ingredient_environment'], rotation=25)
    plot_bars(axs[5], avg_co2g_grams_per_meal, 'CO2e_kg and Total Grams per Meal', targets['avg_co2g_grams_per_meal'], rotation=25)
    plot_bars(axs[6], reward_dict_scores, 'Reward Scores', targets['scores'], rotation=20)

    # Single plot for selected ingredients spanning the entire width
    for i in range(min(len(current_meal_plan), 1)):  # Plotting just the first meal plan
        selected_ingredient = np.array(list(current_meal_plan[i].keys()))
        current_selection = np.array(list(current_meal_plan[i].values()))
        
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
    plt.savefig('evaluation_report.png')


class Args:
    algo = "MASKED_PPO"
    render_mode = None
    num_envs = 1
    plot_reward_history = False
    max_episode_steps = 175
    verbose = 3
    action_update_factor = 5
    memory_monitor = False
    gamma = 0.99
    max_ingredients = 6
    reward_save_interval = 1000
    vecnorm_norm_obs = True
    vecnorm_norm_reward = True
    vecnorm_clip_obs = 10
    vecnorm_clip_reward = 250
    vecnorm_epsilon = 1e-8 
    ingredient_df = get_data("data.csv")
    seed = 22
    env_name = 'SchoolMealSelection-v2'
    initialization_strategy = 'zero'
    verbose = 0
    # vecnorm_norm_obs_keys = ['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
    vecnorm_norm_obs_keys = ['current_selection_value']
    reward_type = 'shaped'
    
def main():
    args = Args()

    from utils.train_utils import setup_environment, get_unique_directory
    from utils.process_data import get_data  # Ensure this import is correct
    
    basepath = os.path.abspath(f"saved_models/evaluation/report_models")

    filename = "report5_seed_2658707013"
        
    negotiated_file_path = os.path.join(basepath, filename, "negotiated_and_unavailable_ingredients.json")
    # Load data from JSON
    with open(negotiated_file_path, 'r') as json_file:
        loaded_data = json.load(json_file)
        
    meal_plan_num =  1
    
    all_predictions = []
    previous_meal_plan = []
    groups_to_remove_from = ['Group A veg', 'Group A fruit', 'Group BC', 'Group D', 'Group E']
    for _ in range(meal_plan_num):  # Assuming `meal_plan_num` is the number of runs

        # Restore the data to args
        args.negotiated_ingredients = loaded_data['negotiated_ingredients']
        args.unavailable_ingredients = loaded_data['unavailable_ingredients']
        selected_ingredients = []
        for ingredient in previous_meal_plan:
            for group, ingredients in args.negotiated_ingredients.items():
                if group in groups_to_remove_from:
                    if ingredient in ingredients:
                        selected_ingredients.append(ingredient)
        
        args.unavailable_ingredients.extend(selected_ingredients)

        env = setup_environment(args, eval=True)

        norm_path = os.path.join(basepath, filename, "vecnormalize_best.pkl")
        
        env = VecNormalize.load(norm_path, env)
        
        env.training = False
        env.norm_reward = False

        # Dummy learning rate schedule function
        def dummy_lr_schedule(_):
            return 1e-3

        model_path = os.path.join(basepath, filename, "best_model.zip")
        
        custom_objects = {'lr_schedule': dummy_lr_schedule}
        
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)  

        predictions, previous_meal_plan = evaluate_model(model, env, deterministic=True)

        all_predictions.append(predictions)  # Accumulate all predictions

    plot_results(all_predictions, len(all_predictions))


if __name__ == "__main__":
    main()        
