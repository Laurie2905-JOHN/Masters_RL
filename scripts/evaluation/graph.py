import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from utils.process_data import get_data

def evaluate_model(model, env, num_episodes=10, deterministic=True):
    """Evaluate the trained model and collect predictions."""
    predictions = []

    for episode in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_predictions = []
        counter = 0

        while not done:
            counter += 1
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            info = info[0]

            if done:
                print(f"Episode {episode + 1} ended at step {counter} with done={done}.")
                episode_predictions.append((
                    info['nutrient_averages'],
                    info['ingredient_group_count'],
                    info['ingredient_environment_count'],
                    info['consumption_average'],
                    info['cost'],
                    info['reward'],
                    info['current_selection']
                ))
                break

        predictions.append(episode_predictions)
    
    return predictions

def average_dicts(dicts):
    keys = dicts[0].keys()
    return {key: np.mean([d[key] for d in dicts]) for key in keys}

def get_meat_color(ingredient_group_counts):
    meat_counter = sum(ingredient_group_counts.get(key, 0) for key in ['non_processed_meat', 'processed_meat'])
    return 'green' if meat_counter >= 1 else 'red'

def plot_results(predictions, ingredient_df, num_episodes):
    """Plot the results from the predictions."""
    flattened_predictions = [pred for episode in predictions for pred in episode]

    nutrient_averages, ingredient_group_counts, ingredient_environment_counts, consumption_averages, costs, rewards, current_selections = zip(*flattened_predictions)

    avg_nutrient_averages = average_dicts(nutrient_averages)
    avg_ingredient_group_counts = average_dicts(ingredient_group_counts)
    avg_ingredient_environment_counts = average_dicts(ingredient_environment_counts)
    avg_consumption_averages = average_dicts(consumption_averages)
    avg_cost = np.mean(costs)
    avg_consumption_averages['avg_cost'] = avg_cost

    targets = {
        'nutrients': {
            'calories': (530, 'range'), 'fat': (20.6, 'range'), 'saturates': (6.5, 'max'),
            'carbs': (70.6, 'range'), 'sugar': (15.5, 'max'), 'fibre': (4.2, 'min'),
            'protein': (7.5, 'min'), 'salt': (0.499, 'max')
        },
        'ingredient_groups': {
            'fruit': (1, 'min'), 'veg': (1, 'min'), 'non_processed_meat': (1, 'min'),
            'processed_meat': (1, 'min'), 'carbs': (1, 'min'), 'dairy': (1, 'min'),
            'bread': (1, 'min'), 'confectionary': (0, 'max')
        },
        'ingredient_environment': {
            'animal_welfare': (1, 'min'), 'rainforest': (1, 'min'), 'water': (1, 'min'),
            'CO2_rating': (1, 'min'), 'CO2_g': (500, 'max')
        },
        'consumption': {
            'average_mean_consumption': (0, 'min'), 'average_cv_ingredients': (0, 'min'),
            'avg_cost': (2, 'max')
        }
    }

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(4, 2)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
           fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    
    font_size = 8

    def plot_bars(ax, data, title, targets=None, rotation=0):
        labels = list(data.keys())
        values = list(data.values())
        
        meat_color = get_meat_color(avg_ingredient_group_counts)
        
        colors = [
            meat_color if label in ['non_processed_meat', 'processed_meat'] 
            else ('red' if (targets and ((targets[label][1] == 'max' and value > targets[label][0]) or 
                                        (targets[label][1] == 'min' and value < targets[label][0]) or 
                                        (targets[label][1] == 'range' and (value <= targets[label][0] * 0.9 or value >= targets[label][0] * 1.1))))
                else 'green')
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
    plot_bars(axs[3], avg_consumption_averages, 'Consumption and Cost Averages Over ', targets['consumption'])
    num_plots = min(len(current_selections), 4)
    
    for i in range(num_plots):
        current_selection = np.array(current_selections[i])
        non_zero_indices = np.nonzero(current_selection)
        non_zero_values = current_selection[non_zero_indices]
        selected_ingredients = ingredient_df['Category7'].iloc[non_zero_indices]

        bars = axs[4 + i].bar(selected_ingredients.values, non_zero_values, color='blue', width=0.5)
        axs[4 + i].set_ylabel('Grams of Ingredient')
        axs[4 + i].set_title(f'Selected Ingredients: Episode {i+1}')
        axs[4 + i].set_xticks(np.arange(len(selected_ingredients)))
        axs[4 + i].set_xticklabels(selected_ingredients.values, rotation=15, ha='center', fontsize=font_size)
        
        axs[4 + i].set_ylim(0, max(non_zero_values) * 1.3)
        
        for bar, actual in zip(bars, non_zero_values):
            height = bar.get_height()
            axs[4 + i].annotate(f'{actual:.2f} g', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', clip_on=True)
    
    for j in range(num_plots + 4, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=1.1, wspace=0.1)
    plt.show()

class Args:
    reward_metrics = ['nutrients', 'groups', 'environment', 'cost', 'consumption']
    render_mode = None
    num_envs = 1
    plot_reward_history = False
    max_episode_steps = 1000

def main():
    args = Args()

    from utils.train_utils import setup_environment, get_unique_directory
    from utils.process_data import get_data  # Ensure this import is correct
    
    ingredient_df = get_data()

    seed = 7859985245

    env = setup_environment(args, seed, ingredient_df, eval=True)
    
    filename = "SchoolMealSelection_v1_A2C_1000000_16env_best_nutrients_groups_environment_cost_consumption_seed_3379026565"
    filename = "SchoolMealSelection-v1_A2C_2000000_1env_best_nutrients_groups_environment_seed2754945570"
    
    
    
    norm_path = os.path.abspath(f"saved_models/evaluation/best_models/{filename}/vec_normalize_best.pkl")
    env = VecNormalize.load(norm_path, env)
    
    env.training = False
    env.norm_reward = False

    # Dummy learning rate schedule function
    def dummy_lr_schedule(_):
        return 1e-3

    model_path = os.path.abspath(f"saved_models/evaluation/best_models/{filename}/best_model.zip")
    custom_objects = {'lr_schedule': dummy_lr_schedule}
    model = A2C.load(model_path, env=env, custom_objects=custom_objects)

    num_episodes = 10

    predictions = evaluate_model(model, env, num_episodes, deterministic=True)

    plot_results(predictions, ingredient_df, num_episodes)

if __name__ == "__main__":
    main()
