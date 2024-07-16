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

def evaluate_model(model, env, algo, num_episodes=10, deterministic=True):
    """Evaluate the trained model and collect predictions."""
    predictions = []

    for episode in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_predictions = []
        counter = 0
        print("Algo:", algo)
        while True:
            counter += 1
            if algo == "MASKED_PPO":
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, state=state, deterministic=deterministic, action_masks = action_masks)
            else:
                action, _states = model.predict(obs, state=state, deterministic=deterministic)
                
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
                    info['co2_g'],
                    info['reward'],
                    info['group_portions'],
                    info['targets_not_met_count'],
                    info['current_meal_plan']
                ))
                break
        predictions.append(episode_predictions)
    
    return predictions

def average_dicts(dicts):
    keys = dicts[0].keys()
    return {key: np.mean([d[key] for d in dicts]) for key in keys}

def plot_results(predictions, num_episodes):
    """Plot the results from the predictions."""
    flattened_predictions = [pred for episode in predictions for pred in episode]

    nutrient_averages, ingredient_group_counts, ingredient_environment_counts, consumption_averages, costs, co2_g, _, _, _, current_meal_plan = zip(*flattened_predictions)
    avg_nutrient_averages = average_dicts(nutrient_averages)
    avg_ingredient_group_counts = average_dicts(ingredient_group_counts)
    avg_ingredient_environment_counts = average_dicts(ingredient_environment_counts)
    avg_consumption_averages = average_dicts(consumption_averages)
    avg_cost = average_dicts(costs)
    avg_co2_g = average_dicts(co2_g)
    
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
                    (targets[label][1] == 'range' and (value <= targets[label][0] * 0.9 or value >= targets[label][0] * 1.1))
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
    num_plots = min(len(current_meal_plan), 4)
    
    for i in range(num_plots):
        selected_ingredient = np.array(list(current_meal_plan[i].keys()))
        current_selection = np.array(list(current_meal_plan[i].values()))
        
        
        bars = axs[4 + i].bar(selected_ingredient, current_selection, color='blue', width=0.5)
        axs[4 + i].set_ylabel('Grams of Ingredient')
        axs[4 + i].set_title(f'Selected Ingredients: Episode {i+1}')
        axs[4 + i].set_xticks(np.arange(len(selected_ingredient)))
        axs[4 + i].set_xticklabels(selected_ingredient, rotation=15, ha='center', fontsize=font_size)
        
        axs[4 + i].set_ylim(0, max(current_selection) * 1.3)
        
        for bar, actual in zip(bars, current_selection):
            height = bar.get_height()
            axs[4 + i].annotate(f'{actual:.2f} g', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', clip_on=True)
    
    for j in range(num_plots + 4, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=1.1, wspace=0.1)
    plt.show()


class Args:
    algo = "MASKED_PPO"
    render_mode = None
    num_envs = 1
    plot_reward_history = False
    max_episode_steps = 1000
    verbose = 3
    action_scaling_factor = 10
    memory_monitor = True
    gamma = 0.99
    max_ingredients = 6
    action_scaling_factor = 10
    reward_save_interval = 1000
    vecnorm_norm_obs = True
    vecnorm_norm_reward = True
    vecnorm_clip_obs = 10
    vecnorm_clip_reward = 10
    vecnorm_epsilon = 1e-8 
    vecnorm_norm_obs_keys = None
    ingredient_df = get_data("data.csv")
    seed = 2789679713
    env_name = 'SchoolMealSelection-v2'
    initialization_strategy = 'zero'
    vecnorm_norm_obs_keys = ['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
    reward_type = 'shaped'
    
def main():
    args = Args()

    from utils.train_utils import setup_environment, get_unique_directory
    from utils.process_data import get_data  # Ensure this import is correct
    
    basepath = os.path.abspath(f"saved_models/evaluation/mum/evaluate/V2")

    filename = "SchoolMealSelection_v2_MASKED_PPO_reward_type_shaped_500000_4env_env_best_seed_2789679713"    

    # if len(filename.split("_")) < 2:
    #     seed = random.randint(0, 1000)
    # else: 
    #     seed = int(filename.split("_")[-1])

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
    
    if args.algo == 'A2C':
        model = A2C.load(model_path, env=env, custom_objects=custom_objects)
    elif args.algo == 'PPO':
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    elif args.algo == "MASKED_PPO":
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)  

    num_episodes = 4

    predictions = evaluate_model(model, env, args.algo, num_episodes, deterministic=True)

    plot_results(predictions, num_episodes)

if __name__ == "__main__":
    main()
