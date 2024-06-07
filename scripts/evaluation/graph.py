import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from utils.process_data import get_data

def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model and collect predictions."""
    predictions = []

    for episode in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_predictions = []

        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            info = info[0]

            if done:    
                episode_predictions.append((
                    obs, action, reward,
                    info['nutrient_averages']['calories'],
                    info['nutrient_averages']['fat'],
                    info['nutrient_averages']['saturates'],
                    info['nutrient_averages']['carbs'],
                    info['nutrient_averages']['sugar'],
                    info['nutrient_averages']['fibre'],
                    info['nutrient_averages']['protein'],
                    info['nutrient_averages']['salt'],
                    info['current_selection']
                ))
        
        predictions.append(episode_predictions)
    
    return predictions

def plot_results(predictions, ingredient_df, env):
    """Plot the results from the predictions."""
    observations, _, rewards, average_calories, average_fat, average_saturates, average_carbs, average_sugar, average_fibre, average_protein, average_salt, current_selection = zip(*[pred for episode in predictions for pred in episode])

    # Ensure rewards are not None and convert to numpy arrays
    rewards = np.array(rewards).flatten()
    
    observations = np.array(observations)
    average_calories = np.array(average_calories)
    average_fat = np.array(average_fat)
    average_saturates = np.array(average_saturates)
    average_carbs = np.array(average_carbs)
    average_sugar = np.array(average_sugar)
    average_fibre = np.array(average_fibre)
    average_protein = np.array(average_protein)
    average_salt = np.array(average_salt)
    current_selection = np.array(current_selection)

    print("Average Calories:", average_calories)
    print("Current Selection:", current_selection)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    font_size = 8

    # Ensure actions are in correct shape and select non-zero values
    current_selection = current_selection.reshape(-1, current_selection.shape[-1])
    non_zero_indices = np.nonzero(current_selection[0])[0]
    non_zero_values = current_selection[0, non_zero_indices]

    selected_ingredients = ingredient_df['Category7'].iloc[non_zero_indices]

    bars = axs[0].bar(selected_ingredients.values, non_zero_values, color='purple', width=0.5)
    axs[0].set_xlabel('Ingredient')
    axs[0].set_ylabel('Grams of Ingredient')
    axs[0].set_title('Selected Ingredients')
    axs[0].set_xticks(np.arange(len(selected_ingredients)))
    axs[0].set_xticklabels(selected_ingredients.values, rotation=45, ha='right', fontsize=font_size)
    
    # Add actual values on top of the bars
    for bar, actual in zip(bars, non_zero_values):
        height = bar.get_height()

        axs[0].annotate(f'{actual:.2f} g', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Calculate percentages of target values
    nutritional_values = {
        'Calories (kcal)': (average_calories, 530, 'max'),
        'Fat (g)': (average_fat, 20.6, 'max'),
        'Saturates (g)': (average_saturates, 6.5, 'max'),
        'Carbs (g)': (average_carbs, 70.6, 'min'),
        'Sugar (g)': (average_sugar, 15.5, 'max'),
        'Fibre (g)': (average_fibre, 4.2, 'min'),
        'Protein (g)': (average_protein, 7.5, 'min'),
        'Salt (g)': (average_salt, 0.499, 'max')
    }

    labels = list(nutritional_values.keys())
    actuals = [nutritional_values[key][0].mean() for key in labels]
    targets = [nutritional_values[key][1] for key in labels]
    target_types = [nutritional_values[key][2] for key in labels]

    percentages = [(actual / target) * 100 for actual, target in zip(actuals, targets)]
    colors = ['red' if (target_type == 'max' and actual > 1.05 * target) or (target_type == 'min' and actual < 0.95 * target) else 'blue' for actual, target, target_type in zip(actuals, targets, target_types)]

    x = np.arange(len(labels))
    width = 0.35

    labels = [lab.split(" ")[0] for lab in labels]
    bars = axs[1].bar(x, percentages, width, color=colors)
    
    # Add actual values on top of the bars
    for bar, actual in zip(bars, actuals):
        height = bar.get_height()
        if bar == bars[0]:
            unit = "kcal"
        else:
            unit = "g"
        axs[1].annotate(f'{actual:.2f} {unit}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    axs[1].set_xlabel('Nutritional Information')
    axs[1].set_ylabel('Percentage of Target (%)')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_title('Nutrients Percentage of Target Values (actual values on top of bar)')
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color='blue', ec='k'), plt.Rectangle((0,0),1,1, color='red', ec='k')]
    labels = ['On Target', 'Off Target']
    axs[1].legend(handles, labels)

    plt.tight_layout()
    plt.show()
    
    
class Args:
    reward_func = 'reward_nutrient_macro_and_groups'
    render_mode = None
    num_envs = 1
    plot_reward_history=False

def main():
    
    args = Args()
    
    from utils.train_utils import setup_environment, get_unique_directory, get_unique_image_directory
    
    # Setup environment and other configurations
    ingredient_df = get_data()
    
    seed = 42
    
    # Create the environment using the setup function
    env = setup_environment(args, seed, ingredient_df)

    # Load normalization statistics
    # norm_path = os.path.abspath("saved_models/evaluation_models/SchoolMealSelection-v1_A2C_100000_1env_best_seed3583778663/vec_normalize_best.pkl")
    norm_path = os.path.abspath("saved_models/evaluation_models/SchoolMealSelection-v1_A2C_100000_1env_best_seed3583778663/vec_normalize_best.pkl")
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False

    # Load the saved agent
    model_path = os.path.abspath("saved_models/evaluation_models/SchoolMealSelection-v1_A2C_100000_1env_best_seed3583778663/best_model.zip")
    model = A2C.load(model_path, env=env)

    # Number of episodes to evaluate
    num_episodes = 10

    # Evaluate the model
    predictions = evaluate_model(model, env, num_episodes)

    # Plot the results
    plot_results(predictions, ingredient_df, env)
    
    # Access the underlying RewardTrackingWrapper for saving rewards
    if args.plot_reward_history: 
        
        reward_dir = os.path.abspath(os.path.join('saved_models', 'reward'))
        reward_prefix = "test_trained"       
        # Save reward distribution for each environment in the vectorized environment
        for i, env_instance in enumerate(env.envs):
                
            reward_dir, reward_prefix = get_unique_directory(reward_dir, f"{reward_prefix}_seed{seed}_env{i}")
            
            env_instance.save_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix)))
            
            reward_prefix_instance = get_unique_image_directory(reward_dir, reward_prefix)

            env_instance.plot_reward_distribution(os.path.abspath(os.path.join(reward_dir, reward_prefix_instance)))

if __name__ == "__main__":
    main()
