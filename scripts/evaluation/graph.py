import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from utils.process_data import get_data

# Setup environment and other configurations
ingredient_df = get_data()
env = gym.make('CalorieOnlyEnv-v1', ingredient_df=ingredient_df, render_mode=None)
env = DummyVecEnv([lambda: env])

# Load the saved agent
model = A2C.load(os.path.abspath("saved_models/best_models/A2C_env2_trainedmodel_100000"))

# Number of episodes or steps to predict
episodes = 1

# List to store predictions
predictions = []

# Run the agent in the environment
obs = env.reset()
for step in range(episodes):
    action, _states = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    print(info)
    predictions.append((obs, action, reward, info[0]['Average Calories per Day'], info[0]['Calories per selected']))
    if terminated:
        obs = env.reset()

# Convert predictions to a more usable form for plotting
observations, actions, rewards, average_calories, calories_perSelected = zip(*predictions)

# Ensure rewards are not None and convert to numpy arrays
rewards = np.array(rewards).flatten()
observations = np.array(observations)
actions = np.array(actions)
average_calories = np.array(average_calories)
calories_perSelected = np.array(calories_perSelected)

print(average_calories)
print(calories_perSelected)

# # Create subplots
# fig, axs = plt.subplots(2, 1, figsize=(14, 8))
# font_size = 8

# # Plotting scaled actions (Quantities)
# selected_ingredients = ingredient_df['Category7'][non_zero_indices]

# axs[0].bar(selected_ingredients.values, non_zero_values, color='purple', width=0.5)
# axs[0].set_xlabel('Ingredient')
# axs[0].set_ylabel('Grams of Ingredient')
# axs[0].set_title(f'Num Ingredient: {len(selected_ingredients)}, Calories per person: {round(average_calories[0], 2)}')
# axs[0].set_xticks(np.arange(len(selected_ingredients)))
# axs[0].set_xticklabels(selected_ingredients.values, rotation=45, ha='right', fontsize=font_size)

# # Plotting calorific amounts
# caloric_values = calories_perSelected[0][non_zero_indices]
# axs[1].bar(selected_ingredients.values, caloric_values, color='orange', width=0.5)
# axs[1].set_xlabel('Ingredient')
# axs[1].set_ylabel('Calories')
# axs[1].set_title('Calories per Ingredient')
# axs[1].set_xticks(np.arange(len(selected_ingredients)))
# axs[1].set_xticklabels(selected_ingredients.values, rotation=45, ha='right', fontsize=font_size)

# plt.tight_layout()
# plt.show()

