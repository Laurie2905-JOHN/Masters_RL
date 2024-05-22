import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from utils.process_data import get_data
from models.envs.env import SimpleCalorieOnlyEnv


# Setup environment and other configurations
ingredient_df = get_data()
# testenv = SimpleCalorieOnlyEnv(ingredient_df)
env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode=None)
env = DummyVecEnv([lambda: env])

# Load the saved agent
model = A2C.load(os.path.abspath("saved_models/best_models/A2C_working_model"))

# Number of episodes or steps to predict
episodes = 1

# List to store predictions
predictions = []

# Run the agent in the environment
obs = env.reset()
for step in range(episodes):
    action, _states = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    predictions.append((action, reward, info))
    if terminated:
        obs = env.reset()

# Convert predictions to a more usable form for plotting
actions, rewards, infos = zip(*predictions)

# Debug prints to check the values
print("Actions:", actions)
print("Rewards:", rewards)
print("Infos:", infos)

# Ensure rewards are not None and convert to numpy arrays
rewards = np.array(rewards)
actions = np.array(actions)
infos = np.array(infos)

# Check if there are any None values in rewards
if None in rewards:
    print("Error: Found None in rewards")
else:
    # Plot the results
    steps = np.arange(1, episodes + 1)

    plt.figure(figsize=(14, 7))

    # Plotting the rewards
    plt.subplot(3, 1, 1)
    plt.bar(steps, rewards, color='blue')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards per Step for RL Agent')

    # Plotting the actions (assuming actions are single-dimensional for simplicity)
    plt.subplot(3, 1, 3)
    plt.plot(steps, actions[:, 0], color='red')
    plt.xlabel('Step')
    plt.ylabel('Action')
    plt.title('Actions per Step for RL Agent')

    plt.tight_layout()
    plt.show()