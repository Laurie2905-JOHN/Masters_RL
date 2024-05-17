import gym
import torch
from stable_baselines3 import A2C
from custom_env import CustomEnv  # Adjust the import based on your actual path

def evaluate_agent(env, model, num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")

if __name__ == "__main__":
    ingredient_df = ...  # Load your ingredient dataframe here
    env = CustomEnv(ingredient_df)
    
    model = A2C.load("../saved_models/a2c_custom_env")
    
    evaluate_agent(env, model)
