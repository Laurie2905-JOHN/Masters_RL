import gym
import torch
from models.algo.actor import Actor
from models.algo.critic import Critic
from models.envs.env import SimpleCalorieOnlyEnv
from utils.process_data import get_data

# def train():
    
#     ingredient_df = get_data()
    
#     env = SimpleCalorieOnlyEnv(ingredient_df)
    
#     model = A2C('MlpPolicy', env, verbose=1)
    
#     model.learn(total_timesteps=10000)
    
#     model.save("../saved_models/a2c_custom_env")

# if __name__ == "__main__":
#     train()
