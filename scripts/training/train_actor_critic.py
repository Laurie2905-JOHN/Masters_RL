# import os, sys
# # Ensure the src directory is in the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
import os
from utils.process_data import get_data
# Make sure environment is imported in
from models.envs.env import SimpleCalorieOnlyEnv

    
ingredient_df = get_data()

# Check if environment is registered
env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')
print("Environment created successfully.")

# Set up log directory
log_dir = os.path.abspath("Masters_RL/saved_models/tensorboard/a2c_simple_calorie_env/")
# Set up save directory
save_dir = os.path.abspath("Masters_RL/saved_models/checkpoints/a2c_simple_calorie_env/")

# Configure logger
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Create callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_dir,
                                            name_prefix='a2c_model')
eval_callback = EvalCallback(env, best_model_save_path=save_dir,
                                log_path=log_dir, eval_freq=500)

callback = CallbackList([checkpoint_callback, eval_callback])

# Train A2C model with logging and callbacks
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)
model.learn(total_timesteps=5000, callback=callback)

# Save the model
model.save(os.path.join(save_dir, "a2c_simple_calorie_env"))

del model
    
