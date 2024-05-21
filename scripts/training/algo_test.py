import os
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from utils.process_data import get_data
from models.envs.env import SimpleCalorieOnlyEnv
from utils.train_utils import InfoLoggerCallback


# Get data and create environment
ingredient_df = get_data()
env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')
env = DummyVecEnv([lambda: env])  # Wrap the environment

print("Environment created successfully.")

# Set up directories
log_dir = os.path.abspath("saved_models/tensorboard/")
save_dir = os.path.abspath("saved_models/checkpoints/")

# List of algorithms to try
algorithms = {
    'A2C': A2C,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

# Loop through algorithms
for algo_name, algo_class in algorithms.items():
    algo_log_dir = os.path.join(log_dir, algo_name)
    algo_save_dir = os.path.join(save_dir, algo_name)
    
    # Configure logger
    new_logger = configure(algo_log_dir, ["stdout", "tensorboard"])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=algo_save_dir, name_prefix=f'{algo_name}_model')
    eval_callback = EvalCallback(env, best_model_save_path=algo_save_dir, log_path=algo_log_dir, eval_freq=50000)
    info_logger_callback = InfoLoggerCallback()
    callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

    # Instantiate the model
    model = algo_class('MlpPolicy', env, verbose=1, tensorboard_log=algo_log_dir)
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=1000000, callback=callback)

    # Save the model
    model.save(os.path.join(algo_save_dir, f"{algo_name}_simple_calorie_env"))

    del model  # Clean up

print("Training with all algorithms completed.")
