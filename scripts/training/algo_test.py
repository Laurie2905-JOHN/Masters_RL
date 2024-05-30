import os
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from utils.process_data import get_data
from models.envs.env1 import SimpleCalorieOnlyEnv
from utils.train_utils import InfoLoggerCallback

try:
    # Get data and create environment
    ingredient_df = get_data()
    env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')
    env = DummyVecEnv([lambda: env])  # Wrap the environment
    print("Environment created successfully.")
except Exception as e:
    print(f"Error creating environment: {e}")
    raise

# Set up directories
log_dir = os.path.abspath("saved_models/tensorboard/")
save_dir = os.path.abspath("saved_models/checkpoints/")

# List of algorithms to try
algorithms = {
    'A2C': A2C,
    'PPO': PPO,
}

# Loop through algorithms
for algo_name, algo_class in algorithms.items():
    try:
        algo_log_dir = os.path.join(log_dir, algo_name)
        algo_save_dir = os.path.join(save_dir, algo_name)
        
        # Configure logger
        new_logger = configure(algo_log_dir, ["stdout", "tensorboard"])

        # Create callbacks
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=algo_save_dir, name_prefix=f'{algo_name}_model')
        eval_callback = EvalCallback(env, best_model_save_path=algo_save_dir, log_path=algo_log_dir, eval_freq=500)
        info_logger_callback = InfoLoggerCallback()
        callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

        # Instantiate the model
        model = algo_class('MlpPolicy', env, verbose=1, tensorboard_log=algo_log_dir)
        model.set_logger(new_logger)

        try:
            # Train the model
            model.learn(total_timesteps=10000, callback=callback)
            print(f"Training completed for {algo_name}.")
        except Exception as e:
            print(f"Error during training with {algo_name}: {e}")

        try:
            # Save the model
            model.save(os.path.join(algo_save_dir, f"{algo_name}_simple_calorie_env"))
            print(f"Model saved for {algo_name}.")
        except Exception as e:
            print(f"Error saving model for {algo_name}: {e}")

        del model  # Clean up

    except Exception as e:
        print(f"Error setting up {algo_name}: {e}")

print("Training with all algorithms completed.")
