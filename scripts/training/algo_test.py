import os
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from utils.process_data import get_data
from models.envs.env import SimpleCalorieOnlyEnv

ingredient_df = get_data()

# Check if environment is registered
env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')
print("Environment created successfully.")

# Set up log directory
log_dir = os.path.abspath("saved_models/tensorboard/")
# Set up save directory
save_dir = os.path.abspath("saved_models/checkpoints/")

# List of algorithms to try
algorithms = {
    'A2C': A2C,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

for algo_name, algo_class in algorithms.items():
    algo_log_dir = os.path.join(log_dir, algo_name)
    algo_save_dir = os.path.join(save_dir, algo_name)
    
    # Configure logger
    new_logger = configure(algo_log_dir, ["stdout", "tensorboard"])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=algo_save_dir, name_prefix=f'{algo_name}_model')
    eval_callback = EvalCallback(env, best_model_save_path=algo_save_dir, log_path=algo_log_dir, eval_freq=50000)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Instantiate the model
    model = algo_class('MlpPolicy', env, verbose=1, tensorboard_log=algo_log_dir)
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=1000000, callback=callback)

    # Save the model
    model.save(os.path.join(algo_save_dir, f"{algo_name}_simple_calorie_env"))

    del model  # Clean up

print("Training with all algorithms completed.")
