import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback
from models.envs.env1 import SimpleCalorieOnlyEnv
import numpy as np
import optuna

# Setup environment and other configurations
ingredient_df = get_data()
env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode=None)
env = DummyVecEnv([lambda: env])

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical('buffer_size', [100000, 200000, 300000])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    tau = trial.suggest_uniform('tau', 0.01, 0.1)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    
    # Set up log and save directories
    log_dir = os.path.abspath(f"saved_models/tensorboard/SAC_test_trial_{trial.number}")
    save_dir = os.path.abspath(f"saved_models/checkpoints/SAC_test_trial_{trial.number}")
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix='SAC_model')
    eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=2500)
    info_logger_callback = InfoLoggerCallback()

    callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

    # Instantiate the model
    model = SAC('MlpPolicy', env, verbose=0, tensorboard_log=log_dir,
                learning_rate=learning_rate, buffer_size=buffer_size,
                batch_size=batch_size, tau=tau, gamma=gamma)
    model.set_logger(new_logger)
    
    # Train the model
    model.learn(total_timesteps=50000, callback=callback)

    # Evaluate the model
    eval_env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode=None)
    eval_env = DummyVecEnv([lambda: eval_env])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    # Save the model
    model.save(os.path.join(save_dir, "SAC_simple_calorie_env"))
    del model

    return mean_reward

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters: ", study.best_params)
