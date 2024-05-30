import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback
from models.envs.env2 import CalorieOnlyEnv  # Updated import to reflect new environment

# Setup environment and other configurations
ingredient_df = get_data()
env = gym.make('CalorieOnlyEnv-v1', ingredient_df=ingredient_df, render_mode=None, num_people=50, target_calories=530)
env = DummyVecEnv([lambda: env])

log_dir = os.path.abspath("saved_models/tensorboard/A2C_newEnv1000stepmil")
save_dir = os.path.abspath("saved_models/checkpoints/A2C_newEnv1000stepmil")
new_logger = configure(log_dir, ["stdout", "tensorboard"])

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix='A2C_newEnv1000stepmil')
eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=500)
info_logger_callback = InfoLoggerCallback()

callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)
model.learn(total_timesteps=1000000, callback=callback)

model.save(os.path.join(save_dir, "A2C_newEnv1000stepmil"))

del model
