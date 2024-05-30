import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback
from models.envs.env2 import CalorieOnlyEnv  # Updated import to reflect new environment

# Setup environment and other configurations
ingredient_df = get_data()

# Number of parallel environments
num_envs = 8

# Create vectorized environments using make_vec_env
env = make_vec_env('CalorieOnlyEnv-v1', n_envs=num_envs, env_kwargs={'ingredient_df': ingredient_df, 'render_mode': None, 'num_people': 50, 'target_calories': 530})
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

log_dir = os.path.abspath("saved_models/tensorboard/A2C_100000_8env")
save_dir = os.path.abspath("saved_models/checkpoints/A2C_100000_8env")
new_logger = configure(log_dir, ["stdout", "tensorboard"])

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix='A2C_100000_8env')
eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=1000, deterministic=True)
info_logger_callback = InfoLoggerCallback()

callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)
model.learn(total_timesteps=100000, callback=callback)

model.save(os.path.join(save_dir, "A2C_100000_8env"))

# Save the VecNormalize statistics
env.save(os.path.join(save_dir, "vec_normalize.pkl"))

del model

# To load the model and VecNormalize later
env = make_vec_env('CalorieOnlyEnv-v1', n_envs=num_envs, env_kwargs={'ingredient_df': ingredient_df, 'render_mode': None, 'num_people': 50, 'target_calories': 530})
env = VecNormalize.load(os.path.join(save_dir, "vec_normalize.pkl"), env)

model = A2C.load(os.path.join(save_dir, "A2C_100000_8env"), env=env)
