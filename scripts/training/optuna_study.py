import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils.process_data import get_data  
from models.envs.env_working import SchoolMealSelection  
import joblib

def ppo_objective(trial, n_gpus):
    n_steps = trial.suggest_int('n_steps', 16, 2048)
    n_envs = 8
    possible_batch_sizes = [bs for bs in range(32, 257) if (n_steps * n_envs) % bs == 0]
    
    if not possible_batch_sizes:
        possible_batch_sizes = [n_steps * n_envs]

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'n_steps': n_steps,
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 1.0),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'batch_size': trial.suggest_categorical('batch_size', possible_batch_sizes),
        'n_epochs': trial.suggest_int('n_epochs', 1, 10),
    }

    # Create the environment
    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    env = make_vec_env(env_fn, n_envs=n_envs)

    # Initialize the PPO model
    model = PPO('MlpPolicy', env, device=f'cuda:{trial.number % n_gpus}', **params, verbose=0)

    # Train the model
    model.learn(total_timesteps=1000000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

def a2c_objective(trial):
    n_steps = trial.suggest_int('n_steps', 16, 2048)
    n_envs = 8
    possible_batch_sizes = [bs for bs in range(32, 257) if (n_steps * n_envs) % bs == 0]
    
    if not possible_batch_sizes:
        possible_batch_sizes = [n_steps * n_envs]

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'n_steps': n_steps,
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 1.0),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'rms_prop_eps': trial.suggest_float('rms_prop_eps', 1e-5, 1e-3, log=True),
        'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
    }

    # Create the environment
    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    env = make_vec_env(env_fn, n_envs=n_envs)

    # Initialize the A2C model
    model = A2C('MlpPolicy', env, device='cpu', **params, verbose=0)

    # Train the model
    model.learn(total_timesteps=1000000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

def optimize_ppo(n_trials=1000, timeout=43200, n_jobs=16, n_gpus=4):
    # Create Optuna study for PPO
    ppo_study = optuna.create_study(direction='maximize')

    # Optimize using joblib for parallel execution
    with joblib.parallel_backend('loky', n_jobs=n_jobs):
        ppo_study.optimize(lambda trial: ppo_objective(trial, n_gpus), n_trials=n_trials, timeout=timeout)

    # Save the best parameters for PPO
    best_params_ppo = ppo_study.best_params
    print("Best parameters for PPO:", best_params_ppo)

def optimize_a2c(n_trials=1000, timeout=43200, n_jobs=16):
    # Create Optuna study for A2C
    a2c_study = optuna.create_study(direction='maximize')

    # Optimize using joblib for parallel execution
    with joblib.parallel_backend('loky', n_jobs=n_jobs):
        a2c_study.optimize(lambda trial: a2c_objective(trial), n_trials=n_trials, timeout=timeout)

    # Save the best parameters for A2C
    best_params_a2c = a2c_study.best_params
    print("Best parameters for A2C:", best_params_a2c)


