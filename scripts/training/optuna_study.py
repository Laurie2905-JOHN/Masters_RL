import optuna
import optuna_distributed
from dask.distributed import Client
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils.process_data import get_data
from models.envs.env_working import SchoolMealSelection

def ppo_objective(trial, n_gpus, num_seeds=5):
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

    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    rewards = []

    for seed in range(num_seeds):
        # Set the seed
        np.random.seed(seed)
        env = make_vec_env(env_fn, n_envs=n_envs)
        model = PPO('MlpPolicy', env, device=f'cuda:{trial.number % n_gpus}', seed=seed, **params, verbose=0)

        # Train the model
        model.learn(total_timesteps=1000000)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        rewards.append(mean_reward)

    return np.mean(rewards)

def a2c_objective(trial, num_seeds=5):
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

    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    rewards = []

    for seed in range(num_seeds):
        # Set the seed
        np.random.seed(seed)
        env = make_vec_env(env_fn, n_envs=n_envs)
        model = A2C('MlpPolicy', env, device='cpu', seed=seed, **params, verbose=0)

        # Train the model
        model.learn(total_timesteps=1000000)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=15)
        rewards.append(mean_reward)

    return np.mean(rewards)

def optimize_ppo(study_name, storage_url, n_trials=1000, timeout=43200, n_gpus=4, num_seeds=5):
    client = None  # Use Dask client if available, e.g., Client("<your.cluster.scheduler.address>")
    base_study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage_url, load_if_exists=True)
    study = optuna_distributed.from_study(base_study, client=client)
    study.optimize(lambda trial: ppo_objective(trial, n_gpus, num_seeds), n_trials=n_trials, timeout=timeout)
    best_params_ppo = study.best_params
    print("Best parameters for PPO:", best_params_ppo)

def optimize_a2c(study_name, storage_url, n_trials=1000, timeout=43200, num_seeds=5):
    client = None  # Use Dask client if available, e.g., Client("<your.cluster.scheduler.address>")
    base_study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage_url, load_if_exists=True)
    study = optuna_distributed.from_study(base_study, client=client)
    study.optimize(lambda trial: a2c_objective(trial, num_seeds), n_trials=n_trials, timeout=timeout)
    best_params_a2c = study.best_params
    print("Best parameters for A2C:", best_params_a2c)

import argparse

def main(algo, study_name, storage_url, n_trials, timeout, n_gpus):
    if algo == 'PPO':
        optimize_ppo(study_name=study_name, storage_url=storage_url, n_trials=n_trials, timeout=timeout, n_gpus=n_gpus)
    elif algo == 'A2C':
        optimize_a2c(study_name=study_name, storage_url=storage_url, n_trials=n_trials, timeout=timeout)
    else:
        raise ValueError("Unsupported algorithm specified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="A2C", help="Algorithm to optimize: PPO or A2C")
    parser.add_argument('--study_name', type=str, required=True, help="Name of the Optuna study")
    parser.add_argument('--storage_url', type=str, required=True, help="URL of the Optuna RDB storage")
    parser.add_argument('--n_trials', type=int, default=1000, help="Number of trials for optimization")
    parser.add_argument('--timeout', type=int, default=259200, help="Timeout for optimization in seconds")
    parser.add_argument('--n_gpus', type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args.algo, args.study_name, args.storage_url, args.n_trials, args.timeout, args.n_gpus)
