import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils.process_data import get_data
from models.envs.env_working import SchoolMealSelection
import argparse

def ppo_objective(trial, n_envs, num_timesteps, num_seeds=5):
    n_steps = trial.suggest_int('n_steps', 16, 2048)
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

    # VecNormalize parameters
    vecnorm_params = {
        'norm_obs': trial.suggest_categorical('vecnorm_norm_obs', [True, False]),
        'norm_reward': trial.suggest_categorical('vecnorm_norm_reward', [True, False]),
        'clip_obs': trial.suggest_float('vecnorm_clip_obs', 1.0, 10.0),
        'clip_reward': trial.suggest_float('vecnorm_clip_reward', 1.0, 10.0),
        'gamma': params['gamma'],  # Use the same gamma as the agent
        'epsilon': trial.suggest_float('vecnorm_epsilon', 1e-8, 1e-3, log=True),
        'norm_obs_keys': None  # Adjust as necessary
    }

    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    rewards = []

    for seed in range(num_seeds):
        # Set the seed
        np.random.seed(seed)
        env = make_vec_env(env_fn, n_envs=n_envs)
        env = VecNormalize(env, **vecnorm_params)
        model = PPO('MlpPolicy', env, seed=seed, **params, verbose=0)

        # Train the model
        model.learn(total_timesteps=num_timesteps)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        rewards.append(mean_reward)

    return np.mean(rewards)

def a2c_objective(trial, n_envs, num_timesteps, num_seeds=5):
    n_steps = trial.suggest_int('n_steps', 16, 2048)

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

    # VecNormalize parameters
    vecnorm_params = {
        'norm_obs': trial.suggest_categorical('vecnorm_norm_obs', [True, False]),
        'norm_reward': trial.suggest_categorical('vecnorm_norm_reward', [True, False]),
        'clip_obs': trial.suggest_float('vecnorm_clip_obs', 1.0, 10.0),
        'clip_reward': trial.suggest_float('vecnorm_clip_reward', 1.0, 10.0),
        'gamma': params['gamma'],  # Use the same gamma as the agent
        'epsilon': trial.suggest_float('vecnorm_epsilon', 1e-8, 1e-3, log=True),
        'norm_obs_keys': None  # Adjust as necessary
    }

    ingredient_df = get_data()  # Replace with your method to get the ingredient data
    env_fn = lambda: SchoolMealSelection(ingredient_df)
    rewards = []

    for seed in range(num_seeds):
        # Set the seed
        np.random.seed(seed)
        env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv) 
        env = VecNormalize(env, **vecnorm_params)
        model = A2C('MlpPolicy', env, seed=seed, **params, verbose=0)

        # Train the model
        model.learn(total_timesteps=num_timesteps)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=15)
        rewards.append(mean_reward)

    return np.mean(rewards)

def trial_complete_callback(study, trial):
    print(f"Trial {trial.number} completed with value: {trial.value} and parameters: {trial.params}")

def optimize_ppo(study_name, storage, n_trials=1000, timeout=43200, n_envs=4, num_timesteps=1000000, num_seeds=5):
    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage, load_if_exists=False)
    try:
        study.optimize(lambda trial: ppo_objective(trial, n_envs, num_timesteps, num_seeds), n_trials=n_trials, timeout=timeout, callbacks=[trial_complete_callback])
    except Exception as e:
        print(f"Optimization stopped due to: {e}")
    best_params_ppo = study.best_params
    print("Best parameters for PPO:", best_params_ppo)

def optimize_a2c(study_name, storage, n_trials=1000, timeout=43200, n_envs=4, num_timesteps=1000000, num_seeds=5):
    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage, load_if_exists=False)
    try:
        study.optimize(lambda trial: a2c_objective(trial, n_envs, num_timesteps, num_seeds), n_trials=n_trials, timeout=timeout, callbacks=[trial_complete_callback])
    except Exception as e:
        print(f"Optimization stopped due to: {e}")
    best_params_a2c = study.best_params
    print("Best parameters for A2C:", best_params_a2c)

def main(algo, study_name, storage, n_trials, timeout, n_envs, num_timesteps):
    if algo == 'PPO':
        optimize_ppo(study_name=study_name, storage=storage, n_trials=n_trials, timeout=timeout, n_envs=n_envs, num_timesteps=num_timesteps)
    elif algo == 'A2C':
        optimize_a2c(study_name=study_name, storage=storage, n_trials=n_trials, timeout=timeout, n_envs=n_envs, num_timesteps=num_timesteps)
    else:
        raise ValueError("Unsupported algorithm specified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="PPO", help="Algorithm to optimize: PPO or A2C")
    parser.add_argument('--study_name', type=str, default=None, help="Name of the Optuna study")
    parser.add_argument('--storage', type=str, default="sqlite:///example_PPO.db", help="Database URL for Optuna storage")
    parser.add_argument('--n_trials', type=int, default=1000, help="Number of trials for optimization")
    parser.add_argument('--timeout', type=int, default=43200, help="Timeout for optimization in seconds")
    parser.add_argument('--n_envs', type=int, default=4, help="Number of environments to use per trial")
    parser.add_argument('--num_timesteps', type=int, default=1000000, help="Number of timesteps for model training")
    args = parser.parse_args()
    
    if args.study_name is None:
        args.study_name = f"{args.algo}_study"
        
    main(args.algo, args.study_name, args.storage, args.n_trials, args.timeout, args.n_envs, args.num_timesteps)
