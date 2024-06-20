import os
import pickle as pkl
import random
import sys
import time
import argparse
from pprint import pprint
import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from utils.optuna_utils.sample_params.ppo import sample_ppo_params
from utils.optuna_utils.sample_params.a2c import sample_a2c_params
from models.envs.env_working import SchoolMealSelection
from utils.optuna_utils.trial_eval_callback import TrialEvalCallback
from utils.process_data import get_data
from gymnasium.wrappers import TimeLimit, NormalizeObservation, NormalizeReward
import time

def objective(trial: optuna.Trial, ingredient_df, study_path, num_timesteps, algo) -> float:
    time.sleep(random.random() * 16)

    max_ingredients = trial.suggest_categorical("max_ingredients", [5, 6, 7, 8])
    action_scaling_factor = trial.suggest_categorical("action_scaling_factor", [10, 15, 20])

    env_kwargs = {
        "ingredient_df": ingredient_df, 
        "max_ingredients": max_ingredients, 
        "action_scaling_factor": action_scaling_factor
    }
    
    path = os.path.abspath(f"{study_path}/trials/trial_{str(trial.number)}")
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    def make_env():
        env = gym.make("SchoolMealSelection-v0", **env_kwargs)
        env = TimeLimit(env, max_episode_steps=1000)
        env = NormalizeObservation(env)
        env = NormalizeReward(env)
        env = Monitor(env)
        return env     
    
    env = make_vec_env(make_env, n_envs=1, seed=None)

    if algo == "PPO":
        sampled_hyperparams = sample_ppo_params(trial)
        model = PPO("MlpPolicy", env=env, seed=None, verbose=0, device='cuda', tensorboard_log=path, **sampled_hyperparams)
    elif algo == "A2C":
        sampled_hyperparams = sample_a2c_params(trial)
        model = A2C("MlpPolicy", env=env, seed=None, verbose=0, device='cpu', tensorboard_log=path, **sampled_hyperparams)
    else:
        raise ValueError("Invalid algorithm")
    
    # stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=3, eval_freq=5000, deterministic=True
    )
    
    if 'ingredient_df' in env_kwargs:
        del env_kwargs['ingredient_df']

    params = env_kwargs | sampled_hyperparams

    try:
        model.learn(num_timesteps, eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward
    
    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))
        f.write(f"\nReward: {reward}")

    return reward

def benchmark(algo, study_name, storage, n_trials, timeout, num_timesteps, n_jobs_list):
    INGREDIENT_DF = get_data()
    study_path = f"saved_models/optuna/{study_name}"
    os.makedirs(study_path, exist_ok=True)
    
    db_dir = os.path.join(study_path, "db")
    os.makedirs(db_dir, exist_ok=True)
    
    if storage is None:
        storage = f"sqlite:///{os.path.join(db_dir, f'{study_name}.db')}"
    
    results = []
    
    for n_jobs in n_jobs_list:
        print(f"Running with n_jobs={n_jobs}...")
        start_time = time.time()
        
        sampler = TPESampler(n_startup_trials=10, multivariate=True)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(lambda trial: objective(trial, INGREDIENT_DF, study_path, num_timesteps=num_timesteps, algo=algo), 
                           n_jobs=n_jobs, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        except KeyboardInterrupt:
            pass

        end_time = time.time()
        elapsed_time = end_time - start_time
        results.append((n_jobs, elapsed_time))
        
        print(f"n_jobs={n_jobs}, elapsed_time={elapsed_time:.2f} seconds")
    
    print("Benchmark results:")
    for n_jobs, elapsed_time in results:
        print(f"n_jobs={n_jobs}: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="PPO", help="Algorithm to optimize: PPO or A2C")
    parser.add_argument('--study_name', type=str, default=None, help="Name of the Optuna study")
    parser.add_argument('--storage', type=str, default=None, help="Database URL for Optuna storage")
    parser.add_argument('--n_trials', type=int, default=8, help="Number of trials for optimization")
    parser.add_argument('--timeout', type=int, default=7200, help="Timeout for optimization in seconds")
    parser.add_argument('--num_timesteps', type=int, default=20000, help="Number of timesteps for model training")
    parser.add_argument('--n_jobs_list', type=int, nargs='+', default=[1, 2, 4, 8], help="List of n_jobs values to benchmark")
    args = parser.parse_args()
    
    if args.study_name is None:
        args.study_name = f"{args.algo}_benchmark_test"
        
    benchmark(args.algo, args.study_name, args.storage, args.n_trials, args.timeout, args.num_timesteps, args.n_jobs_list)
