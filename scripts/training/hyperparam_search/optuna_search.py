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
from gymnasium.wrappers import TimeLimit, NormalizeReward
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Function to set up the environment
def setup_environment(args):
    env_kwargs = {
        "ingredient_df": args.ingredient_df,
        "max_ingredients": args.max_ingredients,
        "action_scaling_factor": args.action_scaling_factor,
        "render_mode": args.render_mode,
        "seed": args.seed,
        "verbose": args.verbose,
        "initialization_strategy": args.initialization_strategy
    }



def objective(trial: optuna.Trial, ingredient_df, study_path, num_timesteps, algo) -> float:
    # Prevent Resource Contention: When multiple trials start simultaneously, they might contend for limited computational resources
    time.sleep(random.random() * 16)

    max_ingredients = trial.suggest_categorical("max_ingredients", [6, 8])

    # initialization_strategy = trial.suggest_categorical("initialization_strategy", ['zero', 'probabilistic', 'perfect'])

    initialization_strategy = 'zero'
    
    env_kwargs = {
        "ingredient_df": ingredient_df, 
        "max_ingredients": max_ingredients, 
        "action_scaling_factor": 10,
        "initialization_strategy": initialization_strategy,
    }
    
    
    path = os.path.abspath(f"{study_path}/trials/trial_{str(trial.number)}")
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    def make_env():
        env = gym.make("SchoolMealSelection-v0", **env_kwargs)
        
        env = TimeLimit(env, max_episode_steps=100)
        env = Monitor(env)  # Monitoring is added to track statistics and/or save logs
        
        return env

    # Wrap the environment with DummyVecEnv for parallel environments
    env = make_vec_env(make_env, n_envs=4, seed=None)

    clip_obs = trial.suggest_categorical("clip_obs", [5, 10, 50, 100])
    norm_reward = trial.suggest_categorical("norm_reward", [False, True])

    if algo == "PPO":
        sampled_hyperparams = sample_ppo_params(trial)
    elif algo == "A2C":
        sampled_hyperparams = sample_a2c_params(trial)
    else:
        raise ValueError("Invalid algorithm")
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=100,
        gamma=sampled_hyperparams['gamma'],
        norm_obs_keys=['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
    )

    if algo == "PPO":
        model = PPO("MultiInputPolicy", env=env, seed=None, verbose=0, device='cuda', tensorboard_log=path, **sampled_hyperparams)
    elif algo == "A2C":
        model = A2C("MultiInputPolicy", env=env, seed=None, verbose=0, device='cpu', tensorboard_log=path, **sampled_hyperparams)
    else:
        raise ValueError("Invalid algorithm")
    
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=50, verbose=1)
    
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=10000, deterministic=True, callback_after_eval=stop_callback
    )
    
    if 'ingredient_df' in env_kwargs:
        del env_kwargs['ingredient_df']

    params = env_kwargs | sampled_hyperparams | {'clip_obs': clip_obs, 'norm_reward': norm_reward}
    

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

def main(algo, study_name, storage, n_trials, timeout, n_jobs, num_timesteps):
    INGREDIENT_DF = get_data()

    study_path = f"saved_models/optuna/{study_name}"
    os.makedirs(study_path, exist_ok=True)  # Ensure the study path directory exists
    
    db_dir = os.path.join(study_path, "db")
    os.makedirs(db_dir, exist_ok=True)  # Ensure the directory exists
    
    if storage is None:
        storage = f"sqlite:///{os.path.join(db_dir, f'{study_name}.db')}"

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

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        if key != "ingredient_df":
            print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        # Save figures to files in the specified directory
        fig1.write_image(f"{study_path}/optimization_history.png")
        fig2.write_image(f"{study_path}/param_importances.png")
        fig3.write_image(f"{study_path}/parallel_coordinate.png")

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="PPO", help="Algorithm to optimize: PPO or A2C")
    parser.add_argument('--study_name', type=str, default=None, help="Name of the Optuna study")
    parser.add_argument('--storage', type=str, default=None, help="Database URL for Optuna storage")
    parser.add_argument('--n_trials', type=int, default=500, help="Number of trials for optimization")
    parser.add_argument('--timeout', type=int, default=260000, help="Timeout for optimization in seconds")
    parser.add_argument('--n_jobs', type=int, default=3, help="Number of jobs to assign")
    parser.add_argument('--num_timesteps', type=int, default=180000, help="Number of timesteps for model training")
    args = parser.parse_args()
    
    if args.study_name is None:
        args.study_name = f"{args.algo}_DenseReward"
        
    main(args.algo, args.study_name, args.storage, args.n_trials, args.timeout, args.n_jobs, args.num_timesteps)
