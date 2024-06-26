import os
import argparse
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure, HumanOutputFormat
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback, SaveVecNormalizeEvalCallback
from utils.train_utils import generate_random_seeds, setup_environment, get_unique_directory, select_device, set_seed, monitor_memory_usage, plot_reward_distribution, linear_schedule
import json
import yaml
from torch import nn

def ensure_dir_exists(directory, verbose=0):
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose > 1:
            print(f"Warning: Directory {directory} did not exist and was created.")

# Main training function
def main(args):
    device = select_device(args)
    
    set_seed(args.seed, device)

    print(f"Using device: {device}")

    # Load data required for environment setup
    try:
        args.ingredient_df = get_data("small_data.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure directories exist
    ensure_dir_exists(args.reward_dir, args.verbose)
    ensure_dir_exists(args.log_dir, args.verbose)
    ensure_dir_exists(args.save_dir, args.verbose)
    ensure_dir_exists(args.best_dir, args.verbose)
    ensure_dir_exists(args.hyperparams_dir, args.verbose)

    if args.plot_reward_history:
        reward_dir, reward_prefix = get_unique_directory(args.reward_dir, f"{args.reward_prefix}_seed_{seed}_env", '.json')
        reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
    else:
        reward_save_path = None

    # Create and normalize vectorized environments with shared reward dictionary
    env = setup_environment(args, reward_save_path, eval=False)

    if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.lower() != 'none':
        checkpoint_path = os.path.abspath(args.pretrained_checkpoint_path)
        print(f"Loading model from checkpoint: {checkpoint_path}")
        tensorboard_log_dir = os.path.join(args.log_dir, f"{args.log_prefix}_seed_{seed}")
        reset_num_timesteps = False

        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)
        parts = checkpoint_name.split("_")
        steps = parts[-2]
        pretrained = "_model_pretrained"
        pkl_name = f"{'_'.join(parts[:-2])}_vecnormalize_{steps}_steps.pkl"
        pkl_path = os.path.join(checkpoint_dir, pkl_name)

        if os.path.exists(pkl_path):
            print(f"Loading VecNormalize from: {pkl_path}")
            env = VecNormalize.load(pkl_path, env)
        else:
            print(f"VecNormalize file does not exist: {pkl_path}")
            return

        if args.algo == 'A2C':
            model = A2C.load(checkpoint_path, env=env, verbose=args.verbose, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
        elif args.algo == 'PPO':
            model = PPO.load(checkpoint_path, env=env, verbose=args.verbose, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
        else:
            raise ValueError(f"Unsupported algorithm: {args.algo}")

    else:
        log_dir, log_prefix = get_unique_directory(args.log_dir, f"{args.log_prefix}_seed_{seed}", ".zip")
        tensorboard_log_dir = os.path.join(log_dir, log_prefix)
        pretrained = ""

        common_hyperparams = {
            'verbose': args.verbose,
            'tensorboard_log': tensorboard_log_dir,
            'device': device,
            'seed': seed,
            'gamma': args.gamma,
        }

        if args.algo == 'A2C':
            a2c_hyperparams = {
                'n_steps': args.a2c_n_steps,
                'learning_rate': args.learning_rate,
                'ent_coef': args.a2c_ent_coef,
                'vf_coef': args.a2c_vf_coef,
                'max_grad_norm': args.a2c_max_grad_norm,
                'rms_prop_eps': args.a2c_rms_prop_eps,
                'use_rms_prop': args.a2c_use_rms_prop,
                'use_sde': args.a2c_use_sde,
                'sde_sample_freq': args.a2c_sde_sample_freq,
                'normalize_advantage': args.a2c_normalize_advantage,
                "gae_lambda": args.a2c_gae_lambda,
                'policy_kwargs': args.policy_kwargs,
            }
            print(f'Training with A2C algorithm and hyper params are {a2c_hyperparams}')
            model = A2C('MultiInputPolicy', env, **common_hyperparams, **a2c_hyperparams)
        elif args.algo == 'PPO':
            ppo_hyperparams = {
                'n_steps': args.ppo_n_steps,
                'batch_size': args.ppo_batch_size,
                'n_epochs': args.ppo_n_epochs,
                'learning_rate': args.ppo_learning_rate,
                'ent_coef': args.ppo_ent_coef,
                'vf_coef': args.ppo_vf_coef,
                'max_grad_norm': args.ppo_max_grad_norm,
                'clip_range': args.ppo_clip_range,
                'clip_range_vf': args.ppo_clip_range_vf,
                'gae_lambda': args.ppo_gae_lambda,
                'normalize_advantage': args.ppo_normalize_advantage,
                'use_sde': args.ppo_use_sde,
                'sde_sample_freq': args.ppo_sde_sample_freq,
                'target_kl': args.ppo_target_kl,
                'policy_kwargs': args.policy_kwargs,
            }
            model = PPO('MultiInputPolicy', env, **common_hyperparams, **ppo_hyperparams)
        else:
            raise ValueError(f"Unsupported algorithm: {args.algo}")
        reset_num_timesteps = True

    best_dir, best_prefix = get_unique_directory(args.best_dir, f"{args.best_prefix}_seed_{seed}{pretrained}", "")
    best_model_path = os.path.join(best_dir, best_prefix)

    new_logger = configure(tensorboard_log_dir, format_strings=["stdout", "tensorboard"])
    for handler in new_logger.output_formats:
        if isinstance(handler, HumanOutputFormat):
            handler.max_length = 100
            
    # Save hyperparameters as JSON
    hyperparams = {
        'algo': args.algo,
        'common_hyperparams': str(common_hyperparams),
    }
    if args.algo == 'A2C':
        hyperparams.update(a2c_hyperparams)
    elif args.algo == 'PPO':
        hyperparams.update(ppo_hyperparams)

    # Add VecNormalize parameters to hyperparams
    vecnormalize_params = {
        'vecnorm_norm_obs': args.vecnorm_norm_obs,
        'vecnorm_norm_reward': args.vecnorm_norm_reward,
        'vecnorm_clip_obs': args.vecnorm_clip_obs,
        'vecnorm_clip_reward': args.vecnorm_clip_reward,
        'vecnorm_epsilon': args.vecnorm_epsilon,
        'vecnorm_norm_obs_keys': args.vecnorm_norm_obs_keys,
    }
    hyperparams['vecnormalize_params'] = vecnormalize_params
    
    hyperparams_dir, hyperparams_prefix = get_unique_directory(args.hyperparams_dir, f"{args.hyperparams_prefix}_seed_{seed}_hyperparameters", ".json")
    
    hyperparams_path = os.path.join(hyperparams_dir, f"{hyperparams_prefix}")
    with open(hyperparams_path, 'w') as f:
        json.dump({k: str(v) for k, v in hyperparams.items()}, f, indent=4)

    save_dir, save_prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{seed}{pretrained}", "")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=save_dir, name_prefix=save_prefix,
        save_vecnormalize=True,
        save_replay_buffer=True,
        verbose=args.verbose
    )

    # save_vec_normalize_callback = SaveVecNormalizeEvalCallback(
    #     vec_normalize_env=env,
    #     save_path=best_model_path, 
    # )
    
    stop_training_on_no_model_improvement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50,
                                                                             min_evals=50,
                                                                             verbose=args.verbose)
    
    # stop_training_on_reward_threshold = StopTrainingOnRewardThreshold(reward_threshold=1000, # max reward is 1000 for this environment
    #                                                                   verbose=args.verbose)
    
    # callback_on_new_best = CallbackList([stop_training_on_no_model_improvement, stop_training_on_reward_threshold, save_vec_normalize_callback])

    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=best_model_path,
        callback_on_new_best=stop_training_on_no_model_improvement,
        n_eval_episodes=10,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        deterministic=True,
        log_path=tensorboard_log_dir,
        render=False,
        verbose=args.verbose
    )

    info_logger_callback = InfoLoggerCallback()

    callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])

    if args.memory_monitor:
        import threading
        monitoring_thread = threading.Thread(target=monitor_memory_usage, args=(3,), daemon=True)
        monitoring_thread.start()

    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)

    env.close()

    if args.plot_reward_history:
        reward_prefix = reward_prefix.split(".")[0]
        dir, pref = get_unique_directory(args.reward_dir, f"{reward_prefix}_plot", '.png')
        plot_path = os.path.abspath(os.path.join(dir, pref))
        plot_reward_distribution(reward_save_path, plot_path)

    try:
        dir, prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{seed}_final", ".zip")
        final_save = os.path.join(dir, prefix)
        model.save(final_save)

        dir, prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{seed}_vec_normalize_final", ".pkl")
        final_vec_normalize = os.path.join(dir, prefix)
        env.save(final_vec_normalize)

        env = setup_environment(args, reward_save_path, eval=False)
        
        env = VecNormalize.load(final_vec_normalize, env)

        if args.algo == 'A2C':
            model = A2C.load(final_save, env=env)
        elif args.algo == 'PPO':
            model = PPO.load(final_save, env=env)

        print("Model and VecNormalize loaded successfully.")

        os.remove(final_save)
        os.remove(final_vec_normalize)

    except Exception as e:
        print(f"Error loading model: {e}")

def str_to_list(value):
    return value.split(',')

def load_hyperparams(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--hyperparams_path", type=str, default=None, help="Path to the hyperparameters YAML file")
    args = parser.parse_args()
    
    if args.hyperparams_path is None:
        args.hyperparams_path = os.path.abspath("scripts/hyperparams/hyperparams.yaml")

    hyperparams = load_hyperparams(args.hyperparams_path)

    # Convert dictionary to an argparse.Namespace object
    args = argparse.Namespace(**hyperparams)
            
    no_name = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env".replace('-', '_')

    if args.log_prefix is None:
        args.log_prefix = no_name
    if args.reward_prefix is None:
        args.reward_prefix = f"{no_name}_reward_data"
    if args.save_prefix is None:
        args.save_prefix = no_name
    if args.best_prefix is None:
        args.best_prefix = f"{no_name}_env_best"
    if args.hyperparams_prefix is None:
        args.hyperparams_prefix = no_name
    
    if args.algo == 'A2C':
        learning_rate = args.a2c_learning_rate
        net_arch = dict(pi=[args.a2c_net_arch_width] * args.a2c_net_arch_depth, vf=[args.a2c_net_arch_width] * args.a2c_net_arch_depth)
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[args.a2c_activation_fn]
        ortho_init = args.a2c_ortho_init
    elif args.algo == 'PPO':
        learning_rate = args.ppo_learning_rate
        net_arch = dict(pi=[args.ppo_net_arch_width] * args.ppo_net_arch_depth, vf=[args.ppo_net_arch_width] * args.ppo_net_arch_depth)
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[args.ppo_activation_fn]
        ortho_init = args.ppo_ortho_init
    
    # Adjust learning rate if schedule is linear
    if args.lr_schedule == "linear":
        args.learning_rate = linear_schedule(learning_rate)
    else:
        args.learning_rate = learning_rate
    

    # Define policy kwargs
    args.policy_kwargs = dict(
                            net_arch=net_arch,
                            activation_fn=activation_fn,
                            ortho_init=ortho_init,
                        )

    if args.seed is None:
        if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.lower() != 'none':
            raise ValueError("Must provide seed when loading from checkpoint. Choose -1 to begin training from random value")
        else:
            args.seed = generate_random_seeds(1)

    elif args.seed == "-1":
        args.seed = generate_random_seeds(2)
    else:
        args.seed = [int(s) for s in args.seed.strip('[]').split(',')]

    original_seed_list = args.seed

    for seed in original_seed_list:
        args.seed = seed
        main(args)
