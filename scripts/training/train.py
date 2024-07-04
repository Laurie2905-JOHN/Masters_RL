import os
import argparse
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure, HumanOutputFormat
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback, generate_random_seeds, setup_environment, get_unique_directory, select_device, set_seed, monitor_memory_usage, plot_reward_distribution, linear_schedule
import json
import yaml
from torch import nn
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

ALGO_YAML_MAP = {
    'A2C': 'a2c.yaml',
    'PPO': 'ppo.yaml',
    'MASKED_PPO': 'masked_ppo.yaml'
}

VEC_NORMALIZE_YAML = 'vec_normalize.yaml'
SETUP_YAML = 'setup.yaml'

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
# Ensure the specified directory exists, creating it if necessary
def ensure_dir_exists(directory, verbose=0):
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose > 1:
            print(f"Warning: Directory {directory} did not exist and was created.")

# Load a pretrained model if a checkpoint is specified
def load_model(args, env, tensorboard_log_dir, seed):
    checkpoint_path = os.path.abspath(args.pretrained_checkpoint_path)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    steps = checkpoint_name.split("_")[-2]
    pkl_name = f"{'_'.join(checkpoint_name.split('_')[:-2])}_vecnormalize_{steps}_steps.pkl"
    pkl_path = os.path.join(checkpoint_dir, pkl_name)

    if os.path.exists(pkl_path):
        print(f"Loading VecNormalize from: {pkl_path}")
        env = VecNormalize.load(pkl_path, env)
    else:
        print(f"VecNormalize file does not exist: {pkl_path}")
        return None

    model = {
        'A2C': A2C.load,
        'PPO': PPO.load
    }.get(args.algo, lambda *args, **kwargs: None)(checkpoint_path, env=env, verbose=args.verbose, tensorboard_log=tensorboard_log_dir, device=args.device, seed=seed)

    if model is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    return model

# Create a new model with the specified algorithm and hyperparameters
def create_model(args, env, tensorboard_log_dir, seed):
    common_hyperparams = {
        'verbose': args.verbose,
        'tensorboard_log': tensorboard_log_dir,
        'device': args.device,
        'seed': seed,
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'normalize_advantage': args.normalize_advantage,
        'gae_lambda': args.gae_lambda,
        'policy_kwargs': args.policy_kwargs,
    }

    if args.algo == 'A2C':
        algo_hyperparams = {
            'n_steps': args.n_steps,
            'rms_prop_eps': args.rms_prop_eps,
            'use_rms_prop': args.use_rms_prop,
            'use_sde': args.use_sde,
            'sde_sample_freq': args.sde_sample_freq,
        }
    elif args.algo == 'PPO':
        algo_hyperparams = {
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'clip_range': args.clip_range,
            'clip_range_vf': args.clip_range_vf,
            'target_kl': args.target_kl,
            'use_sde': args.use_sde,
            'sde_sample_freq': args.sde_sample_freq,
        }
    elif args.algo == 'MASKED_PPO':
        algo_hyperparams = {
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'clip_range': args.clip_range,
            'clip_range_vf': args.clip_range_vf,
            'target_kl': args.target_kl,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    model_class = {
        'A2C': A2C,
        'PPO': PPO,
        'MASKED_PPO': MaskablePPO
    }.get(args.algo, None)

    if model_class is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    return model_class('MultiInputPolicy', env, **common_hyperparams, **algo_hyperparams)

# Save hyperparameters to a JSON file
def save_hyperparams(hyperparams, args, seed):
    hyperparams_dir, hyperparams_prefix = get_unique_directory(args.hyperparams_dir, f"{args.hyperparams_prefix}_seed_{seed}_hyperparameters", ".json")
    hyperparams_path = os.path.join(hyperparams_dir, f"{hyperparams_prefix}")
    with open(hyperparams_path, 'w') as f:
        json.dump({k: str(v) for k, v in hyperparams.items()}, f, indent=4)

# Main training function
def main(args):
    device = select_device(args)  # Select the appropriate device (CPU or GPU)
    set_seed(args.seed, device)  # Set the seed for reproducibility
    args.device = device  # Store the device in the arguments

    print(f"Using device: {device}")

    # Load the data for environment setup
    try:
        args.ingredient_df = get_data(f"{args.data_name}.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure all necessary directories exist
    for directory in [args.reward_dir, args.log_dir, args.save_dir, args.best_dir, args.hyperparams_dir]:
        ensure_dir_exists(directory, args.verbose)

    reward_save_path = None
    if args.plot_reward_history:
        reward_dir, reward_prefix = get_unique_directory(args.reward_dir, f"{args.reward_prefix}_seed_{args.seed}_env", '.json')
        reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))

    env = setup_environment(args, reward_save_path, eval=False)  # Set up the training environment
    tensorboard_log_dir = os.path.join(args.log_dir, f"{args.log_prefix}_seed_{args.seed}")

    if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.lower() != 'none':
        model = load_model(args, env, tensorboard_log_dir, args.seed)  # Load the model from checkpoint
        reset_num_timesteps = False
    else:
        model = create_model(args, env, tensorboard_log_dir, args.seed)  # Create a new model
        reset_num_timesteps = True

    best_dir, best_prefix = get_unique_directory(args.best_dir, f"{args.best_prefix}_seed_{args.seed}", "")
    best_model_path = os.path.join(best_dir, best_prefix)

    new_logger = configure(tensorboard_log_dir, format_strings=["stdout", "tensorboard"])
    for handler in new_logger.output_formats:
        if isinstance(handler, HumanOutputFormat):
            handler.max_length = 100

    # Prepare hyperparameters dictionary for saving
    hyperparams = {
        'algo': args.algo,
        'common_hyperparams': {
            'verbose': args.verbose,
            'tensorboard_log': tensorboard_log_dir,
            'device': args.device,
            'seed': args.seed,
            'gamma': args.gamma,
        },
        'vecnormalize_params': {
            'vecnorm_norm_obs': args.vecnorm_norm_obs,
            'vecnorm_norm_reward': args.vecnorm_norm_reward,
            'vecnorm_clip_obs': args.vecnorm_clip_obs,
            'vecnorm_clip_reward': args.vecnorm_clip_reward,
            'vecnorm_epsilon': args.vecnorm_epsilon,
            'vecnorm_norm_obs_keys': args.vecnorm_norm_obs_keys,
        }
    }

    save_hyperparams(hyperparams, args, args.seed)

    save_dir, save_prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{args.seed}", "")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=save_dir, name_prefix=save_prefix,
        save_vecnormalize=True,
        save_replay_buffer=True,
        verbose=args.verbose
    )

    stop_training_on_no_model_improvement = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,
        min_evals=50,
        verbose=args.verbose
    )

    eval_callback_class = MaskableEvalCallback if args.algo == "MASKED_PPO" else EvalCallback

    eval_callback = eval_callback_class(
        eval_env=env,
        best_model_save_path=best_model_path,
        callback_on_new_best=stop_training_on_no_model_improvement,
        n_eval_episodes=5,
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
        # Save the final model and VecNormalize
        final_save_path = os.path.join(*get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{args.seed}_final", ".zip"))
        model.save(final_save_path)
        final_vec_normalize_path = os.path.join(*get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{args.seed}_vec_normalize_final", ".pkl"))
        env.save(final_vec_normalize_path)

        # Reload environment and model to test loading
        env = setup_environment(args, reward_save_path, eval=False)
        env = VecNormalize.load(final_vec_normalize_path, env)

        if args.algo == 'A2C':
            model = A2C.load(final_save_path, env=env)
        elif args.algo == 'PPO':
            model = PPO.load(final_save_path, env=env)
        elif args.algo == "MASKED_PPO":
            model = MaskablePPO.load(final_save_path, env=env)

        print("Model and VecNormalize loaded successfully.")
        os.remove(final_save_path)
        os.remove(final_vec_normalize_path)

    except Exception as e:
        print(f"Error loading model: {e}")

# Convert a comma-separated string into a list
def str_to_list(value):
    return value.split(',')

# Load hyperparameters from a YAML file
def load_hyperparams(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
def set_default_prefixes(args):
    # Function to set the default prefixes if not provided
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
    return args

def get_activation_fn(name):
    return {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }.get(name.lower(), nn.ReLU)
    
# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--hyperparams_dir", type=str, default="scripts/hyperparams", help="Path to the hyperparameters YAML file")
    args = parser.parse_args()

    # Load setup file
    setup_params = load_yaml(os.path.join(args.hyperparams_dir, SETUP_YAML))
    
    # Select the YAML files based on the chosen algorithm
    yaml_file = ALGO_YAML_MAP.get(setup_params['algo'])
    if not yaml_file:
        raise ValueError(f"Unsupported algorithm: {setup_params['algo']}")
    
    # Load hyperparameters from the selected YAML file
    algo_hyperparams_dir = os.path.join(args.hyperparams_dir, yaml_file)
    algo_hyperparams = load_yaml(algo_hyperparams_dir)

    # Load additional parameters
    vec_normalize_params = load_yaml(os.path.join(args.hyperparams_dir, VEC_NORMALIZE_YAML))
    
    # Combine all parameters
    hyperparams = {**algo_hyperparams, **vec_normalize_params, **setup_params}

    # Convert dictionary to an argparse.Namespace object
    args = argparse.Namespace(**hyperparams)
            
    # Set default prefixes if not provided
    args = set_default_prefixes(args)

    # Set learning rate and policy kwargs
    learning_rate = args.learning_rate 

    # Adjust learning rate if schedule is linear
    if args.lr_schedule == "linear":
        args.learning_rate = linear_schedule(learning_rate)
    else:
        args.learning_rate = learning_rate

    # Define policy kwargs
    args.policy_kwargs = dict(
        net_arch=dict(
            pi=[args.net_arch_width] * args.net_arch_depth,
            vf=[args.net_arch_width] * args.net_arch_depth
        ),
        activation_fn=get_activation_fn(args.activation_fn),
        ortho_init=args.ortho_init
    )

    # Generate random seeds if not provided
    if args.seed is None:
        if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.lower() != 'none':
            raise ValueError("Must provide seed when loading from checkpoint. Choose -1 to begin training from random value")
        else:
            args.seed = generate_random_seeds(2)
    elif args.seed == "-1":
        args.seed = generate_random_seeds(1)
    else:
        args.seed = [int(s) for s in args.seed.strip('[]').split(',')]

    original_seed_list = args.seed

    # Train the model for each seed
    for seed in original_seed_list:
        args.seed = seed
        main(args)
