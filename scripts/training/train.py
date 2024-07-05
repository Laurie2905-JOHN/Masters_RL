import os
import argparse
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure, HumanOutputFormat
from utils.process_data import get_data
from utils.train_utils import *
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from models.callbacks.callback import *

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

    # stop_training_on_no_model_improvement = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=50,
    #     min_evals=50,
    #     verbose=args.verbose
    # )
    
    save_vec_normalize = SaveVecNormalizeCallback(
        save_freq = args.save_freq,
        save_path = args.save_dir,
        name_prefix = args.save_prefix
    )

    eval_callback_class = MaskableEvalCallback if args.algo == "MASKED_PPO" else EvalCallback

    eval_callback = eval_callback_class(
        eval_env=env,
        best_model_save_path=best_model_path,
        callback_on_new_best=save_vec_normalize,
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

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--hyperparams_file", type=str, default="scripts/hyperparams/setup.yaml", help="Path to the hyperparameters YAML file")
    args = parser.parse_args()
    
    ALGO_YAML_MAP = {
    'A2C': 'scripts/hyperparams/a2c.yaml',
    'PPO': 'scripts/hyperparams/ppo.yaml',
    'MASKED_PPO': 'scripts/hyperparams/masked_ppo.yaml'
}

    VEC_NORMALIZE_YAML = 'scripts/hyperparams/vec_normalize.yaml'
    SETUP_YAML = 'setup.yaml'
    
    # Load setup file
    # setup_params = load_yaml(os.path.join(args.hyperparams_dir, SETUP_YAML))
    # Load setup file
    setup_params = load_yaml(args.hyperparams_file)
    
    # Select the YAML files based on the chosen algorithm
    yaml_file = ALGO_YAML_MAP.get(setup_params['algo'])
    if not yaml_file:
        raise ValueError(f"Unsupported algorithm: {setup_params['algo']}")
    
    if setup_params['env_name'] == "SchoolMealSelection-v0" and setup_params['algo'] == "MASKED_PPO":
        raise ValueError(f"Unsupported algorithm for environment")
    
    # Load hyperparameters from the selected YAML file
    algo_hyperparams_dir = yaml_file
    algo_hyperparams = load_yaml(algo_hyperparams_dir)

    # Load additional parameters
    vec_normalize_params = load_yaml(os.path.join(VEC_NORMALIZE_YAML))
    
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
            args.seed = generate_random_seeds(1)
    elif args.seed == "-1":
        args.seed = generate_random_seeds(1)
    else:
        args.seed = [int(s) for s in args.seed.strip('[]').split(',')]

    original_seed_list = args.seed

    # Train the model for each seed
    for seed in original_seed_list:
        args.seed = seed
        main(args)
