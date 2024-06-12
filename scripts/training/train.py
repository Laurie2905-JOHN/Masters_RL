import os
import argparse
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure, HumanOutputFormat
from utils.process_data import get_data
from utils.train_utils import InfoLoggerCallback, SaveVecNormalizeEvalCallback, SaveVecNormalizeCallback
from utils.train_utils import generate_random_seeds, setup_environment, get_unique_directory, select_device, set_seed, monitor_memory_usage
from models.wrappers.common import RewardTrackingWrapper
from models.envs.env_working import SchoolMealSelection

# Main training function
def main(args, seed):
    device = select_device(args)
    set_seed(seed, device)

    print(f"Using device: {device}")

    # Load data required for environment setup
    ingredient_df = get_data()
    
    if args.plot_reward_history:
        reward_dir, reward_prefix = get_unique_directory(args.reward_dir, f"{args.reward_prefix}_seed_{seed}_env", '.json')
        reward_save_path = os.path.abspath(os.path.join(reward_dir, reward_prefix))
    else:
        reward_save_path = None

    # Create and normalize vectorized environments with shared reward dictionary
    env = setup_environment(args, seed, ingredient_df, args.reward_save_interval, reward_save_path)
    
    # Initialize or load the model
    if args.checkpoint_path and args.checkpoint_path.lower() != 'none':
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        print(f"Loading model from checkpoint: {checkpoint_path}")
        tensorboard_log_dir = os.path.join(args.log_dir, f"{args.log_prefix}_seed_{seed}")
        reset_num_timesteps = False
        if os.path.exists(f"{checkpoint_path}.zip"):
            pkl_path = f"{checkpoint_path}_vec_normalize.pkl"
            print(f"Loading VecNormalize from: {pkl_path}")
            env = VecNormalize.load(pkl_path, env)
            if args.algo == 'A2C':
                model = A2C.load(checkpoint_path, env=env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
            elif args.algo == 'PPO':
                model = PPO.load(checkpoint_path, env=env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
            else:
                raise ValueError(f"Unsupported algorithm: {args.algo}")
            
        else:
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return
        
    else:
        # Set up TensorBoard logging directory
        log_dir, log_prefix = get_unique_directory(args.log_dir, f"{args.log_prefix}_seed_{seed}", ".zip")
        tensorboard_log_dir = os.path.join(log_dir, log_prefix)
        
        # Choose the RL algorithm (A2C or PPO)
        if args.algo == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
        elif args.algo == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device, seed=seed)
        else:
            raise ValueError(f"Unsupported algorithm: {args.algo}")
        reset_num_timesteps = True
 
    best_dir, best_prefix = get_unique_directory(args.best_dir, f"{args.best_prefix}_seed_{seed}", ".zip")
    best_model_path = os.path.join(best_dir, best_prefix)
    
    # Configure logger for TensorBoard and stdout
    new_logger = configure(tensorboard_log_dir, format_strings=["stdout", "tensorboard"])

    # Accessing the current formatters to update max_length
    for handler in new_logger.output_formats:
        if isinstance(handler, HumanOutputFormat):
            handler.max_length = 50
        

    # Set up callbacks for saving checkpoints, evaluating models, and logging additional information
    checkpoint_callback = SaveVecNormalizeCallback(save_freq=args.save_freq, save_path=args.save_dir, name_prefix=f"{args.save_prefix}_seed_{seed}", vec_normalize_env=env)
    eval_callback = SaveVecNormalizeEvalCallback(vec_normalize_env=env, eval_env=env, best_model_save_path=best_model_path, eval_freq=args.eval_freq, log_path=tensorboard_log_dir, deterministic=True, render=False)
    info_logger_callback = InfoLoggerCallback()
    callback = CallbackList([checkpoint_callback, eval_callback, info_logger_callback])
    
    if args.memory_monitor:
        # Start the memory monitoring in a separate thread
        import threading
        monitoring_thread = threading.Thread(target=monitor_memory_usage, args=(3,), daemon=True)
        monitoring_thread.start()
    
    model.set_logger(new_logger)
    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
    
    # Save the final model and VecNormalize statistics
    dir, prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{seed}_final", ".zip")
    final_save = os.path.join(dir, prefix)
    model.save(final_save)
    
    dir, prefix = get_unique_directory(args.save_dir, f"{args.save_prefix}_seed_{seed}_vec_normalize_final", ".pkl")
    final_vec_normalize = os.path.join(dir, prefix)
    env.save(final_vec_normalize)

    # Access the underlying RewardTrackingWrapper for saving rewards
    if args.plot_reward_history:
        # Save reward distribution
        # Plotting data from all envs though as they agregate to the same file
        reward_prefix = reward_prefix.split(".")[0]
        dir, pref = get_unique_directory(args.reward_dir, f"{reward_prefix}_plot", '.png')
        plot_path = os.path.abspath(os.path.join(dir, pref))
        env.envs[0].plot_reward_distribution(reward_save_path, plot_path)
            
    del model, env

    # Load the model and VecNormalize to verify they have saved correctly
    try:
        env = setup_environment(args, seed, ingredient_df, reward_save_path)
        
        env = VecNormalize.load(final_vec_normalize, env)

        if args.algo == 'A2C':
            model = A2C.load(final_save, env=env)
        elif args.algo == 'PPO':
            model = PPO.load(final_save, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        
# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--env_name", type=str, default='SchoolMealSelection-v1', help="Name of the environment")
    parser.add_argument("--reward_metrics", type=list, default=['nutrients', 'groups', 'environment', 'cost', 'consumption'], help="Metrics to give reward for")
    parser.add_argument("--algo", type=str, choices=['A2C', 'PPO'], default='A2C', help="RL algorithm to use (A2C or PPO)")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--plot_reward_history", type=bool, default=True, help="Save and plot the reward history for the environment")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode for the environment")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="Total number of timesteps for training")
    parser.add_argument("--log_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'tensorboard')), help="Directory for tensorboard logs")
    parser.add_argument("--log_prefix", type=str, default=None, help="Filename for tensorboard logs")
    parser.add_argument("--save_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'checkpoints')), help="Directory to save models and checkpoints")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for saved model files")
    parser.add_argument("--best_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'best_models')), help="Directory for best model")
    parser.add_argument("--best_prefix", type=str, default=None, help="Prefix for saving best model")
    parser.add_argument("--reward_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'reward')), help="Directory to save reward data")
    parser.add_argument("--reward_prefix", type=str, default=None, help="Prefix for saved reward data")
    parser.add_argument("--save_freq", type=int, default=1000, help="Frequency of saving checkpoints")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency of evaluations")
    parser.add_argument("--seed", type=str, default="1", help="Random seed for the environment (use -1 for random, or multiple values for multiple seeds)")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device to use for training (cpu, cuda, or auto)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max episode steps")
    parser.add_argument("--memory_monitor", type=bool, default=True, help="Monitor memory usage during training")
    parser.add_argument("--reward_save_interval", type=int, default=2500, help="Number of timestep between saving reward data")
    
    args = parser.parse_args()
    
    metric_str = ""
    
    for i, val in enumerate(args.reward_metrics):
        if i == len(args.reward_metrics) - 1:
            metric_str += val
            break
        else:
            metric_str += val + "_"

    # Set default log and save directories and prefix if not provided
    if args.log_prefix is None:
        args.log_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env_{metric_str}"
        args.log_prefix = args.log_prefix.replace('-', '_')
    if args.reward_prefix is None:
        args.reward_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}envs_{metric_str}_reward_data"
        args.reward_prefix = args.reward_prefix.replace('-', '_')
    if args.save_prefix is None:
        args.save_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env_{metric_str}"
        args.save_prefix = args.save_prefix.replace('-', '_')
    if args.best_prefix is None:
        args.best_prefix = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env_best_{metric_str}"
        args.best_prefix = args.best_prefix.replace('-', '_')
        
    if args.seed is None:
        if args.checkpoint_path and args.checkpoint_path.lower() != 'none':
            raise ValueError("Must provide seed when loading from checkpoint. Choose -1 to begin training from random value")
        else:
            args.seed = generate_random_seeds(1)
    elif args.seed == "-1":
        args.seed = generate_random_seeds(1)
    else:
        args.seed = [int(s) for s in args.seed.strip('[]').split(',')]
        
    for seed in args.seed:
        main(args, seed)
        

