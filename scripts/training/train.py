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
from torch import nn

# Main training function
def main(args):
    
    device = select_device(args)
    
    set_seed(args.seed, device)

    print(f"Using device: {device}")

    # Load data required for environment setup
    args.ingredient_df = get_data()

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
            handler.max_length = 50

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
    
    stop_training_on_no_model_improvement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=(args.total_timesteps // args.eval_freq) * 0.1, # Stop training if no improvement in 10% of total training time
                                                                             min_evals=(args.total_timesteps // args.eval_freq) * 0.2, # Minimum 20% of total evals before callback starts
                                                                             verbose=args.verbose)
    
    # stop_training_on_reward_threshold = StopTrainingOnRewardThreshold(reward_threshold=1000, # max reward is 1000 for this environment
    #                                                                   verbose=args.verbose)
    
    # callback_on_new_best = CallbackList([stop_training_on_no_model_improvement, stop_training_on_reward_threshold, save_vec_normalize_callback])

    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=best_model_path,
        callback_on_new_best=stop_training_on_no_model_improvement,
        n_eval_episodes=15,
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
        
    # Save hyperparameters as JSON
    hyperparams = {
        'algo': args.algo,
        'common_hyperparams': common_hyperparams,
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
        json.dump(hyperparams, f, indent=4)

def str_to_list(value):
    return value.split(',')

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on an environment")
    parser.add_argument("--env_name", type=str, default='SchoolMealSelection-v1', help="Name of the environment")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max episode steps")
    parser.add_argument("--max_ingredients", type=int, default=6, help="Max number of ingredients in plan")
    parser.add_argument("--action_scaling_factor", type=int, default=20, help="Max number of ingredients in plan")
    parser.add_argument("--algo", type=str, choices=['A2C', 'PPO'], default='A2C', help="RL algorithm to use (A2C or PPO)")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--render_mode", type=str, default=None, help="Render mode for the environment")
    parser.add_argument("--total_timesteps", type=int, default=10000000, help="Total number of timesteps for training")
    parser.add_argument("--reward_metrics", type=str, default='nutrients,groups,environment,cost,consumption', help="Metrics to give reward for (comma-separated list)")
    parser.add_argument("--log_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'tensorboard')), help="Directory for tensorboard logs")
    parser.add_argument("--log_prefix", type=str, default=None, help="Filename for tensorboard logs")
    parser.add_argument("--save_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'checkpoints')), help="Directory to save models and checkpoints")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for saved model files")
    parser.add_argument("--save_freq", type=int, default=1000, help="Frequency of saving checkpoints")
    parser.add_argument("--best_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'best_models')), help="Directory for best model")
    parser.add_argument("--best_prefix", type=str, default=None, help="Prefix for saving best model")
    parser.add_argument("--hyperparams_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'hyperparams')), help="Directory for hyperparams")
    parser.add_argument("--hyperparams_prefix", type=str, default=None, help="Prefix for saving hyperparams")
    parser.add_argument("--reward_dir", type=str, default=os.path.abspath(os.path.join('saved_models', 'reward')), help="Directory to save reward data")
    parser.add_argument("--reward_prefix", type=str, default=None, help="Prefix for saved reward data")
    parser.add_argument("--reward_save_interval", type=int, default=20000, help="Number of timestep between saving reward data")
    parser.add_argument("--plot_reward_history", type=bool, default=True, help="Save and plot the reward history for the environment")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluations")
    parser.add_argument("--seed", type=str, default="-1", help="Random seed for the environment (use -1 for random, or multiple values for multiple seeds)")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto', help="Device to use for training (cpu, cuda, or auto)")
    parser.add_argument("--memory_monitor", type=bool, default=False, help="Monitor memory usage during training")
    parser.add_argument("--pretrained_checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount of future rewards")
    parser.add_argument("--lr_schedule", type=str, default='constant', choices=['constant', 'linear'], help="Learning rate schedule")

    # A2C specific hyperparameters
    parser.add_argument("--a2c_n_steps", type=int, default=5, help="The number of steps to run for each environment per update")
    parser.add_argument("--a2c_learning_rate", type=float, default=7e-4, help="The learning rate for the optimizer")
    parser.add_argument("--a2c_ent_coef", type=float, default=0.0, help="Entropy coefficient for the loss calculation")
    parser.add_argument("--a2c_vf_coef", type=float, default=0.5, help="Value function coefficient for the loss calculation")
    parser.add_argument("--a2c_max_grad_norm", type=float, default=0.5, help="The maximum value for gradient clipping")
    parser.add_argument("--a2c_rms_prop_eps", type=float, default=1e-5, help="Epsilon parameter for RMSProp optimizer")
    parser.add_argument("--a2c_use_rms_prop", type=bool, default=True, help="Whether to use RMSProp optimizer")
    parser.add_argument("--a2c_use_sde", type=bool, default=False, help="Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration")
    parser.add_argument("--a2c_sde_sample_freq", type=int, default=-1, help="Sample a new noise matrix every n steps when using gSDE")
    parser.add_argument("--a2c_normalize_advantage", type=bool, default=False, help="Whether to normalize or not the advantage")
    parser.add_argument("--a2c_gae_lambda", type=float, default=1, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.")
    parser.add_argument("--a2c_net_arch_width", type=int, default=512, help="Define policy network width")
    parser.add_argument("--a2c_net_arch_depth", type=int, default=2, help="Define policy network depth")
    parser.add_argument("--a2c_activation_fn", type=str, default='relu', choices=["tanh", "relu", "elu", "leaky_relu"], help="Activation function for policy network")
    parser.add_argument("--a2c_ortho_init", type=bool, default=True, help="Orthogonal initialization")

            
    # PPO specific hyperparameters
    parser.add_argument("--ppo_n_steps", type=int, default=2048, help="The number of steps to run for each environment per update")
    parser.add_argument("--ppo_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--ppo_n_epochs", type=int, default=10, help="Number of epoch when optimizing the surrogate loss")
    parser.add_argument("--ppo_learning_rate", type=float, default=3e-4, help="The learning rate for the optimizer")
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0, help="Entropy coefficient for the loss calculation")
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5, help="Value function coefficient for the loss calculation")
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="The maximum value for the gradient clipping")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--ppo_clip_range_vf", type=float, default=None, help="Clipping parameter for the value function")
    parser.add_argument("--ppo_gae_lambda", type=float, default=1.0, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
    parser.add_argument("--ppo_normalize_advantage", type=bool, default=False, help="Whether to normalize or not the advantage")
    parser.add_argument("--ppo_use_sde", type=bool, default=False, help="Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration")
    parser.add_argument("--ppo_sde_sample_freq", type=int, default=-1, help="Sample a new noise matrix every n steps when using gSDE")
    parser.add_argument("--ppo_rollout_buffer_class", default=None, help="Rollout buffer class to use")
    parser.add_argument("--ppo_rollout_buffer_kwargs", default=None, help="Keyword arguments to pass to the rollout buffer on creation")
    parser.add_argument("--ppo_target_kl", type=float, default=None, help="Limit the KL divergence between updates")
    parser.add_argument("--ppo_stats_window_size", type=int, default=100, help="Window size for the rollout logging")

    # VecNormalize parameters
    parser.add_argument("--vecnorm_gamma", type=bool, default=True, help="discount factor for VecNormalize")
    parser.add_argument("--vecnorm_norm_obs", type=bool, default=True, help="Whether to normalize observations or not")
    parser.add_argument("--vecnorm_norm_reward", type=bool, default=True, help="Whether to normalize rewards or not")
    parser.add_argument("--vecnorm_clip_obs", type=float, default=10.0, help="Max absolute value for observation")
    parser.add_argument("--vecnorm_clip_reward", type=float, default=10.0, help="Max absolute value for discounted reward")
    parser.add_argument("--vecnorm_epsilon", type=float, default=1e-8, help="To avoid division by zero")
    parser.add_argument("--vecnorm_norm_obs_keys", type=str, default=None, help="Which keys from observation dict to normalize")

    args = parser.parse_args()

    args.reward_metrics = str_to_list(args.reward_metrics)

    metric_str = ""

    for i, val in enumerate(args.reward_metrics):
        if i == len(args.reward_metrics) - 1:
            metric_str += val
            break
        else:
            metric_str += val + "_"
            
    no_name = f"{args.env_name}_{args.algo}_{args.total_timesteps}_{args.num_envs}env_{metric_str}".replace('-', '_')

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