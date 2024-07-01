import gymnasium as gym
import numpy as np
from models.envs.env import SchoolMealSelection

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from utils.process_data import get_data

class Args:
    render_mode = None
    num_envs = 2
    plot_reward_history = True
    max_episode_steps = 100
    verbose = 2
    action_scaling_factor = 10
    memory_monitor = True
    gamma = 0.99
    max_ingredients = 6
    action_scaling_factor = 10
    reward_save_interval = 1000
    vecnorm_norm_obs = True
    vecnorm_norm_reward = True
    vecnorm_clip_obs = 10
    vecnorm_clip_reward = 10
    vecnorm_epsilon = 1e-8 
    vecnorm_norm_obs_keys = None
    ingredient_df = get_data("small_data.csv")
    seed = 10
    env_name = 'SchoolMealSelection-v1'
    initialization_strategy = 'perfect'
    vecnorm_norm_obs_keys = ['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
    reward_calculator_type = 'sparse'
    
args = Args()

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()



# Function to set up the environment
def setup_environment(args, eval=False):
    
    env_kwargs = {
                "ingredient_df": args.ingredient_df,
                "max_ingredients": args.max_ingredients,
                "action_scaling_factor": args.action_scaling_factor,
                "render_mode": args.render_mode,
                "seed": args.seed,
                "verbose": args.verbose,
                "initialization_strategy": args.initialization_strategy
                }
        
    def make_env():
        
        env = gym.make(args.env_name, **env_kwargs)
        # # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode. Need to repeat this so if i want to experiment with different steps.
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = Monitor(env)  # wrap it with monitor env again to explicitely take the change into account
  
        return env

    env = make_vec_env(make_env, n_envs=args.num_envs, seed=args.seed)

    if eval:
        return env

    return VecNormalize(
        env, 
        norm_obs=args.vecnorm_norm_obs, 
        norm_reward=args.vecnorm_norm_reward, 
        clip_obs=args.vecnorm_clip_obs, 
        clip_reward=args.vecnorm_clip_reward, 
        gamma=args.gamma, 
        epsilon=args.vecnorm_epsilon, 
        norm_obs_keys=args.vecnorm_norm_obs_keys
    )
    
env = setup_environment(args)

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.learn(timesteps=1000)


# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)