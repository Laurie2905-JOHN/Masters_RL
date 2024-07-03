import gymnasium as gym
import numpy as np
from models.envs.env import *

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from utils.process_data import get_data
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
class Args:
    render_mode = None
    num_envs = 1
    plot_reward_history = True
    max_episode_steps = 100
    verbose = 0
    action_scaling_factor = 50
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
    env_name = 'SchoolMealSelection-v2'
    initialization_strategy = 'zero'
    vecnorm_norm_obs_keys = ['current_selection_value', 'cost', 'consumption', 'co2_g', 'nutrients']
    reward_calculator_type = 'shaped'
    
args = Args()

def mask_fn(self) -> np.ndarray:
    """
    Generate an action mask indicating valid actions.
    """
    
    n_ingredients = self.env.get_wrapper_attr('n_ingredients')
    nsteps = self.env.get_wrapper_attr('nsteps')
    current_meal_plan, nonzero_indices, nonzero_values = self.get_current_meal_plan()
    self.get_metrics()
    action_mask = np.zeros(n_ingredients + 3, dtype=np.int8)

    # Cache references for faster access using env.get_wrapper_attr
    ingredient_group_count = self.env.get_wrapper_attr('ingredient_group_count')
    ingredient_group_count_targets = self.env.get_wrapper_attr('ingredient_group_count_targets')
    group_info = self.env.get_wrapper_attr('group_info')
    current_selection = self.env.get_wrapper_attr('current_selection')
    
    all_group_target_met = all(
        value == ingredient_group_count_targets[key]
        for key, value in ingredient_group_count.items()
    )

    for key, value in ingredient_group_count.items():
        target = ingredient_group_count_targets[key]
        indexes = group_info[key]['indexes']
        selected = [idx for idx in indexes if current_selection[idx] > 0]

        if target == 0:
            # If the target is zero, block all actions for these ingredients
            for idx in indexes:
                action_mask[idx + 3] = 0
        elif value >= target:
            # If the value equals or is greater than the target, hide all other ingredients in that group until no longer selected
            # If all groups not chosen yet enforce the condition of selecting from the groups before any other action
            for idx in indexes:
                if all_group_target_met:
                    action_mask[idx + 3] = 1 if idx in selected else 0
                else:
                    action_mask[idx + 3] = 0
        else:
            # If the value is less than the target, allow all ingredients of that group
            for idx in indexes:
                action_mask[idx + 3] = 1

    if all_group_target_met:
        if nsteps > 25:
            action_mask[:3] = [0, 1, 1]
        else:
            action_mask[:3] = [0, 0, 1]
    else:
        action_mask[:3] = [0, 0, 1]
    # print(action_mask)  # For debugging purposes
    return action_mask


env_kwargs = {
            "ingredient_df": args.ingredient_df,
            "max_ingredients": args.max_ingredients,
            "action_scaling_factor": args.action_scaling_factor,
            "render_mode": args.render_mode,
            "seed": args.seed,
            "verbose": args.verbose,
            "initialization_strategy": args.initialization_strategy
            }

env = gym.make(args.env_name, **env_kwargs)
# # Apply the TimeLimit wrapper to enforce a maximum number of steps per episode. Need to repeat this so if i want to experiment with different steps.
env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
env = Monitor(env)  # wrap it with monitor env again to explicitely take the change into account

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
tensorboard_log_dir = "saved_models/tensorboard/test_maskable"
model = MaskablePPO(MaskableMultiInputActorCriticPolicy,  env, tensorboard_log = tensorboard_log_dir, verbose=1)

callback = MaskableEvalCallback(
        eval_env=env,
        # best_model_save_path=best_model_path,
        # callback_on_new_best=stop_training_on_no_model_improvement,
        n_eval_episodes=5,
        eval_freq=1000,
        deterministic=True,
        log_path=tensorboard_log_dir,
        render=False,
        verbose=1
    )

model.learn(total_timesteps=1000000, callback=callback)
