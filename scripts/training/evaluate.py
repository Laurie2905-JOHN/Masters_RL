import os
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from utils.process_data import get_data
import models.envs.env1

def evaluate_agent(model_path, env, n_eval_episodes=10, render=False):
    """
    Evaluate the agent and return the mean reward and standard deviation.

    :param model_path: (str) Path to the saved model.
    :param env: (gym.Env) The environment to evaluate on.
    :param n_eval_episodes: (int) Number of episodes to evaluate over.
    :param render: (bool) Whether to render the environment.
    :return: (tuple) Mean reward and standard deviation of the reward.
    """
    # Load the model
    model = A2C.load(model_path)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render)

    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    return mean_reward, std_reward

if __name__ == '__main__':
    # Example usage
    save_dir = os.path.abspath("saved_models/checkpoints/")
    model_path = os.path.join(save_dir, "a2c_simple_calorie_env")
    ingredient_df = get_data()

    env = gym.make('SimpleCalorieOnlyEnv-v0', ingredient_df=ingredient_df, render_mode='human')

    # Evaluate the agent
    evaluate_agent(model_path, env, n_eval_episodes=10, render=True)
