import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json
from models.reward.reward import reward_nutrient_macro as calculate_reward
from models.envs.env_working import SchoolMealSelection

def run_episodes(env, num_episodes, steps_per_episode):
    successful_terminations = 0

    for episode in range(num_episodes):
        obs, info = env.reset()

        for step in range(steps_per_episode):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)

            if done:
                if 0 == info['termination_reason']:
                    successful_terminations += 1
                break

    return successful_terminations

def shooting_method(ingredient_df, num_episodes, steps_per_episode, low, high, tolerance=0.1, eval_runs=15):
    def evaluate_scale_factor(scale_factor):
        env = SchoolMealSelection(ingredient_df=ingredient_df, action_scaling_factor=scale_factor)
        successful_terminations = np.mean([run_episodes(env, num_episodes, steps_per_episode) for _ in range(eval_runs)])
        env.close()
        return successful_terminations

    best_scale_factor = low
    max_successful_terminations = 0
    scale_factor_results = {}

    while high - low > tolerance:
        mid = (low + high) / 2

        low_terminations = evaluate_scale_factor(low)
        mid_terminations = evaluate_scale_factor(mid)
        high_terminations = evaluate_scale_factor(high)

        scale_factor_results[low] = {"successful_terminations": low_terminations}
        scale_factor_results[mid] = {"successful_terminations": mid_terminations}
        scale_factor_results[high] = {"successful_terminations": high_terminations}

        if low_terminations > high_terminations:
            high = mid
            if low_terminations > max_successful_terminations:
                max_successful_terminations = low_terminations
                best_scale_factor = low
        else:
            low = mid
            if high_terminations > max_successful_terminations:
                max_successful_terminations = high_terminations
                best_scale_factor = high

        if mid_terminations > max_successful_terminations:
            max_successful_terminations = mid_terminations
            best_scale_factor = mid

        print(f"Low: {low}, Mid: {mid}, High: {high}, Best: {best_scale_factor}, Max Terminations: {max_successful_terminations}")

    return best_scale_factor, max_successful_terminations, scale_factor_results

if __name__ == '__main__':
    from utils.process_data import get_data

    ingredient_df = get_data()

    num_episodes = 10
    steps_per_episode = 1000
    initial_low = 10
    initial_high = 25

    best_scale_factor, max_successful_terminations, scale_factor_results = shooting_method(ingredient_df, num_episodes, steps_per_episode, initial_low, initial_high)
    print(f"Best Scale Factor: {best_scale_factor}, Max Successful Terminations: {max_successful_terminations}")

    with open('scale_factor_results.json', 'w') as json_file:
        json.dump(scale_factor_results, json_file, indent=4)
