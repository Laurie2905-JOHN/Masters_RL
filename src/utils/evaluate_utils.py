import torch


def evaluate_agent(env, model, num_episodes=100):
    """
    Evaluates the agent in the given environment.

    Parameters:
    env (gym.Env): The environment to evaluate the agent in.
    model (BaseAlgorithm): The trained model to evaluate.
    num_episodes (int): The number of episodes to evaluate the agent.

    Returns:
    list: A list of total rewards for each episode.
    """
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    return total_rewards

def calculate_average_reward(total_rewards):
    """
    Calculates the average reward from a list of total rewards.

    Parameters:
    total_rewards (list): A list of total rewards from multiple episodes.

    Returns:
    float: The average reward.
    """
    return sum(total_rewards) / len(total_rewards)

