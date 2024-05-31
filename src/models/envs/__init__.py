# Register the environment
from gymnasium.envs.registration import register

register(
    id='SimpleCalorieOnlyEnv-v0',
    entry_point='models.envs.env1:SimpleCalorieOnlyEnv',
    max_episode_steps=1,
)

register(
    id='CalorieOnlyEnv-v2',
    entry_point='models.envs.env2:CalorieOnlyEnv',
    max_episode_steps=100,  # Allow multiple steps per episode, adjust as needed
)

register(
    id='CalorieOnlyEnv-v2',
    entry_point='models.envs.env3:CalorieOnlyEnv',
    max_episode_steps=100,  # Allow multiple steps per episode, adjust as needed
)