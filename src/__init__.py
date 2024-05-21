# Register the environment
from gymnasium.envs.registration import register

register(
    id='SimpleCalorieOnlyEnv-v0',
    entry_point='models.envs.env:SimpleCalorieOnlyEnv',
    max_episode_steps=1000,
)

print("SimpleCalorieOnlyEnv registered successfully.")