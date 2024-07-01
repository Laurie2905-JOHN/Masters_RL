from models.envs.env_working import SchoolMealSelection
from models.envs.env import SchoolMealSelection

# Register the environment
from gymnasium.envs.registration import register


register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env_working:SchoolMealSelection',
    max_episode_steps=100,  # Allow multiple steps per episode, adjust as needed
)

# Register the environment
register(
    id='SchoolMealSelection-v1',
    entry_point='models.envs.env:SchoolMealSelection',
    max_episode_steps=100
)