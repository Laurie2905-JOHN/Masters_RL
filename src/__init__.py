from models.envs.env_working import SchoolMealSelection

# Register the environment
from gymnasium.envs.registration import register


register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env_working:SchoolMealSelection',
    max_episode_steps=1000,  # Allow multiple steps per episode, adjust as needed
)